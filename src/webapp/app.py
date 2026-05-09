"""FastAPI webapp: upload DICOM folder/zip → AI nodule detection → 3D viewer.

Run:  python app.py        (listens on 127.0.0.1:8081)
"""
import asyncio
import io
import json
import shutil
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from predict import (
    clean_mask,
    filter_subpleural,
    find_nodules,
    get_malignancy_model,
    get_model,
    get_swa_model,
    merge_nearby_nodules,
    predict_malignancy_for_nodules,
    predict_nodules,
    read_dicom_series,
    render_3d_html,
    render_nodule_thumb,
    segment_lung,
)
from clinical import (
    detect_nodule_type_from_hu,
    is_upper_lobe,
)
# lung_rads lives inside predict.py (was defined there for predict_malignancy_for_nodules)
from predict import lung_rads


WEBAPP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = WEBAPP_DIR / "results"
UPLOADS_DIR = WEBAPP_DIR / "uploads"
RESULTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("Warming up models on GPU...")
    get_model()
    get_swa_model()        # ensemble — best.pt + swa.pt
    get_malignancy_model()
    from predict import _get_lung_inferer
    try:
        _get_lung_inferer()
    except Exception as e:
        print(f"Lung inferer warmup failed: {e}")
    print("Ready. API at http://127.0.0.1:8081/api/* — frontend at http://localhost:3000")
    yield


app = FastAPI(title="Nodura API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve generated PNG thumbnails (and any other case files) at /results-files/<id>/...
from fastapi.staticfiles import StaticFiles
app.mount("/results-files", StaticFiles(directory=RESULTS_DIR), name="results-files")


def _list_cases():
    items = []
    for c in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not c.is_dir():
            continue
        meta_p = c / "meta.json"
        if not meta_p.exists():
            continue
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        items.append({
            "id": c.name,
            "name": m.get("name", c.name),
            "n_nodules": m.get("n_nodules", 0),
            "n_slices": m.get("n_slices", 0),
            "ts": m.get("timestamp", ""),
        })
    return items


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/cases")
def list_cases():
    return _list_cases()


# In-memory progress log per case_id. Each entry is (pct, message).
PROGRESS: dict[str, list[tuple[int, str]]] = {}


def _emit(case_id: str, pct: int, msg: str):
    PROGRESS.setdefault(case_id, []).append((pct, msg))
    print(f"[{case_id}] {pct}% {msg}", flush=True)


def _do_analyze(work: Path, dcm_paths: list, case_id: str, case_label: str):
    """Heavy synchronous work — runs in a thread to keep event loop free."""
    try:
        _emit(case_id, 5, f"Bắt đầu xử lý {len(dcm_paths)} file DICOM")
        t = {}
        t0 = time.time(); vol, voxel_sp = read_dicom_series(dcm_paths); t["read_dicom"] = time.time() - t0
        _emit(case_id, 15, f"Đọc DICOM xong ({t['read_dicom']:.1f}s) — volume {vol.shape}, spacing {voxel_sp}")

        t0 = time.time(); lung = segment_lung(vol); t["lung_seg"] = time.time() - t0
        _emit(case_id, 25, f"Phân vùng phổi xong ({t['lung_seg']:.1f}s) — {int(lung.sum()):,} voxel phổi")

        N = vol.shape[0]
        last_emit = [0.0]
        def progress_cb(i, total):
            now = time.time()
            if now - last_emit[0] > 0.5:
                pct = 25 + int(55 * i / max(total, 1))  # 25 -> 80 during AI
                _emit(case_id, pct, f"AI đang infer slice {i+1}/{total}")
                last_emit[0] = now
        t0 = time.time()
        prob_vol, pred = predict_nodules(vol, threshold=0.65, tta=True, ensemble=True,
                                         progress=progress_cb)
        t["ai_predict"] = time.time() - t0
        _emit(case_id, 80, f"AI inference xong ({t['ai_predict']:.1f}s, ensemble best+swa) — pred {int(pred.sum()):,} voxel")

        pred = (pred & lung).astype("uint8")
        _emit(case_id, 81, f"Lọc trong phổi — {int(pred.sum()):,} voxel")

        pred = clean_mask(pred, voxel_sp)
        _emit(case_id, 82, f"Clean mask (closing + opening) — {int(pred.sum()):,} voxel")

        t0 = time.time()
        # Pass prob_vol for tight (core>0.85) diameter measurement
        nodules, labeled = find_nodules(pred, voxel_sp, min_voxels=200, max_elongation=4.0,
                                        prob_volume=prob_vol, core_threshold=0.85)
        t["postprocess"] = time.time() - t0
        _emit(case_id, 84, f"Tìm nodule ({t['postprocess']:.1f}s) — {len(nodules)} candidate, đo size từ core (prob>0.85)")

        before = len(nodules)
        nodules = merge_nearby_nodules(nodules, voxel_sp, max_dist_mm=10.0)
        if before > len(nodules):
            _emit(case_id, 85, f"Merge {before - len(nodules)} nodule cùng vùng vật lý")

        before = len(nodules)
        nodules = filter_subpleural(nodules, lung, voxel_sp, min_dist_mm=2.0)
        if before > len(nodules):
            _emit(case_id, 86, f"Bỏ {before - len(nodules)} subpleural FP")

        # Per-nodule descriptors: type, lobe location, Lung-RADS by size
        t0 = time.time()
        n_total = vol.shape[0]
        for n in nodules:
            n["nodule_type"] = detect_nodule_type_from_hu(vol, n["bbox_zyx_voxel"], labeled, n["id"])
            n["upper_lobe"] = is_upper_lobe(n["centroid_zyx_voxel"][0], n_total)
            n["lung_rads"] = lung_rads(n["diameter_mm"])
            # Confidence = mean probability inside the core voxels (proxy for AI certainty)
            zmin, ymin, xmin, zmax, ymax, xmax = n["bbox_zyx_voxel"]
            blob_in_box = (labeled[zmin:zmax, ymin:ymax, xmin:xmax] == n["id"])
            prob_in_box = prob_vol[zmin:zmax, ymin:ymax, xmin:xmax]
            if blob_in_box.any():
                n["confidence"] = float(prob_in_box[blob_in_box].mean())
            else:
                n["confidence"] = 0.0
        # Sort by confidence (highest first) — radiologist reviews top suspects first
        nodules.sort(key=lambda n: -n["confidence"])
        # Renumber so #1 is most confident
        for new_id, n in enumerate(nodules, 1):
            n["id"] = new_id
        t["postprocess_extra"] = time.time() - t0
        _emit(case_id, 92, f"Sort theo confidence — #1 cao nhất {nodules[0]['confidence']*100:.0f}% nếu có" if nodules else "Không có nodule")

        t0 = time.time()
        html_3d = render_3d_html(
            lung, pred, voxel_sp, labeled, nodules,
            title=f"{case_label} — {len(nodules)} nodule(s) detected",
        )
        t["render_3d"] = time.time() - t0
        _emit(case_id, 98, f"Render 3D xong ({t['render_3d']:.1f}s)")

        case_dir = RESULTS_DIR / case_id
        case_dir.mkdir(exist_ok=True)
        (case_dir / "3d.html").write_text(html_3d, encoding="utf-8")

        # Render bbox thumbnails
        thumbs_dir = case_dir / "thumbs"
        for n in nodules:
            try:
                fname = f"nodule_{n['id']:04d}.png"
                render_nodule_thumb(vol, n, thumbs_dir / fname)
                n["thumb_url"] = f"/results-files/{case_id}/thumbs/{fname}"
            except Exception as e:
                print(f"  thumb gen failed for #{n['id']}: {e}", flush=True)
        _emit(case_id, 99, f"Render {len(nodules)} thumbnail")
        meta = {
            "id": case_id, "name": case_label or case_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_slices": int(vol.shape[0]),
            "n_nodules": len(nodules),
            "voxel_spacing_zyx_mm": [float(s) for s in voxel_sp],
            "lung_voxels": int(lung.sum()),
            "pred_voxels": int(pred.sum()),
            "nodules": nodules,
            "timing_sec": {k: float(v) for k, v in t.items()},
        }
        (case_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _emit(case_id, 100, "DONE")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _emit(case_id, -1, f"LỖI: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    """Save upload, schedule analysis in a thread, return case_id immediately."""
    case_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    work = UPLOADS_DIR / case_id
    work.mkdir()
    case_label = ""

    for f in files:
        data = await f.read()
        fname = (f.filename or "uploaded").replace("\\", "/")
        stem = fname.split("/")[-1]
        if not case_label:
            case_label = fname.split("/")[0] if "/" in fname else stem
        if stem.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(work)
            continue
        target = work / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    dcm_paths = list(work.rglob("*.dcm")) + list(work.rglob("*.DCM"))
    if not dcm_paths:
        dcm_paths = [
            p for p in work.rglob("*")
            if p.is_file() and p.suffix.lower() not in {".zip", ".json", ".txt"}
        ]
    if not dcm_paths:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Không tìm thấy file DICOM trong upload.")

    _emit(case_id, 1, f"Upload xong ({len(dcm_paths)} file). Đang chờ AI...")
    asyncio.create_task(asyncio.to_thread(_do_analyze, work, dcm_paths, case_id, case_label))
    return JSONResponse({"case_id": case_id, "name": case_label, "n_dicoms": len(dcm_paths)})


@app.get("/api/events/{case_id}")
async def events(case_id: str):
    """Server-Sent Events stream of progress messages for a case."""
    async def gen():
        sent = 0
        # Give it a moment in case the client subscribes before the task started
        for _ in range(20):
            if case_id in PROGRESS:
                break
            await asyncio.sleep(0.1)
        while True:
            msgs = PROGRESS.get(case_id, [])
            while sent < len(msgs):
                pct, text = msgs[sent]
                payload = json.dumps({"pct": pct, "msg": text})
                yield f"data: {payload}\n\n"
                sent += 1
                if pct == 100 or pct == -1:
                    return
            await asyncio.sleep(0.4)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/case/{case_id}")
def case_get(case_id: str):
    cdir = RESULTS_DIR / case_id
    meta_p = cdir / "meta.json"
    if not meta_p.exists():
        raise HTTPException(status_code=404, detail="Case không tồn tại.")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    html_3d = (cdir / "3d.html").read_text(encoding="utf-8")
    return {"meta": meta, "plot_html": html_3d}


@app.delete("/api/case/{case_id}")
def case_delete(case_id: str):
    cdir = RESULTS_DIR / case_id
    if cdir.exists():
        shutil.rmtree(cdir, ignore_errors=True)
    return {"deleted": case_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)

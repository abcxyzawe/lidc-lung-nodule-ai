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

import predict as _predict_module
from predict import (
    clean_mask,
    filter_subpleural,
    find_nodules,
    nms_merge_duplicates,
    predict_fpr_for_nodules,
    get_malignancy_model,
    get_model,
    get_swa_model,
    merge_nearby_nodules,
    predict_malignancy_for_nodules,
    predict_nodules,
    read_dicom_series,
    render_3d_html,
    render_nodule_thumb,
    render_nodule_thumb_gt,
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
            "model": m.get("model", "mine"),
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

# Allowlist for doctor verdict values (None = clear verdict)
ALLOWED_VERDICTS: frozenset = frozenset({"accept", "reject", "review"})


def _emit(case_id: str, pct: int, msg: str):
    PROGRESS.setdefault(case_id, []).append((pct, msg))
    # F-40: Windows-safe print — fallback if console encoding is not UTF-8 (e.g. cp1252)
    try:
        print(f"[{case_id}] {pct}% {msg}", flush=True)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(f"[{case_id}] {pct}% {safe}", flush=True)


def _build_synthetic_labeled(vol_shape, nodules):
    """Fill spherical masks inside each nodule bbox for visualization (MONAI/GT).
    Returns labeled volume (int) where nid==original_id."""
    import numpy as np
    labeled = np.zeros(vol_shape, dtype=np.int32)
    for n in nodules:
        nid = n.get("original_id", n["id"])
        zmin, ymin, xmin, zmax, ymax, xmax = n["bbox_zyx_voxel"]
        zmin = max(0, int(zmin))
        ymin = max(0, int(ymin))
        xmin = max(0, int(xmin))
        zmax = min(vol_shape[0], int(zmax))
        ymax = min(vol_shape[1], int(ymax))
        xmax = min(vol_shape[2], int(xmax))
        if zmax <= zmin or ymax <= ymin or xmax <= xmin:
            continue  # degenerate bbox, skip
        labeled[zmin:zmax, ymin:ymax, xmin:xmax] = nid
    return labeled


def _do_analyze(work: Path, dcm_paths: list, case_id: str, case_label: str, model: str = "mine"):
    """Heavy synchronous work. model: 'mine' | 'monai' | 'gt'."""
    try:
        _emit(case_id, 5, f"Bắt đầu xử lý {len(dcm_paths)} file DICOM [model={model}]")
        t = {}
        t0 = time.time(); vol, voxel_sp = read_dicom_series(dcm_paths); t["read_dicom"] = time.time() - t0
        _emit(case_id, 15, f"Đọc DICOM xong ({t['read_dicom']:.1f}s) — volume {vol.shape}, spacing {voxel_sp}")

        t0 = time.time(); lung = segment_lung(vol); t["lung_seg"] = time.time() - t0
        _emit(case_id, 25, f"Phân vùng phổi xong ({t['lung_seg']:.1f}s) — {int(lung.sum()):,} voxel phổi")

        # ---- Model branch ----
        if model == "monai":
            _emit(case_id, 30, "Chạy MONAI RetinaNet 3D (LUNA16 pretrained)...")
            from predict_monai import predict_nodules_monai
            t0 = time.time()
            nodules = predict_nodules_monai(vol, tuple(float(s) for s in voxel_sp))
            t["monai_predict"] = time.time() - t0
            _emit(case_id, 80, f"MONAI xong ({t['monai_predict']:.1f}s) — {len(nodules)} bbox")
            prob_vol = None
            pred = (lung & 0).astype("uint8")  # dummy empty
            labeled = _build_synthetic_labeled(vol.shape, nodules)
        elif model == "gt":
            _emit(case_id, 30, "Đang load ground truth từ LIDC XML...")
            from load_gt import load_gt_for_upload
            t0 = time.time()
            nodules = load_gt_for_upload(dcm_paths, vol.shape, tuple(float(s) for s in voxel_sp))
            t["gt_load"] = time.time() - t0
            if not nodules:
                _emit(case_id, 80, "Upload không match LIDC patient — không có GT")
            else:
                _emit(case_id, 80, f"GT loaded — {len(nodules)} nodule annotated")
            prob_vol = None
            pred = (lung & 0).astype("uint8")
            labeled = _build_synthetic_labeled(vol.shape, nodules)
        else:
            # ---- Original "mine" pipeline ----
            N = vol.shape[0]
            last_emit = [0.0]
            def progress_cb(i, total):
                now = time.time()
                if now - last_emit[0] > 0.5:
                    pct = 25 + int(55 * i / max(total, 1))
                    _emit(case_id, pct, f"AI đang infer slice {i+1}/{total}")
                    last_emit[0] = now
            t0 = time.time()
            prob_vol, pred = predict_nodules(vol, threshold=0.97, tta=True, ensemble=True,
                                             progress=progress_cb)
            t["ai_predict"] = time.time() - t0
            _emit(case_id, 80, f"AI inference xong ({t['ai_predict']:.1f}s, ensemble best+swa) — pred {int(pred.sum()):,} voxel")

            pred = (pred & lung).astype("uint8")
            _emit(case_id, 81, f"Lọc trong phổi — {int(pred.sum()):,} voxel")
            pred = clean_mask(pred, voxel_sp)
            _emit(case_id, 82, f"Clean mask — {int(pred.sum()):,} voxel")

            t0 = time.time()
            nodules, labeled = find_nodules(pred, voxel_sp, min_voxels=120, max_elongation=4.0,
                                            prob_volume=prob_vol, core_threshold=0.85)
            t["postprocess"] = time.time() - t0
            _emit(case_id, 84, f"Tìm nodule ({t['postprocess']:.1f}s) — {len(nodules)} candidate")

            before = len(nodules)
            nodules = merge_nearby_nodules(nodules, voxel_sp, max_dist_mm=10.0)
            if before > len(nodules):
                _emit(case_id, 85, f"Merge {before - len(nodules)} nodule cùng vùng vật lý")

            before = len(nodules)
            nodules = filter_subpleural(nodules, lung, voxel_sp, min_dist_mm=0.0)
            if before > len(nodules):
                _emit(case_id, 86, f"Bỏ {before - len(nodules)} subpleural FP")

            nodules = predict_fpr_for_nodules(vol, nodules)
            # _FPR_THR is set by get_fpr_model() side-effect during predict_fpr_for_nodules() call above
            # — read AFTER that call to get live ckpt value (default 0.5 → ckpt 0.85)
            fpr_thr = _predict_module._FPR_THR
            before_fpr = len(nodules)
            # fail-open: nodules with fpr_prob=None (model unavailable) are kept
            nodules = [n for n in nodules
                       if (n.get("fpr_prob") is None) or (n["fpr_prob"] >= fpr_thr)]
            _emit(case_id, 87, f"FPR filter: {len(nodules)}/{before_fpr} candidates (thr={fpr_thr:.2f})")

            before_nms = len(nodules)
            nodules = nms_merge_duplicates(nodules, voxel_sp)
            _emit(case_id, 88, f"NMS: {len(nodules)}/{before_nms} unique nodules")

        # ---- Common per-nodule descriptors (works for mine/monai/gt) ----
        t0 = time.time()
        n_total = vol.shape[0]
        for n in nodules:
            if model == "mine":
                n["nodule_type"] = detect_nodule_type_from_hu(vol, n["bbox_zyx_voxel"], labeled, n["id"])
            n["upper_lobe"] = is_upper_lobe(n["centroid_zyx_voxel"][0], n_total)
            n["lung_rads"] = lung_rads(n["diameter_mm"])
            zmin, ymin, xmin, zmax, ymax, xmax = n["bbox_zyx_voxel"]
            if model == "mine" and prob_vol is not None:
                blob_in_box = (labeled[zmin:zmax, ymin:ymax, xmin:xmax] == n["id"])
                prob_in_box = prob_vol[zmin:zmax, ymin:ymax, xmin:xmax]
                if blob_in_box.any():
                    n["confidence"] = float(prob_in_box[blob_in_box].mean())
                else:
                    n["confidence"] = 0.0
            # else: keep existing confidence (set by MONAI score or GT=1.0)
        # Sort by confidence (highest first) — radiologist reviews top suspects first
        # F-39: .get() defensive — monai/gt nodule schema may not have "confidence"
        nodules.sort(key=lambda n: -n.get("confidence", 0.0))
        # Renumber so #1 is most confident — keep original_id so labeled_pred lookups still work
        for new_id, n in enumerate(nodules, 1):
            n["original_id"] = n["id"]
            n["id"] = new_id
        t["postprocess_extra"] = time.time() - t0
        _emit(case_id, 92, f"Sort theo confidence — #1 cao nhất {nodules[0]['confidence']*100:.0f}% nếu có" if nodules else "Không có nodule")

        case_dir = RESULTS_DIR / case_id
        case_dir.mkdir(exist_ok=True)

        if model == "gt":
            # GT mode: skip slow 3D render — doctor just needs bbox images
            (case_dir / "3d.html").write_text(
                "<div style='padding:40px;text-align:center;color:#666;font-family:sans-serif'>"
                "Ground truth mode — chỉ hiển thị bbox 2D bên dưới.</div>",
                encoding="utf-8",
            )
            t["render_3d"] = 0.0
            _emit(case_id, 98, "GT mode — bỏ qua render 3D")
        else:
            t0 = time.time()
            html_3d = render_3d_html(
                lung, pred, voxel_sp, labeled, nodules,
                title=f"{case_label} — {len(nodules)} nodule(s) detected",
            )
            t["render_3d"] = time.time() - t0
            _emit(case_id, 98, f"Render 3D xong ({t['render_3d']:.1f}s)")
            (case_dir / "3d.html").write_text(html_3d, encoding="utf-8")

        # Render bbox thumbnails (3-slice montage for GT, single-slice for others)
        thumb_fn = render_nodule_thumb_gt if model == "gt" else render_nodule_thumb
        thumbs_dir = case_dir / "thumbs"
        for n in nodules:
            try:
                fname = f"nodule_{n['id']:04d}.png"
                thumb_fn(vol, n, thumbs_dir / fname)
                n["thumb_url"] = f"/results-files/{case_id}/thumbs/{fname}"
            except Exception as e:
                print(f"  thumb gen failed for #{n['id']}: {e}", flush=True)
        _emit(case_id, 99, f"Render {len(nodules)} thumbnail")

        # F-38: pred_voxels only meaningful for "mine" — monai/gt pred is dummy zeros
        pred_voxels = int(pred.sum()) if model == "mine" else None

        # ---- Under-prediction warning (mine only) ----
        model_confidence_warning = False
        low_pred_reason = None
        if model == "mine":
            THRESHOLD_LOW_PRED_VOXELS = 5000  # empirical: LIDC-IDRI-0754 had 951 → fail
            n_nod = len(nodules)
            low_voxels = pred_voxels < THRESHOLD_LOW_PRED_VOXELS
            no_candidate = n_nod == 0

            model_confidence_warning = no_candidate or low_voxels
            if no_candidate and low_voxels:
                low_pred_reason = "no_candidate_and_low_pred_voxels"
            elif no_candidate:
                low_pred_reason = "no_candidate"
            elif low_voxels:
                low_pred_reason = "low_pred_voxels"

            if model_confidence_warning:
                _emit(case_id, 95, f"⚠️ Mô hình under-predict (pred_voxels={pred_voxels}, n_nodules={n_nod}) — khuyến nghị tham khảo MONAI/GT")

        meta = {
            "id": case_id, "name": case_label or case_id,
            "model": model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_slices": int(vol.shape[0]),
            "n_nodules": len(nodules),
            "voxel_spacing_zyx_mm": [float(s) for s in voxel_sp],
            "lung_voxels": int(lung.sum()),
            "pred_voxels": pred_voxels,
            "model_confidence_warning": model_confidence_warning,
            "low_pred_reason": low_pred_reason,
            "nodules": nodules,
            "timing_sec": {k: float(v) for k, v in t.items()},
        }
        (case_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _emit(case_id, 100, "DONE")
    except FileNotFoundError:
        import traceback
        traceback.print_exc()
        _emit(case_id, -1, f"Thiếu file cần thiết cho mode {model}. Vui lòng kiểm tra cài đặt server.")
    except Exception:
        import traceback
        traceback.print_exc()
        _emit(case_id, -1, f"Lỗi không xác định khi xử lý mode {model}.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...), model: str = Form("mine")):
    """Save upload, schedule analysis in a thread, return case_id immediately.
    model: 'mine' | 'monai' | 'gt'
    """
    if model not in {"mine", "monai", "gt"}:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Pre-flight: verify required artifacts exist before spawning any thread.
    PROJECT_ROOT = WEBAPP_DIR.parent.parent
    if model == "monai":
        _bundle_ts = PROJECT_ROOT / "bundles" / "lung_nodule_ct_detection" / "models" / "model.ts"
        if not _bundle_ts.exists():
            raise HTTPException(
                status_code=503,
                detail="MONAI bundle chưa được cài. Vui lòng cài bundle hoặc chọn mode khác.",
            )
    elif model == "gt":
        _xml_dir = PROJECT_ROOT / "tcia-lidc-xml"
        if not _xml_dir.exists() or not any(_xml_dir.iterdir()):
            raise HTTPException(
                status_code=503,
                detail="Thiếu LIDC XML annotations. Vui lòng kiểm tra dataset hoặc chọn mode khác.",
            )

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
                work_resolved = work.resolve()
                for member in z.infolist():
                    # Block zip-slip: resolved member path must be inside work
                    member_target = (work / member.filename).resolve()
                    try:
                        if not member_target.is_relative_to(work_resolved):
                            continue  # silently skip malicious entries
                    except (ValueError, OSError):
                        continue
                    z.extract(member, work)
            continue
        target = work / fname
        # Block path traversal: target must stay inside work dir
        try:
            if not target.resolve().is_relative_to(work.resolve()):
                continue  # silently skip malicious paths
        except (ValueError, OSError):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    dcm_paths = sorted({p for p in work.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm"})
    if not dcm_paths:
        dcm_paths = [
            p for p in work.rglob("*")
            if p.is_file() and p.suffix.lower() not in {".zip", ".json", ".txt"}
        ]
    if not dcm_paths:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Không tìm thấy file DICOM trong upload.")

    _emit(case_id, 1, f"Upload xong ({len(dcm_paths)} file). Model={model}. Đang chờ AI...")
    asyncio.create_task(asyncio.to_thread(_do_analyze, work, dcm_paths, case_id, case_label, model))
    return JSONResponse({"case_id": case_id, "name": case_label, "n_dicoms": len(dcm_paths), "model": model})


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
    verdicts_p = cdir / "verdicts.json"
    verdicts = json.loads(verdicts_p.read_text(encoding="utf-8")) if verdicts_p.exists() else {}
    return {"meta": meta, "plot_html": html_3d, "verdicts": verdicts}


@app.post("/api/case/{case_id}/verdict")
async def case_verdict(case_id: str, payload: dict):
    """Persist doctor verdict for a single nodule.
    Body: {nodule_id: int, verdict: 'accepted'|'rejected'|null, reason?: str}
    """
    cdir = RESULTS_DIR / case_id
    if not (cdir / "meta.json").exists():
        raise HTTPException(status_code=404, detail="Case không tồn tại.")
    nid = payload.get("nodule_id")
    verdict = payload.get("verdict")  # may be None to clear
    reason = payload.get("reason", "")
    if nid is None:
        raise HTTPException(status_code=400, detail="nodule_id required")
    if verdict is not None and verdict not in ALLOWED_VERDICTS:
        raise HTTPException(status_code=400, detail="Verdict không hợp lệ. Phải là accept/reject/review hoặc null.")
    vp = cdir / "verdicts.json"
    data = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else {}
    if verdict is None:
        data.pop(str(nid), None)
    else:
        data[str(nid)] = {
            "verdict": verdict, "reason": reason,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    vp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "verdicts": data}


@app.delete("/api/case/{case_id}")
def case_delete(case_id: str):
    cdir = RESULTS_DIR / case_id
    if cdir.exists():
        shutil.rmtree(cdir, ignore_errors=True)
    return {"deleted": case_id}


@app.get("/api/training-metrics")
def training_metrics():
    from training_metrics import parse_stage2_log, parse_fpr_log, get_summary
    try:
        return {
            "stage2": parse_stage2_log(),
            "fpr": parse_fpr_log(),
            "summary": get_summary(),
        }
    except FileNotFoundError:
        raise HTTPException(503, detail="Training log not found. Please verify server configuration.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)

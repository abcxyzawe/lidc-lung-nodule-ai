"""Step 08 — Render 3D visualizations from detection artifacts.

Reads outputs/<pid>/<series>/{images.npy, pred_volume.npy, gt_volume.npy, meta.json}
and writes:
    3d_interactive.html        — Plotly Mesh3d, GT (red) + Pred (blue), browser viewable
    nodule_<i>.stl             — One STL file per detected nodule
    nodule_<i>_slices.png      — Axial/Sagittal/Coronal slices through centroid + overlay
    6views.png                 — 6 fixed-angle 3D screenshots
    summary.html               — index page linking everything

Run:  python 08_render_3d.py                # all patients in outputs/
      python 08_render_3d.py --patient LIDC-IDRI-0837
"""
import argparse
import json
from pathlib import Path

import numpy as np
from skimage import measure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from configs import OUTPUTS, HU_LO, HU_HI


def hu_to_gray(img):
    x = np.clip(img.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_LO) / (HU_HI - HU_LO)


def marching(mask_volume, voxel_spacing):
    """Returns (verts, faces) in mm or (None, None) if mask too small."""
    if mask_volume.sum() < 30:
        return None, None
    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask_volume.astype(np.float32), level=0.5, spacing=voxel_spacing,
        )
        return verts, faces
    except (ValueError, RuntimeError):
        return None, None


def export_stl(verts, faces, out_stl):
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=verts, faces=faces)
        m.export(out_stl)
    except ImportError:
        with open(out_stl, "w") as f:
            f.write("solid nodule\n")
            for tri in faces:
                v = verts[tri]
                n = np.cross(v[1] - v[0], v[2] - v[0])
                n = n / (np.linalg.norm(n) + 1e-9)
                f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n  outer loop\n")
                for vv in v:
                    f.write(f"    vertex {vv[0]} {vv[1]} {vv[2]}\n")
                f.write("  endloop\n  endfacet\n")
            f.write("endsolid nodule\n")


def render_html(meshes, out_html, title=""):
    import plotly.graph_objects as go
    fig = go.Figure()
    for verts, faces, name, color, opacity in meshes:
        if verts is None or len(verts) == 0: continue
        # Plotly: x/y/z — use (x_mm, y_mm, z_mm); our verts are (z, y, x)
        fig.add_trace(go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=opacity, name=name, showlegend=True,
            flatshading=True,
        ))
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data",
                   xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                   bgcolor="rgb(20,20,28)"),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgb(20,20,28)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def render_6views(meshes, out_png, title=""):
    angles = [(20, 30), (20, 120), (20, 210), (20, 300), (75, 30), (-30, 30)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("#15151c")
    all_verts = [v for v, *_ in meshes if v is not None and len(v) > 0]
    if not all_verts:
        plt.close(fig); return
    allv = np.concatenate(all_verts)
    mn, mx = allv.min(0), allv.max(0)
    for ax, (elev, az) in zip(axes.flat, angles):
        ax.set_facecolor("#15151c")
        for verts, faces, name, color, op in meshes:
            if verts is None or len(verts) == 0: continue
            poly = Poly3DCollection(verts[faces], alpha=op,
                                    facecolor=color, edgecolor="none")
            ax.add_collection3d(poly)
        ax.set_xlim(mn[2], mx[2]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[0], mx[0])
        ax.view_init(elev=elev, azim=az)
        ax.set_title(f"el={elev}° az={az}°", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color("white")
    fig.suptitle(title, color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=90, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_slice_views(images, gt_mask, pred_mask, centroid_zyx, voxel_sp,
                       out_png, title=""):
    """3-panel: axial/sagittal/coronal through centroid, with mask overlays."""
    z, y, x = [int(c) for c in centroid_zyx]
    z = np.clip(z, 0, images.shape[0] - 1)
    y = np.clip(y, 0, images.shape[1] - 1)
    x = np.clip(x, 0, images.shape[2] - 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#15151c")

    panels = [
        ("Axial (z=%d)" % z, hu_to_gray(images[z]),
         gt_mask[z], pred_mask[z], (voxel_sp[2], voxel_sp[1])),
        ("Coronal (y=%d)" % y, hu_to_gray(images[:, y, :]),
         gt_mask[:, y, :], pred_mask[:, y, :], (voxel_sp[2], voxel_sp[0])),
        ("Sagittal (x=%d)" % x, hu_to_gray(images[:, :, x]),
         gt_mask[:, :, x], pred_mask[:, :, x], (voxel_sp[1], voxel_sp[0])),
    ]
    for ax, (label, im, gtm, prm, asp) in zip(axes, panels):
        ax.imshow(im, cmap="gray", aspect=asp[1] / asp[0])
        if gtm.any():
            gt_overlay = np.zeros((*gtm.shape, 4))
            gt_overlay[gtm > 0] = [1, 0, 0, 0.4]
            ax.imshow(gt_overlay, aspect=asp[1] / asp[0])
        if prm.any():
            pr_overlay = np.zeros((*prm.shape, 4))
            pr_overlay[prm > 0] = [0, 0.6, 1, 0.4]
            ax.imshow(pr_overlay, aspect=asp[1] / asp[0])
        ax.set_title(label, color="white", fontsize=10)
        ax.axis("off")
    fig.suptitle(title + "  GT=red, Pred=blue", color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def process_series(case_dir):
    """case_dir = outputs/<pid>/<series_short>/"""
    meta = json.loads((case_dir / "meta.json").read_text())
    images = np.load(case_dir / "images.npy")
    pred = np.load(case_dir / "pred_volume.npy")
    gt = np.load(case_dir / "gt_volume.npy")
    voxel_sp = tuple(meta["voxel_spacing_zyx_mm"])
    pid = meta["patient_id"]; nodules = meta["predicted_nodules"]

    print(f"\n[{pid} / {case_dir.name}] vol {images.shape}, "
          f"GT vox {int(gt.sum())}, Pred vox {int(pred.sum())}, "
          f"{len(nodules)} predicted nodules")

    # Whole-volume meshes (GT + Pred)
    gt_v, gt_f = marching(gt, voxel_sp)
    pred_v, pred_f = marching(pred, voxel_sp)
    meshes_whole = []
    if gt_v is not None: meshes_whole.append((gt_v, gt_f, "GT", "#ff4444", 0.45))
    if pred_v is not None: meshes_whole.append((pred_v, pred_f, "Pred", "#4ab5ff", 0.45))

    if meshes_whole:
        render_html(meshes_whole, case_dir / "3d_interactive.html",
                    title=f"{pid} / {case_dir.name} — Lung nodules 3D")
        render_6views(meshes_whole, case_dir / "6views.png",
                      title=f"{pid} / {case_dir.name}")

    # Per-nodule STL + slice views
    for nod in nodules:
        nid = nod["id"]
        # Isolate this nodule's mask
        zmin, ymin, xmin, zmax, ymax, xmax = nod["bbox_zyx_voxel"]
        nod_mask = np.zeros_like(pred)
        # Re-find the connected blob for this id by intersecting bbox + label match
        from scipy.ndimage import label as cc_label
        sub = pred[zmin:zmax, ymin:ymax, xmin:xmax]
        if sub.sum() == 0: continue
        labeled, _ = cc_label(sub, structure=np.ones((3, 3, 3), dtype=np.uint8))
        # Pick largest blob inside bbox (this should be the nodule)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        if sizes.max() == 0: continue
        keep_id = sizes.argmax()
        nod_mask[zmin:zmax, ymin:ymax, xmin:xmax] = (labeled == keep_id).astype(np.uint8)

        v, f = marching(nod_mask, voxel_sp)
        if v is not None:
            export_stl(v, f, case_dir / f"nodule_{nid:02d}.stl")

        render_slice_views(images, gt, nod_mask, nod["centroid_zyx_voxel"],
                           voxel_sp, case_dir / f"nodule_{nid:02d}_slices.png",
                           title=(f"{pid} nodule #{nid}: {nod['voxels']} vox, "
                                  f"{nod['volume_mm3']:.0f} mm³, "
                                  f"~{nod['diameter_mm']:.1f} mm diameter"))

    # Summary HTML for this series
    rows = "".join(
        f"<tr><td>{n['id']}</td><td>{n['voxels']}</td>"
        f"<td>{n['volume_mm3']:.0f}</td><td>{n['diameter_mm']:.2f}</td></tr>"
        for n in nodules
    ) or "<tr><td colspan=4>No nodules detected.</td></tr>"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>{pid} / {case_dir.name}</title>
<style>
body {{ font-family: sans-serif; background:#15151c; color:#eee; padding:20px }}
table {{ border-collapse: collapse; margin: 10px 0 }}
td, th {{ padding: 6px 12px; border: 1px solid #555 }}
img {{ max-width: 100%; margin: 10px 0; border:1px solid #444 }}
a {{ color: #4ab5ff }}
</style></head><body>
<h1>{pid} / {case_dir.name}</h1>
<p>Volume: {images.shape}, voxel spacing: {tuple(round(s,2) for s in voxel_sp)} mm.</p>
<p>GT voxels: {int(gt.sum())}, Predicted voxels: {int(pred.sum())},
   Nodules detected: {len(nodules)}</p>
<h3>Detected nodules</h3>
<table>
<tr><th>ID</th><th>Voxels</th><th>Volume (mm³)</th><th>Diameter (mm)</th></tr>
{rows}
</table>
<h3>Interactive 3D</h3>
<p><a href='3d_interactive.html'>Open 3d_interactive.html</a></p>
<h3>6-view 3D collage</h3>
<img src='6views.png'/>
<h3>Per-nodule slice views</h3>
{''.join(f"<h4>Nodule {n['id']}</h4><img src='nodule_{n['id']:02d}_slices.png'/>" for n in nodules)}
</body></html>"""
    (case_dir / "report.html").write_text(html, encoding="utf-8")
    print(f"  -> wrote 3d_interactive.html, 6views.png, report.html, "
          f"{len(nodules)} STL + slice PNGs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="", help="Render only this patient")
    args = ap.parse_args()

    patient_dirs = sorted([p for p in OUTPUTS.iterdir() if p.is_dir()])
    if args.patient:
        patient_dirs = [p for p in patient_dirs if p.name == args.patient]

    cases = []
    for pd in patient_dirs:
        for sd in pd.iterdir():
            if sd.is_dir() and (sd / "meta.json").exists():
                cases.append(sd)
    print(f"Rendering {len(cases)} series across {len(patient_dirs)} patients")
    for cd in cases:
        try:
            process_series(cd)
        except Exception as e:
            print(f"ERROR {cd}: {e}")

    # Top-level index
    items = []
    for cd in cases:
        meta = json.loads((cd / "meta.json").read_text())
        items.append(
            f"<li><a href='{cd.relative_to(OUTPUTS).as_posix()}/report.html'>"
            f"{meta['patient_id']} / {cd.name}</a> — "
            f"{meta['n_predicted_nodules']} nodules</li>"
        )
    (OUTPUTS / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>LIDC 3D Reports</title>"
        "<body style='font-family:sans-serif;background:#15151c;color:#eee;padding:20px'>"
        f"<h1>LIDC 3D Reports ({len(cases)} cases)</h1><ul>"
        + "".join(items) + "</ul></body>",
        encoding="utf-8",
    )
    print(f"\nDone. Open {OUTPUTS / 'index.html'} in browser.")


if __name__ == "__main__":
    main()

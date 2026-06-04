#!/usr/bin/env python3
"""Stage I2: WiLoR hand mesh + RGB-D depth alignment.

This replaces the DexYCB MANO debug hand source with an image frontend:
RGB -> WiLoR predicted MANO params -> MANO mesh -> projection-preserving RGB-D depth fitting -> reference camera frame.

Alignment rule:
- use WiLoR full-image 2D projections from its demo camera model
- estimate a robust hand target depth from RGB-D pixels inside the selected hand bbox,
  excluding the current visible object mask when available
- shift WiLoR camera-space z to the target median depth
- unproject the WiLoR 2D projections with the real RGB-D intrinsics at the fitted z
This is a minimal MANO-param-preserving RGB-D fit: MANO pose/shape are kept from WiLoR,
and only the global camera-depth translation is fitted to RGB-D. It avoids blindly scaling xyz by depth ratio.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


def rel_to_root(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def load_intrinsics(path: Path) -> dict:
    return yaml.unsafe_load(path.read_text())["color"]


def load_depth_m(path: Path) -> np.ndarray:
    raw = np.array(Image.open(path))
    return raw.astype(np.float32) / 1000.0


def project_real(points: np.ndarray, intr: dict) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-6)
    u = points[:, 0] / z * float(intr["fx"]) + float(intr["ppx"])
    v = points[:, 1] / z * float(intr["fy"]) + float(intr["ppy"])
    return np.stack([u, v], axis=1)


def unproject_uv_depth(uv: np.ndarray, z: np.ndarray, intr: dict) -> np.ndarray:
    x = (uv[:, 0] - float(intr["ppx"])) / float(intr["fx"]) * z
    y = (uv[:, 1] - float(intr["ppy"])) / float(intr["fy"]) * z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def project_wilor_full(points: np.ndarray, cam_trans: np.ndarray, focal_length: float, img_size_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = points + cam_trans[None, :]
    z = np.maximum(pts[:, 2], 1e-6)
    u = pts[:, 0] / z * focal_length + img_size_xy[0] / 2.0
    v = pts[:, 1] / z * focal_length + img_size_xy[1] / 2.0
    return np.stack([u, v], axis=1), pts


def write_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, color=(50, 160, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        r, g, b = color
        for v in vertices:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {r} {g} {b}\n")
        for tri in faces.astype(int):
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def overlay(rgb_path: Path, verts_ref: np.ndarray, joints_ref: np.ndarray, bbox: np.ndarray, intr: dict, out_path: Path) -> None:
    img = Image.open(rgb_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox.tolist()
    draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0, 240), width=2)
    uv = project_real(verts_ref, intr)
    valid = (verts_ref[:, 2] > 0) & np.isfinite(uv).all(axis=1)
    idx = np.where(valid)[0]
    for i in idx[::max(1, len(idx)//1200)]:
        x, y = uv[i]
        if 0 <= x < img.width and 0 <= y < img.height:
            draw.ellipse((x-1, y-1, x+1, y+1), fill=(0, 140, 255, 180))
    juv = project_real(joints_ref, intr)
    for j, (x, y) in enumerate(juv):
        if 0 <= x < img.width and 0 <= y < img.height:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 255, 0, 230))
            draw.text((x+4, y+2), str(j), fill=(255, 255, 0, 230))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)


def bbox_depth_stats(depth_m: np.ndarray, bbox: np.ndarray, exclude_mask: np.ndarray | None, margin: int = 0) -> tuple[np.ndarray, dict]:
    h, w = depth_m.shape
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1-margin), max(0, y1-margin)
    x2, y2 = min(w, x2+margin), min(h, y2+margin)
    region = np.zeros_like(depth_m, dtype=bool)
    region[y1:y2, x1:x2] = True
    valid = region & (depth_m > 0) & np.isfinite(depth_m)
    if exclude_mask is not None:
        valid = valid & (~exclude_mask)
    vals = depth_m[valid]
    if vals.size < 50 and exclude_mask is not None:
        valid = region & (depth_m > 0) & np.isfinite(depth_m)
        vals = depth_m[valid]
    stats = {
        "pixels": int(vals.size),
        "median_m": float(np.median(vals)) if vals.size else None,
        "p25_m": float(np.percentile(vals, 25)) if vals.size else None,
        "p75_m": float(np.percentile(vals, 75)) if vals.size else None,
    }
    return vals, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--config", default="configs/contactmap_cjq_mug_capture_001.yaml")
    ap.add_argument("--wilor-root", default="/home/originflow/project/WiLoR")
    ap.add_argument("--wilor-ckpt", default="pretrained_models/wilor_final.ckpt")
    ap.add_argument("--wilor-cfg", default="pretrained_models/model_config.yaml")
    ap.add_argument("--detector", default="pretrained_models/detector.pt")
    ap.add_argument("--object-mask", default="outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001/sam_mask_visible_raw.png")
    ap.add_argument("--object-npz", default="outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001/foundationpose_posed_cad_object_mesh_ref.npz")
    ap.add_argument("--out-dir", default="outputs/wilor_depth_aligned_hand_mesh_single_frame/contactmap_cjq_mug_capture_001_frame000001")
    ap.add_argument("--rescale-factor", type=float, default=2.0)
    args = ap.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    wilor_root = Path(args.wilor_root).resolve()
    sys.path.insert(0, str(wilor_root))

    from ultralytics import YOLO
    from wilor.models import load_wilor
    from wilor.utils import recursive_to
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils.renderer import cam_crop_to_full

    cfg = yaml.safe_load(rel_to_root(root, args.config).read_text())
    li = cfg["local_inputs"]
    rgb_path = rel_to_root(root, li["rgb"])
    depth_path = rel_to_root(root, li["depth"])
    intr_path = rel_to_root(root, li["intrinsics_yml"])
    object_mask_path = rel_to_root(root, args.object_mask)
    object_npz_path = rel_to_root(root, args.object_npz)
    out_dir = rel_to_root(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intr = load_intrinsics(intr_path)
    depth_m = load_depth_m(depth_path)
    exclude_mask = np.array(Image.open(object_mask_path).convert("L")) > 0 if object_mask_path.exists() else None
    img_cv2 = cv2.imread(str(rgb_path))
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    h, w = img_cv2.shape[:2]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # WiLoR config contains relative MANO paths such as ./mano_data; load from WiLoR root.
    import os
    old_cwd = os.getcwd()
    os.chdir(str(wilor_root))
    try:
        model, model_cfg = load_wilor(checkpoint_path=str(wilor_root / args.wilor_ckpt), cfg_path=str(wilor_root / args.wilor_cfg))
    finally:
        os.chdir(old_cwd)
    # Ultralytics detector.pt is an older trusted local checkpoint; PyTorch>=2.6 defaults
    # torch.load(weights_only=True), which rejects its PoseModel pickle. Restore legacy load
    # only for this local detector construction.
    _orig_torch_load = torch.load
    def _torch_load_legacy(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_torch_load(*a, **kw)
    torch.load = _torch_load_legacy
    try:
        detector = YOLO(str(wilor_root / args.detector))
    finally:
        torch.load = _orig_torch_load
    model = model.to(device).eval()
    detector = detector.to(device)

    detections = detector(img_cv2, conf=0.25, verbose=False)[0]
    candidates = []
    for det in detections:
        data = det.boxes.data.detach().cpu().numpy().reshape(-1)
        if data.size < 6:
            continue
        bbox = data[:4].astype(np.float32)
        conf = float(data[4])
        cls = float(data[5])
        vals, stats = bbox_depth_stats(depth_m, bbox, exclude_mask)
        area = float(max(0, bbox[2]-bbox[0]) * max(0, bbox[3]-bbox[1]))
        candidates.append({"bbox": bbox, "conf": conf, "cls": cls, "depth_stats": stats, "area": area})
    if not candidates:
        raise RuntimeError("WiLoR detector found no hand candidates")

    # Prefer right-hand class when available; otherwise highest confidence.
    # WiLoR demo treats class value as is_right indicator.
    right_candidates = [c for c in candidates if c["cls"] >= 0.5]
    selected = max(right_candidates or candidates, key=lambda c: c["conf"])
    boxes = np.stack([selected["bbox"]])
    right = np.array([1.0 if selected["cls"] >= 0.5 else selected["cls"]], dtype=np.float32)

    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor, fp16=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    pred = None
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        multiplier = (2 * batch["right"] - 1)
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = float((model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()).detach().cpu().numpy())
        cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)[0].detach().cpu().numpy()
        verts = out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32)
        joints = out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32)
        # WiLoR returns MANO rotation matrices and shape parameters; pred_vertices/joints
        # are generated from these params by model.mano in WiLoR.forward_step().
        pred_mano_params = {
            k: v[0].detach().cpu().numpy().astype(np.float32)
            for k, v in out["pred_mano_params"].items()
        }
        pred_cam_raw = out["pred_cam"][0].detach().cpu().numpy().astype(np.float32)
        pred_cam_t_crop = out["pred_cam_t"][0].detach().cpu().numpy().astype(np.float32)
        is_right = float(batch["right"][0].detach().cpu().numpy())
        verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
        joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
        pred = {
            "verts": verts,
            "joints": joints,
            "faces": model.mano.faces.astype(np.int32),
            "mano_params": pred_mano_params,
            "pred_cam_raw": pred_cam_raw,
            "pred_cam_t_crop": pred_cam_t_crop,
            "cam_t": cam_t,
            "scaled_focal_length": scaled_focal_length,
            "img_size_xy": img_size[0].detach().cpu().numpy(),
            "is_right": np.array(is_right, dtype=np.float32),
        }
        break
    if pred is None:
        raise RuntimeError("No WiLoR prediction produced")

    verts_uv, verts_wilor_cam = project_wilor_full(pred["verts"], pred["cam_t"], pred["scaled_focal_length"], pred["img_size_xy"])
    joints_uv, joints_wilor_cam = project_wilor_full(pred["joints"], pred["cam_t"], pred["scaled_focal_length"], pred["img_size_xy"])
    vals, selected_depth_stats = bbox_depth_stats(depth_m, selected["bbox"], exclude_mask)
    if vals.size == 0:
        raise RuntimeError("No valid RGB-D depth pixels for selected hand bbox")
    target_depth = float(np.median(vals))
    z_shift = target_depth - float(np.median(verts_wilor_cam[:, 2]))
    verts_z = verts_wilor_cam[:, 2] + z_shift
    joints_z = joints_wilor_cam[:, 2] + z_shift
    verts_ref = unproject_uv_depth(verts_uv, verts_z, intr)
    joints_ref = unproject_uv_depth(joints_uv, joints_z, intr)

    object_geom = np.load(object_npz_path)
    if "object_vertices_refined_ref" in object_geom:
        object_vertices = object_geom["object_vertices_refined_ref"].astype(np.float32)
        object_source_label = "SAM_visible_mask_CAD_depth_refined_stage_h"
    else:
        object_vertices = object_geom["object_vertices_ref"].astype(np.float32)
        object_source_label = "FoundationPose_posed_CAD_stage_k"
    object_faces = object_geom["object_faces"].astype(np.int32)

    hand_ply = out_dir / "wilor_hand_mesh_depth_aligned_ref.ply"
    object_ply = out_dir / "object_mesh_ref_copy.ply"
    npz_path = out_dir / "wilor_hand_mesh_depth_aligned_ref.npz"
    overlay_png = out_dir / "rgb_wilor_hand_depth_aligned_overlay.png"
    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / "report.md"
    html_path = out_dir / "index.html"
    viewer_path = out_dir / "scene_3d.html"

    write_ply_mesh(hand_ply, verts_ref, pred["faces"], color=(40, 150, 255))
    write_ply_mesh(object_ply, object_vertices, object_faces, color=(180, 180, 180))
    overlay(rgb_path, verts_ref, joints_ref, selected["bbox"], intr, overlay_png)
    np.savez_compressed(
        npz_path,
        hand_vertices_ref=verts_ref,
        hand_faces=pred["faces"],
        hand_joints_ref=joints_ref,
        # Explicit WiLoR predicted MANO params. Rotation params are 3x3 matrices.
        wilor_pred_mano_global_orient=pred["mano_params"]["global_orient"],
        wilor_pred_mano_hand_pose=pred["mano_params"]["hand_pose"],
        wilor_pred_mano_betas=pred["mano_params"]["betas"],
        wilor_pred_cam=pred["pred_cam_raw"],
        wilor_pred_cam_t_crop=pred["pred_cam_t_crop"],
        wilor_is_right=pred["is_right"],
        # Local MANO mesh/joints produced from the params before RGB-D global-depth fitting.
        wilor_mano_vertices_local=pred["verts"],
        wilor_mano_joints_local=pred["joints"],
        wilor_vertices_local=pred["verts"],
        wilor_joints_local=pred["joints"],
        wilor_cam_t_full=pred["cam_t"],
        wilor_projected_uv=verts_uv,
        rgbd_fit_z_shift_m=np.array(z_shift, dtype=np.float32),
        rgbd_fit_target_depth_m=np.array(target_depth, dtype=np.float32),
        rgbd_fit_method=np.array("mano_params_mesh_global_z_translation_to_bbox_median_depth"),
        bbox_xyxy=selected["bbox"],
        object_vertices_ref=object_vertices,
        object_faces=object_faces,
    )

    hand_js = {"x": verts_ref[:,0].round(6).tolist(), "y": verts_ref[:,1].round(6).tolist(), "z": verts_ref[:,2].round(6).tolist(), "i": pred["faces"][:,0].astype(int).tolist(), "j": pred["faces"][:,1].astype(int).tolist(), "k": pred["faces"][:,2].astype(int).tolist()}
    obj_js = {"x": object_vertices[:,0].round(6).tolist(), "y": object_vertices[:,1].round(6).tolist(), "z": object_vertices[:,2].round(6).tolist(), "i": object_faces[:,0].astype(int).tolist(), "j": object_faces[:,1].astype(int).tolist(), "k": object_faces[:,2].astype(int).tolist()}
    joints_js = {"x": joints_ref[:,0].round(6).tolist(), "y": joints_ref[:,1].round(6).tolist(), "z": joints_ref[:,2].round(6).tolist()}
    viewer_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Stage I2 WiLoR hand/object 3D QC</title><script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>"
        "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}.note{padding:10px}#plot{width:100vw;height:94vh}</style></head><body>"
        "<div class='note'>Blue=WiLoR depth-aligned hand mesh; yellow=WiLoR joints; gray=object mesh from current object pose stage. Axes: X right, Y down, Z forward.</div><div id='plot'></div>"
        "<script>const hand="+json.dumps(hand_js)+";const obj="+json.dumps(obj_js)+";const joints="+json.dumps(joints_js)+";"
        "const traces=[{type:'mesh3d',name:'WiLoR hand mesh',x:hand.x,y:hand.y,z:hand.z,i:hand.i,j:hand.j,k:hand.k,color:'rgb(40,150,255)',opacity:0.55},"
        "{type:'scatter3d',mode:'markers+text',name:'WiLoR joints',x:joints.x,y:joints.y,z:joints.z,text:joints.x.map((_,i)=>String(i)),marker:{size:4,color:'yellow'}},"
        "{type:'mesh3d',name:'object CAD',x:obj.x,y:obj.y,z:obj.z,i:obj.i,j:obj.j,k:obj.k,color:'lightgray',opacity:0.45}];"
        "Plotly.newPlot('plot',traces,{paper_bgcolor:'#111',plot_bgcolor:'#111',font:{color:'#eee'},scene:{aspectmode:'data',xaxis:{title:'X right'},yaxis:{title:'Y down'},zaxis:{title:'Z forward'}},margin:{l:0,r:0,b:0,t:0}});</script></body></html>"
    )

    metrics = {
        "hand_vertices": int(len(verts_ref)),
        "hand_faces": int(len(pred["faces"])),
        "hand_joints": int(len(joints_ref)),
        "mano_param_source": "WiLoR out['pred_mano_params']",
        "mano_mesh_source": "model.mano(pred_mano_params, pose2rot=False) via WiLoR forward output",
        "rgbd_fit_method": "global z-translation fit to median RGB-D depth in selected hand bbox, excluding visible object mask",
        "detected_candidates": len(candidates),
        "selected_bbox_xyxy": [float(x) for x in selected["bbox"]],
        "selected_detector_confidence": selected["conf"],
        "selected_detector_class_is_right": selected["cls"],
        "selected_bbox_depth_pixels": selected_depth_stats["pixels"],
        "wilor_pred_mano_global_orient_shape": list(pred["mano_params"]["global_orient"].shape),
        "wilor_pred_mano_hand_pose_shape": list(pred["mano_params"]["hand_pose"].shape),
        "wilor_pred_mano_betas_shape": list(pred["mano_params"]["betas"].shape),
        "target_hand_depth_median_m": target_depth,
        "wilor_cam_median_z_before_alignment_m": float(np.median(verts_wilor_cam[:,2])),
        "z_shift_m": float(z_shift),
        "aligned_hand_median_z_m": float(np.median(verts_ref[:,2])),
        "hand_centroid_ref_m": [float(x) for x in verts_ref.mean(axis=0)],
        "object_centroid_ref_m": [float(x) for x in object_vertices.mean(axis=0)],
        "hand_object_centroid_distance_m": float(np.linalg.norm(verts_ref.mean(axis=0) - object_vertices.mean(axis=0))),
    }
    manifest = {
        "stage": "stage_i2_wilor_depth_aligned_hand_mesh",
        "sample_id": cfg["sample_id"],
        "hand_source": "WiLoR_predicted_MANO_params_RGBD_global_depth_fit",
        "coordinate_frame": "reference_camera_color_frame",
        "unit": "meter",
        "inputs": {
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "intrinsics": str(intr_path),
            "wilor_root": str(wilor_root),
            "wilor_ckpt": str(wilor_root / args.wilor_ckpt),
            "detector": str(wilor_root / args.detector),
            "object_mask_excluded_from_depth_stats": str(object_mask_path),
            "object_mesh_npz": str(object_npz_path),
            "object_source": object_source_label,
        },
        "metrics": metrics,
        "outputs": {
            "hand_mesh_ply": str(hand_ply),
            "object_mesh_copy_ply": str(object_ply),
            "npz": str(npz_path),
            "projection_overlay_png": str(overlay_png),
            "viewer_3d_html": str(viewer_path),
            "html": str(html_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
        "qc_status": "not_checked",
        "warnings": [
            "WiLoR predicted MANO params are now saved and used as the mesh source; RGB-D fitting is currently limited to global z-translation, not full MANO pose/shape optimization.",
            "Hand depth target comes from detector bbox RGB-D pixels excluding the visible object mask when possible; manual 2D/3D QC is required before contact-map computation.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    report_path.write_text(
        "# Stage I2 WiLoR MANO + RGB-D fitted hand mesh\n\n"
        f"sample: {cfg['sample_id']}\n\n"
        "RGB -> WiLoR predicted MANO params -> MANO mesh -> global RGB-D depth fitting.\n\n"
        f"- selected_bbox_xyxy: {metrics['selected_bbox_xyxy']}\n"
        f"- target_hand_depth_median_m: {target_depth:.6f}\n"
        f"- aligned_hand_median_z_m: {metrics['aligned_hand_median_z_m']:.6f}\n"
        f"- hand_object_centroid_distance_m: {metrics['hand_object_centroid_distance_m']:.6f}\n"
    )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Stage I2 WiLoR hand mesh QC</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:24px}img{max-width:900px;border:1px solid #555;margin:10px 0}code{color:#9ef}</style></head><body>"
        f"<h1>Stage I2 WiLoR depth-aligned hand mesh QC</h1><p>{cfg['sample_id']}</p>"
        "<p>Blue projected points = WiLoR hand mesh after RGB-D depth alignment; yellow = WiLoR joints; yellow bbox = selected hand detector bbox.</p>"
        f"<p><a href='{viewer_path.name}'>Open 3D hand/object viewer</a></p>"
        f"<img src='{overlay_png.name}'>"
        "<h2>Metrics</h2><pre>" + json.dumps(metrics, indent=2, ensure_ascii=False) + "</pre>"
        "</body></html>"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

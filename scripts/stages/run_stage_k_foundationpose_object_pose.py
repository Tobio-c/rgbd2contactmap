#!/usr/bin/env python3
"""Stage K: FoundationPose object 6D pose from RGB-D + CAD + SAM visible mask.

Mainline role:
  SAM click visible mask + RGB-D + object CAD -> FoundationPose -> object pose.

This wrapper does not use DexYCB object GT pose. Dataset labels are not needed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh
import yaml
from PIL import Image, ImageDraw


def rel_to_root(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def load_intrinsics(path: Path) -> np.ndarray:
    data = yaml.unsafe_load(path.read_text())["color"]
    return np.array(
        [
            [float(data["fx"]), 0.0, float(data["ppx"])],
            [0.0, float(data["fy"]), float(data["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_depth_m(path: Path, scale: float) -> np.ndarray:
    depth = np.array(Image.open(path))
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * scale
    return depth.astype(np.float32)


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def depth_mask_points(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    valid = (mask > 0) & (depth > 1e-4) & np.isfinite(depth)
    v, u = np.where(valid)
    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[v, u]
    x = (u.astype(np.float32) - K[0, 2]) / K[0, 0] * z
    y = (v.astype(np.float32) - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def export_ply_points(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(pts) == 0:
        path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        return
    cloud = trimesh.PointCloud(pts)
    cloud.export(path)


def overlay_pose(rgb: np.ndarray, K: np.ndarray, mesh: trimesh.Trimesh, pose: np.ndarray, mask: np.ndarray, path: Path) -> None:
    img = Image.fromarray(rgb).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    px = layer.load()
    for y, x in zip(*np.nonzero(mask)):
        px[x, y] = (0, 255, 120, 95)
    img = Image.alpha_composite(img, layer)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    verts_h = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float32)], axis=1)
    cam = (pose @ verts_h.T).T[:, :3]
    z = cam[:, 2]
    ok = z > 1e-4
    draw = ImageDraw.Draw(img)
    if ok.any():
        u = cam[ok, 0] / z[ok] * K[0, 0] + K[0, 2]
        v = cam[ok, 1] / z[ok] * K[1, 1] + K[1, 2]
        x0, y0, x1, y1 = int(np.floor(u.min())), int(np.floor(v.min())), int(np.ceil(u.max())), int(np.ceil(v.max()))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0, 255), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path)


def check_foundationpose_env(repo: Path) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if not repo.exists():
        missing.append(f"FoundationPose repo not found: {repo}")
        return missing, warnings
    for mod in ["torch", "pytorch3d", "nvdiffrast.torch", "open3d", "warp", "joblib"]:
        try:
            importlib.import_module(mod)
        except Exception as e:
            missing.append(f"import {mod}: {type(e).__name__}: {str(e)[:120]}")
    for stamp in ["2024-01-11-20-02-45", "2023-10-28-18-33-37"]:
        d = repo / "weights" / stamp
        for fname in ["model_best.pth", "config.yml"]:
            if not (d / fname).is_file():
                warnings.append(f"missing FoundationPose weight file: {d / fname}")
    if not list((repo / "mycpp" / "build").glob("mycpp*.so")):
        warnings.append("missing mycpp build output; run bash build_all_conda.sh inside FoundationPose env")
    return missing, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--config", default="configs/contactmap_cjq_mug_capture_001.yaml")
    ap.add_argument("--foundationpose-root", default="/home/originflow/project/FoundationPose")
    ap.add_argument("--mask-png", default="outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001/sam_mask_visible_raw.png")
    ap.add_argument("--out-dir", default="outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001")
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--est-refine-iter", type=int, default=5)
    ap.add_argument("--debug", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="Only validate inputs/env and emit readiness manifest.")
    args = ap.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(rel_to_root(root, args.config).read_text())
    li = cfg["local_inputs"]
    out_dir = rel_to_root(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = rel_to_root(root, li["rgb"])
    depth_path = rel_to_root(root, li["depth"])
    intr_path = rel_to_root(root, li["intrinsics_yml"])
    mesh_path = rel_to_root(root, li.get("cad_mesh_simple_obj") or li["cad_mesh_obj"])
    mask_path = rel_to_root(root, args.mask_png)
    fp_root = Path(args.foundationpose_root).resolve()

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = load_depth_m(depth_path, args.depth_scale)
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    # FoundationPose/PyTorch crop utilities may keep candidate poses in float64;
    # keep K float64 too to avoid CUDA matmul dtype mismatch on recent PyTorch.
    K = load_intrinsics(intr_path).astype(np.float64)
    mesh = trimesh.load(mesh_path, force="mesh")

    visible_pts = depth_mask_points(depth, mask, K)
    p_visible = out_dir / "visible_object_pointcloud_from_sam_mask_ref.ply"
    export_ply_points(p_visible, visible_pts)

    env_missing, env_warnings = check_foundationpose_env(fp_root)
    manifest = {
        "stage": "stage_k_foundationpose_object_pose",
        "sample_id": cfg["sample_id"],
        "object_name": cfg["source"]["object_name"],
        "mainline_role": "RGB-D + CAD + SAM visible mask -> FoundationPose object 6D pose",
        "uses_object_pose_seed": False,
        "inputs": {
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "intrinsics": str(intr_path),
            "cad_mesh": str(mesh_path),
            "sam_visible_mask": str(mask_path),
            "foundationpose_root": str(fp_root),
        },
        "precheck": {
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "mask_pixels": int(mask.sum()),
            "valid_mask_depth_points": int(len(visible_pts)),
            "mask_bbox_xyxy": mask_bbox(mask),
            "cad_vertices": int(len(mesh.vertices)),
            "cad_faces": int(len(mesh.faces)),
            "env_missing": env_missing,
            "env_warnings": env_warnings,
        },
        "outputs": {
            "visible_object_pointcloud_ref_ply": str(p_visible),
            "manifest": str(out_dir / "manifest.json"),
        },
        "qc_status": "not_run" if (args.dry_run or env_missing or env_warnings) else "not_checked",
    }

    if args.dry_run or env_missing or env_warnings:
        manifest["status"] = "blocked" if (env_missing or env_warnings) else "ready"
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        (out_dir / "report.md").write_text(
            "# Stage K FoundationPose object pose\n\n"
            f"status: {manifest['status']}\n\n"
            f"valid_mask_depth_points: {len(visible_pts)}\n\n"
            "env_missing:\n" + "\n".join([f"- {x}" for x in env_missing]) + "\n\n"
            "env_warnings:\n" + "\n".join([f"- {x}" for x in env_warnings]) + "\n"
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0 if args.dry_run else 2

    os.chdir(fp_root)
    sys.path.insert(0, str(fp_root))
    sys.path.insert(0, str(fp_root / "mycpp" / "build"))
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # type: ignore
    from Utils import set_logging_format, set_seed  # type: ignore
    import nvdiffrast.torch as dr  # type: ignore

    set_logging_format()
    set_seed(0)
    debug_dir = out_dir / "foundationpose_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=args.debug,
        glctx=glctx,
    )
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)

    posed_mesh = mesh.copy()
    posed_mesh.apply_transform(pose)
    p_pose = out_dir / "object_pose_cam.npy"
    p_npz = out_dir / "foundationpose_posed_cad_object_mesh_ref.npz"
    p_ply = out_dir / "foundationpose_posed_cad_object_mesh_ref.ply"
    p_overlay = out_dir / "rgb_sam_mask_foundationpose_bbox_overlay.png"
    np.save(p_pose, pose)
    np.savez_compressed(
        p_npz,
        object_pose_ref=pose,
        object_vertices_ref=np.asarray(posed_mesh.vertices, dtype=np.float32),
        object_faces=np.asarray(posed_mesh.faces, dtype=np.int32),
        cad_mesh_source=str(mesh_path),
    )
    posed_mesh.export(p_ply)
    overlay_pose(rgb, K, mesh, pose, mask, p_overlay)

    manifest["status"] = "success"
    manifest["object_pose_ref"] = pose.tolist()
    manifest["outputs"].update({
        "object_pose_cam_npy": str(p_pose),
        "posed_cad_npz": str(p_npz),
        "posed_cad_ply": str(p_ply),
        "overlay_png": str(p_overlay),
        "debug_dir": str(debug_dir),
    })
    manifest["qc_status"] = "not_checked"
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (out_dir / "report.md").write_text(
        "# Stage K FoundationPose object pose\n\n"
        "status: success\n\n"
        f"valid_mask_depth_points: {len(visible_pts)}\n\n"
        f"pose:\n```text\n{pose}\n```\n"
    )
    (out_dir / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>FoundationPose QC</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:24px}img{max-width:900px;border:1px solid #555}</style></head><body>"
        f"<h1>FoundationPose object pose</h1><p>{cfg['sample_id']}</p>"
        f"<h2>Overlay</h2><img src='{p_overlay.name}'>"
        "<h2>Manifest</h2><pre>" + json.dumps(manifest, indent=2, ensure_ascii=False) + "</pre>"
        "</body></html>"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

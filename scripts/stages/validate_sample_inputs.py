#!/usr/bin/env python3
"""Validate the clean custom RGB-D mug smoke-test sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def rel_to_root(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/contactmap_cjq_mug_capture_001.yaml")
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    cfg_path = rel_to_root(root, args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    li = cfg["local_inputs"]

    required_keys = [
        "rgb",
        "depth",
        "descriptor_json",
        "camera_json",
        "intrinsics_yml",
        "cad_mesh_obj",
        "cad_mesh_simple_obj",
    ]
    optional_keys = [
        "labels_npz",
        "sequence_pose_npz",
        "meta_yml",
        "extrinsics_yml",
    ]

    missing_required = []
    present = {}
    optional = {}
    for key in required_keys:
        p = rel_to_root(root, li[key])
        if not p.exists():
            missing_required.append({"key": key, "path": str(p)})
        else:
            present[key] = {"path": str(p), "bytes": p.stat().st_size}
    for key in optional_keys:
        if key not in li:
            continue
        p = rel_to_root(root, li[key])
        optional[key] = {"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0}

    descriptor = json.loads(rel_to_root(root, li["descriptor_json"]).read_text()) if not missing_required else None
    camera = json.loads(rel_to_root(root, li["camera_json"]).read_text()) if not missing_required else None
    intr = yaml.safe_load(rel_to_root(root, li["intrinsics_yml"]).read_text()) if not missing_required else None

    checks = {}
    if descriptor:
        checks["descriptor_sample_id"] = descriptor.get("sample_id")
        checks["reference_camera"] = descriptor.get("reference_camera")
        checks["num_cameras"] = len(descriptor.get("cameras", []))
    if camera:
        checks["camera_name"] = camera.get("camera")
        checks["rgb_size"] = [camera.get("rgb_intrinsic", {}).get("width"), camera.get("rgb_intrinsic", {}).get("height")]
        checks["depth_unit"] = camera.get("depth_unit")
    if intr:
        checks["intrinsics_color_keys"] = sorted(list(intr.get("color", {}).keys()))

    result = {
        "sample_id": cfg["sample_id"],
        "ok": not missing_required,
        "missing_required": missing_required,
        "present_required": present,
        "optional_inputs": optional,
        "checks": checks,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

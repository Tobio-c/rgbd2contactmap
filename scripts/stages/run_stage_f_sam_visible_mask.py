#!/usr/bin/env python3
"""Stage F: SAM visible object mask from manual click/point prompt.

Mainline rule: this stage must not use object pose seed or CAD-projected bbox.
It uses human-provided point prompts on the RGB image to obtain the object visible mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry


def rel_to_root(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def parse_point(text: str) -> tuple[float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be 'x,y' or 'x y'")
    return float(parts[0]), float(parts[1])


def load_prompt_points(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    points: list[tuple[float, float]] = []
    labels: list[int] = []
    prompt_source = {"type": "manual_click_points", "positive_points_xy": [], "negative_points_xy": []}

    if args.points_json:
        data = json.loads(Path(args.points_json).read_text())
        for p in data.get("positive_points_xy", []):
            points.append((float(p[0]), float(p[1])))
            labels.append(1)
        for p in data.get("negative_points_xy", []):
            points.append((float(p[0]), float(p[1])))
            labels.append(0)
        prompt_source["points_json"] = str(Path(args.points_json).resolve())

    for p in args.point:
        points.append(p)
        labels.append(1)
    for p in args.negative_point:
        points.append(p)
        labels.append(0)

    if not points:
        raise ValueError("At least one --point x,y is required for SAM click prompt")

    prompt_source["positive_points_xy"] = [[float(x), float(y)] for (x, y), l in zip(points, labels) if l == 1]
    prompt_source["negative_points_xy"] = [[float(x), float(y)] for (x, y), l in zip(points, labels) if l == 0]
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32), prompt_source


def save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def draw_overlay(
    rgb: Image.Image,
    mask: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    path: Path,
    color=(0, 255, 120, 120),
) -> None:
    base = rgb.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    px = layer.load()
    for y, x in zip(*np.nonzero(mask)):
        px[x, y] = color
    base = Image.alpha_composite(base, layer)
    draw = ImageDraw.Draw(base)
    for (x, y), label in zip(points, labels):
        r = 7
        c = (0, 255, 0, 255) if int(label) == 1 else (255, 0, 0, 255)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=c, fill=c, width=2)
        draw.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), outline=(255, 255, 255, 255), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(path)


def compare_overlay(
    rgb: Image.Image,
    sam_mask: np.ndarray,
    dataset_mask: np.ndarray | None,
    points: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> None:
    base = rgb.convert("RGBA")
    layers = []
    if dataset_mask is not None:
        layers.append((dataset_mask, (255, 0, 0, 70)))
    layers.append((sam_mask, (0, 255, 120, 130)))
    for mask, color in layers:
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        px = layer.load()
        for y, x in zip(*np.nonzero(mask)):
            px[x, y] = color
        base = Image.alpha_composite(base, layer)
    draw = ImageDraw.Draw(base)
    for (x, y), label in zip(points, labels):
        r = 7
        c = (0, 255, 0, 255) if int(label) == 1 else (255, 0, 0, 255)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255, 255), fill=c, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(path)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--config", default="configs/contactmap_cjq_mug_capture_001.yaml")
    ap.add_argument("--sam-checkpoint", default="/home/originflow/project/contact_pipeline/models/sam/sam_vit_b_01ec64.pth")
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--point", action="append", type=parse_point, default=[], help="Positive click point: x,y. Can be repeated.")
    ap.add_argument("--negative-point", action="append", type=parse_point, default=[], help="Negative click point: x,y. Can be repeated.")
    ap.add_argument("--points-json", default=None, help="Optional JSON with positive_points_xy and negative_points_xy lists.")
    ap.add_argument("--select-idx", type=int, default=None, help="Manually choose a SAM candidate index after QC.")
    ap.add_argument("--select-mode", default="largest", choices=["largest", "sam_score"], help="Candidate selection when --select-idx is not set. Click prompts often need largest, because SAM score may select a tiny part.")
    ap.add_argument("--qc-status", default="not_checked", choices=["not_checked", "pass", "pass_with_warning", "fail"])
    ap.add_argument("--out-dir", default="outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001")
    args = ap.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(rel_to_root(root, args.config).read_text())
    li = cfg["local_inputs"]
    out_dir = rel_to_root(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = rel_to_root(root, li["rgb"])
    labels_path = rel_to_root(root, li["labels_npz"])
    ckpt_path = Path(args.sam_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")

    rgb_pil = Image.open(rgb_path).convert("RGB")
    rgb = np.array(rgb_pil)
    points, point_labels, prompt = load_prompt_points(args)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=str(ckpt_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=point_labels,
        multimask_output=True,
    )
    masks = masks.astype(bool)
    scores = np.asarray(scores, dtype=np.float32)
    if args.select_idx is not None:
        if args.select_idx < 0 or args.select_idx >= len(masks):
            raise ValueError(f"--select-idx must be in [0, {len(masks)-1}], got {args.select_idx}")
        best_idx = int(args.select_idx)
        selection_reason = "manual_select_idx"
    elif args.select_mode == "largest":
        best_idx = int(np.argmax([int(m.sum()) for m in masks]))
        selection_reason = "largest_mask_area"
    else:
        best_idx = int(np.argmax(scores))
        selection_reason = "sam_predicted_iou_score"
    best_mask = masks[best_idx]

    object_id = int(cfg["source"]["object_id"])
    dataset_mask = None
    if labels_path.exists():
        labels = np.load(labels_path)
        if "seg" in labels:
            dataset_mask = labels["seg"] == object_id

    p_best = out_dir / "sam_mask_visible_raw.png"
    p_overlay = out_dir / "sam_mask_visible_overlay.png"
    p_compare = out_dir / "sam_mask_compare_overlay.png"
    p_prompt = out_dir / "sam_prompt.json"
    p_manifest = out_dir / "manifest.json"
    p_report = out_dir / "report.md"
    p_html = out_dir / "index.html"

    save_binary(p_best, best_mask)
    draw_overlay(rgb_pil, best_mask, points, point_labels, p_overlay)
    compare_overlay(rgb_pil, best_mask, dataset_mask, points, point_labels, p_compare)
    for i, m in enumerate(masks):
        save_binary(out_dir / f"sam_candidate_{i}_score_{scores[i]:.4f}.png", m)
        draw_overlay(rgb_pil, m, points, point_labels, out_dir / f"sam_candidate_{i}_overlay.png", color=(0, 255, 120, 115))

    prompt.update({
        "note": "Manual click prompt; no object pose seed and no CAD-projected bbox used.",
    })
    p_prompt.write_text(json.dumps(prompt, indent=2, ensure_ascii=False))

    candidate_metrics = []
    for i, m in enumerate(masks):
        item = {
            "idx": i,
            "sam_predicted_iou_score": float(scores[i]),
            "pixels": int(m.sum()),
        }
        if dataset_mask is not None:
            item["iou_with_dataset_mask_debug"] = mask_iou(m, dataset_mask)
        candidate_metrics.append(item)

    metrics = {
        "best_idx": best_idx,
        "selection_reason": selection_reason,
        "best_score": float(scores[best_idx]),
        "best_pixels": int(best_mask.sum()),
        "candidate_metrics": candidate_metrics,
    }
    if dataset_mask is not None:
        metrics["dataset_mask_pixels_debug"] = int(dataset_mask.sum())
        metrics["iou_with_dataset_mask_debug"] = mask_iou(best_mask, dataset_mask)

    manifest = {
        "stage": "stage_f_sam_click_visible_mask_frontend",
        "sample_id": cfg["sample_id"],
        "object_id": object_id,
        "object_name": cfg["source"]["object_name"],
        "mask_source": "SAM with manual click/point prompt",
        "mask_semantics_target": "visible_object_mask",
        "coordinate_frame": "image_pixel_frame",
        "mainline_role": "object visible mask before FoundationPose; no object pose seed dependency",
        "inputs": {
            "rgb": str(rgb_path),
            "sam_checkpoint": str(ckpt_path),
            "dataset_mask_debug_only": f"{labels_path}::seg=={object_id}" if dataset_mask is not None else None,
        },
        "prompt": prompt,
        "metrics": metrics,
        "outputs": {
            "sam_mask_visible_raw_png": str(p_best),
            "sam_mask_visible_overlay_png": str(p_overlay),
            "sam_mask_compare_overlay_png": str(p_compare),
            "sam_prompt_json": str(p_prompt),
            "html": str(p_html),
            "report": str(p_report),
            "manifest": str(p_manifest),
        },
        "qc_status": args.qc_status,
        "warnings": [
            "SAM mask is a visible-mask frontend candidate; use QC before sending to FoundationPose.",
            "Dataset mask IoU is debug-only and is not part of the target route.",
        ],
    }
    p_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    p_report.write_text(
        "# Stage F SAM click visible mask frontend\n\n"
        f"sample: {cfg['sample_id']}\n\n"
        f"positive_points_xy: {prompt['positive_points_xy']}\n\n"
        f"negative_points_xy: {prompt['negative_points_xy']}\n\n"
        f"best_idx: {best_idx}\n"
        f"selection_reason: {selection_reason}\n"
        f"best_score: {scores[best_idx]:.6f}\n"
        f"best_pixels: {int(best_mask.sum())}\n"
        + (f"iou_with_dataset_mask_debug: {metrics['iou_with_dataset_mask_debug']:.4f}\n" if dataset_mask is not None else "")
    )
    p_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Stage F SAM click mask QC</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:24px}img{max-width:900px;border:1px solid #555;margin:10px 0}code{color:#9ef}</style></head><body>"
        f"<h1>Stage F SAM click visible mask QC</h1><p>{cfg['sample_id']}</p>"
        "<p>Green dots = positive clicks; red dots = negative clicks. Green overlay = selected SAM visible mask. Compare overlay: red=dataset mask debug, green=SAM.</p>"
        f"<h2>Selected SAM overlay</h2><img src='{p_overlay.name}'>"
        f"<h2>Compare overlay</h2><img src='{p_compare.name}'>"
        "<h2>Candidate masks</h2>"
        + "".join([f"<h3>candidate {i}, score={scores[i]:.4f}</h3><img src='sam_candidate_{i}_overlay.png'>" for i in range(len(masks))])
        + "<h2>Metrics</h2><pre>" + json.dumps(metrics, indent=2, ensure_ascii=False) + "</pre>"
        "</body></html>"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

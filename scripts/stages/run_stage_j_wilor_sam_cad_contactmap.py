#!/usr/bin/env python3
"""Stage J: hand-object contactmap from WiLoR depth-aligned hand and SAM-refined CAD object.

Computes bidirectional vertex-to-triangle-surface distances plus a simple
signed inside/penetration diagnostic:
- hand vertex -> object triangle surface
- object vertex -> hand triangle surface
- vertices inside the opposite mesh are saved as penetration diagnostics only
Thresholds are intentionally loose for RGB-D/monocular frontend validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def rel_to_root(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def write_colored_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for tri in faces.astype(int):
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def point_to_mesh_surface_distance(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, chunk_size: int = 64):
    """Unsigned distance from each point to the nearest triangle surface.

    This intentionally avoids optional trimesh proximity dependencies such as
    rtree/pyembree, keeping Stage J self-contained in the existing environment.
    """
    points = np.asarray(points, dtype=np.float64)
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    ab = b - a
    ac = c - a
    out_dist = np.empty(len(points), dtype=np.float64)
    out_idx = np.empty(len(points), dtype=np.int32)

    for start in range(0, len(points), chunk_size):
        p = points[start:start + chunk_size]
        pp = p[:, None, :]
        aa = a[None, :, :]
        abb = ab[None, :, :]
        acc = ac[None, :, :]

        ap = pp - aa
        d1 = np.einsum("cti,cti->ct", abb, ap)
        d2 = np.einsum("cti,cti->ct", acc, ap)

        bp = pp - b[None, :, :]
        d3 = np.einsum("cti,cti->ct", abb, bp)
        d4 = np.einsum("cti,cti->ct", acc, bp)

        cp = pp - c[None, :, :]
        d5 = np.einsum("cti,cti->ct", abb, cp)
        d6 = np.einsum("cti,cti->ct", acc, cp)

        best = np.full(d1.shape, np.inf, dtype=np.float64)

        m = (d1 <= 0.0) & (d2 <= 0.0)
        best[m] = np.einsum("cti,cti->ct", ap, ap)[m]

        m = (d3 >= 0.0) & (d4 <= d3)
        best[m] = np.einsum("cti,cti->ct", bp, bp)[m]

        vc = d1 * d4 - d3 * d2
        m = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        denom = d1 - d3
        v = np.divide(d1, denom, out=np.zeros_like(d1), where=np.abs(denom) > 1e-12)
        proj = aa + v[:, :, None] * abb
        dist2 = np.einsum("cti,cti->ct", pp - proj, pp - proj)
        best[m] = dist2[m]

        m = (d6 >= 0.0) & (d5 <= d6)
        best[m] = np.einsum("cti,cti->ct", cp, cp)[m]

        vb = d5 * d2 - d1 * d6
        m = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        denom = d2 - d6
        w = np.divide(d2, denom, out=np.zeros_like(d2), where=np.abs(denom) > 1e-12)
        proj = aa + w[:, :, None] * acc
        dist2 = np.einsum("cti,cti->ct", pp - proj, pp - proj)
        best[m] = dist2[m]

        va = d3 * d6 - d5 * d4
        m = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
        denom = (d4 - d3) + (d5 - d6)
        w = np.divide(d4 - d3, denom, out=np.zeros_like(d4), where=np.abs(denom) > 1e-12)
        proj = b[None, :, :] + w[:, :, None] * (c - b)[None, :, :]
        dist2 = np.einsum("cti,cti->ct", pp - proj, pp - proj)
        best[m] = dist2[m]

        m = np.isinf(best)
        denom = va + vb + vc
        v = np.divide(vb, denom, out=np.zeros_like(vb), where=np.abs(denom) > 1e-12)
        w = np.divide(vc, denom, out=np.zeros_like(vc), where=np.abs(denom) > 1e-12)
        proj = aa + v[:, :, None] * abb + w[:, :, None] * acc
        dist2 = np.einsum("cti,cti->ct", pp - proj, pp - proj)
        best[m] = dist2[m]

        idx = np.argmin(best, axis=1)
        out_idx[start:start + len(p)] = idx.astype(np.int32)
        out_dist[start:start + len(p)] = np.sqrt(best[np.arange(len(p)), idx])
    return out_dist, out_idx


def points_inside_mesh_ray(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, chunk_size: int = 64) -> np.ndarray:
    """Parity ray-cast inside test along +X for penetration marking.

    Signed inside tests are reliable only for closed meshes. For open/noisy meshes,
    treat this as a conservative penetration diagnostic rather than exact physics.
    """
    points = np.asarray(points, dtype=np.float64).copy()
    # Deterministic tiny offsets reduce ray-on-edge degeneracy.
    if len(points):
        jitter = (np.arange(len(points), dtype=np.float64) % 997) * 1e-10
        points[:, 1] += jitter
        points[:, 2] += jitter[::-1]
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    # For ray dir=(1,0,0), h = cross(dir, e2) = (0, -e2_z, e2_y)
    h = np.stack([np.zeros(len(e2)), -e2[:, 2], e2[:, 1]], axis=1)
    a_det = np.einsum("ti,ti->t", e1, h)
    valid_tri = np.abs(a_det) > 1e-12
    inv_det = np.zeros_like(a_det)
    inv_det[valid_tri] = 1.0 / a_det[valid_tri]
    inside = np.zeros(len(points), dtype=bool)

    for start in range(0, len(points), chunk_size):
        p = points[start:start + chunk_size]
        s = p[:, None, :] - v0[None, :, :]
        u = inv_det[None, :] * np.einsum("cti,ti->ct", s, h)
        q = np.cross(s, e1[None, :, :])
        # dot(ray_dir, q) is q_x for ray dir=(1,0,0)
        v = inv_det[None, :] * q[:, :, 0]
        t = inv_det[None, :] * np.einsum("ti,cti->ct", e2, q)
        hit = valid_tri[None, :] & (t > 1e-9) & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0)
        inside[start:start + len(p)] = (hit.sum(axis=1) % 2) == 1
    return inside


def contact_colors(base, dist, contact_thr, near_thr, penetration=None):
    colors = np.tile(np.array(base, dtype=np.uint8), (len(dist), 1))
    near = dist <= near_thr
    contact = dist <= contact_thr
    # Penetration is intentionally diagnostic-only here. The parity ray-cast
    # inside test is unstable for open/non-watertight hand meshes, so it must
    # not force distant vertices into the final contact mask.
    colors[near] = np.array([255, 160, 0], dtype=np.uint8)
    colors[contact] = np.array([255, 30, 30], dtype=np.uint8)
    return colors, contact, near


def summarize_dist(d):
    return {
        "count": int(len(d)),
        "min_m": float(np.min(d)),
        "mean_m": float(np.mean(d)),
        "median_m": float(np.median(d)),
        "p10_m": float(np.percentile(d, 10)),
        "p90_m": float(np.percentile(d, 90)),
        "max_m": float(np.max(d)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--config", default="configs/contactmap_cjq_mug_capture_001.yaml")
    ap.add_argument("--hand-npz", default="outputs/wilor_depth_aligned_hand_mesh_single_frame/contactmap_cjq_mug_capture_001_frame000001/wilor_hand_mesh_depth_aligned_ref.npz")
    ap.add_argument("--object-npz", default="outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001/foundationpose_posed_cad_object_mesh_ref.npz")
    ap.add_argument("--out-dir", default="outputs/wilor_foundationpose_contactmap_single_frame/contactmap_cjq_mug_capture_001_frame000001")
    ap.add_argument("--contact-threshold-m", type=float, default=0.01)
    ap.add_argument("--near-threshold-m", type=float, default=0.03)
    args = ap.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(rel_to_root(root, args.config).read_text())
    hand_npz_path = rel_to_root(root, args.hand_npz)
    object_npz_path = rel_to_root(root, args.object_npz)
    out_dir = rel_to_root(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hz = np.load(hand_npz_path)
    oz = np.load(object_npz_path)
    hand_v = hz["hand_vertices_ref"].astype(np.float32)
    hand_f = hz["hand_faces"].astype(np.int32)
    if "object_vertices_refined_ref" in oz:
        object_v = oz["object_vertices_refined_ref"].astype(np.float32)
        object_source_label = "SAM_visible_mask_CAD_depth_refined_stage_h"
    else:
        object_v = oz["object_vertices_ref"].astype(np.float32)
        object_source_label = "FoundationPose_posed_CAD_stage_k"
    object_f = oz["object_faces"].astype(np.int32)

    hand_to_object_dist, hand_nearest_object_idx = point_to_mesh_surface_distance(hand_v, object_v, object_f)
    object_to_hand_dist, object_nearest_hand_idx = point_to_mesh_surface_distance(object_v, hand_v, hand_f)
    hand_penetration = points_inside_mesh_ray(hand_v, object_v, object_f)
    object_penetration = points_inside_mesh_ray(object_v, hand_v, hand_f)
    hand_signed_dist = hand_to_object_dist.copy()
    object_signed_dist = object_to_hand_dist.copy()
    hand_signed_dist[hand_penetration] *= -1.0
    object_signed_dist[object_penetration] *= -1.0

    hand_colors, hand_contact, hand_near = contact_colors((50, 150, 255), hand_to_object_dist, args.contact_threshold_m, args.near_threshold_m, hand_penetration)
    obj_colors, obj_contact, obj_near = contact_colors((190, 190, 190), object_to_hand_dist, args.contact_threshold_m, args.near_threshold_m, object_penetration)

    hand_contact_ply = out_dir / "hand_contact_map_wilor_ref.ply"
    object_contact_ply = out_dir / "object_contact_map_sam_cad_ref.ply"
    npz_path = out_dir / "contact_map_wilor_sam_cad_ref.npz"
    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / "report.md"
    html_path = out_dir / "index.html"
    viewer_path = out_dir / "scene_3d.html"

    write_colored_ply_mesh(hand_contact_ply, hand_v, hand_f, hand_colors)
    write_colored_ply_mesh(object_contact_ply, object_v, object_f, obj_colors)
    np.savez_compressed(
        npz_path,
        hand_vertices_ref=hand_v,
        hand_faces=hand_f,
        object_vertices_ref=object_v,
        object_faces=object_f,
        hand_to_object_distance_m=hand_to_object_dist.astype(np.float32),
        object_to_hand_distance_m=object_to_hand_dist.astype(np.float32),
        hand_to_object_signed_distance_m=hand_signed_dist.astype(np.float32),
        object_to_hand_signed_distance_m=object_signed_dist.astype(np.float32),
        hand_nearest_object_triangle_idx=hand_nearest_object_idx.astype(np.int32),
        object_nearest_hand_triangle_idx=object_nearest_hand_idx.astype(np.int32),
        hand_penetration_mask=hand_penetration.astype(bool),
        object_penetration_mask=object_penetration.astype(bool),
        hand_contact_mask=hand_contact.astype(bool),
        object_contact_mask=obj_contact.astype(bool),
        hand_near_mask=hand_near.astype(bool),
        object_near_mask=obj_near.astype(bool),
        contact_threshold_m=np.array(args.contact_threshold_m, dtype=np.float32),
        near_threshold_m=np.array(args.near_threshold_m, dtype=np.float32),
        distance_method=np.array("vertex_to_triangle_surface_penetration_diagnostic_only"),
    )

    metrics = {
        "hand_vertices": int(len(hand_v)),
        "object_vertices": int(len(object_v)),
        "contact_threshold_m": args.contact_threshold_m,
        "near_threshold_m": args.near_threshold_m,
        "hand_contact_vertices": int(hand_contact.sum()),
        "hand_contact_ratio": float(hand_contact.mean()),
        "hand_near_vertices": int(hand_near.sum()),
        "hand_near_ratio": float(hand_near.mean()),
        "object_contact_vertices": int(obj_contact.sum()),
        "object_contact_ratio": float(obj_contact.mean()),
        "object_near_vertices": int(obj_near.sum()),
        "object_near_ratio": float(obj_near.mean()),
        "hand_penetration_vertices": int(hand_penetration.sum()),
        "hand_penetration_ratio": float(hand_penetration.mean()),
        "object_penetration_vertices": int(object_penetration.sum()),
        "object_penetration_ratio": float(object_penetration.mean()),
        "hand_to_object_distance": summarize_dist(hand_to_object_dist),
        "object_to_hand_distance": summarize_dist(object_to_hand_dist),
        "hand_to_object_signed_distance": summarize_dist(hand_signed_dist),
        "object_to_hand_signed_distance": summarize_dist(object_signed_dist),
    }

    hand_js = {"x": hand_v[:,0].round(6).tolist(), "y": hand_v[:,1].round(6).tolist(), "z": hand_v[:,2].round(6).tolist(), "i": hand_f[:,0].astype(int).tolist(), "j": hand_f[:,1].astype(int).tolist(), "k": hand_f[:,2].astype(int).tolist(), "color": [f"rgb({r},{g},{b})" for r,g,b in hand_colors]}
    obj_js = {"x": object_v[:,0].round(6).tolist(), "y": object_v[:,1].round(6).tolist(), "z": object_v[:,2].round(6).tolist(), "i": object_f[:,0].astype(int).tolist(), "j": object_f[:,1].astype(int).tolist(), "k": object_f[:,2].astype(int).tolist(), "color": [f"rgb({r},{g},{b})" for r,g,b in obj_colors]}
    hc = hand_v[hand_contact]
    oc = object_v[obj_contact]
    hnear = hand_v[hand_near & (~hand_contact)]
    onear = object_v[obj_near & (~obj_contact)]
    pts_js = {
        "hc": {"x": hc[:,0].round(6).tolist(), "y": hc[:,1].round(6).tolist(), "z": hc[:,2].round(6).tolist()},
        "oc": {"x": oc[:,0].round(6).tolist(), "y": oc[:,1].round(6).tolist(), "z": oc[:,2].round(6).tolist()},
        "hnear": {"x": hnear[:,0].round(6).tolist(), "y": hnear[:,1].round(6).tolist(), "z": hnear[:,2].round(6).tolist()},
        "onear": {"x": onear[:,0].round(6).tolist(), "y": onear[:,1].round(6).tolist(), "z": onear[:,2].round(6).tolist()},
    }
    viewer_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Stage J contactmap QC</title><script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>"
        "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}.note{padding:10px}#plot{width:100vw;height:94vh}</style></head><body>"
        f"<div class='note'>Stage J WiLoR+SAM-CAD contactmap. Red=surface contact <= {args.contact_threshold_m}m; orange=near <= {args.near_threshold_m}m. Distances are vertex-to-triangle-surface. Penetration/signed distance uses parity ray-cast inside test and is diagnostic only, not included in contact mask.</div><div id='plot'></div>"
        "<script>const hand="+json.dumps(hand_js)+";const obj="+json.dumps(obj_js)+";const pts="+json.dumps(pts_js)+";"
        "const traces=[{type:'mesh3d',name:'hand contact mesh',x:hand.x,y:hand.y,z:hand.z,i:hand.i,j:hand.j,k:hand.k,vertexcolor:hand.color,opacity:0.55},"
        "{type:'mesh3d',name:'object contact mesh',x:obj.x,y:obj.y,z:obj.z,i:obj.i,j:obj.j,k:obj.k,vertexcolor:obj.color,opacity:0.45},"
        "{type:'scatter3d',mode:'markers',name:'HAND CONTACT',x:pts.hc.x,y:pts.hc.y,z:pts.hc.z,marker:{size:4,color:'red'}},"
        "{type:'scatter3d',mode:'markers',name:'OBJECT CONTACT',x:pts.oc.x,y:pts.oc.y,z:pts.oc.z,marker:{size:4,color:'red'}},"
        "{type:'scatter3d',mode:'markers',name:'hand near',x:pts.hnear.x,y:pts.hnear.y,z:pts.hnear.z,marker:{size:2,color:'orange'}},"
        "{type:'scatter3d',mode:'markers',name:'object near',x:pts.onear.x,y:pts.onear.y,z:pts.onear.z,marker:{size:2,color:'orange'}}];"
        "Plotly.newPlot('plot',traces,{paper_bgcolor:'#111',plot_bgcolor:'#111',font:{color:'#eee'},scene:{aspectmode:'data',xaxis:{title:'X right'},yaxis:{title:'Y down'},zaxis:{title:'Z forward'}},margin:{l:0,r:0,b:0,t:0}});</script></body></html>"
    )

    manifest = {
        "stage": "stage_j_wilor_foundationpose_contactmap",
        "sample_id": cfg["sample_id"],
        "coordinate_frame": "reference_camera_color_frame",
        "unit": "meter",
        "hand_source": "WiLoR_depth_aligned_stage_i2",
        "object_source": object_source_label,
        "distance_method": "vertex_to_triangle_surface_penetration_diagnostic_only",
        "inputs": {"hand_npz": str(hand_npz_path), "object_npz": str(object_npz_path)},
        "metrics": metrics,
        "outputs": {
            "hand_contact_ply": str(hand_contact_ply),
            "object_contact_ply": str(object_contact_ply),
            "npz": str(npz_path),
            "viewer_3d_html": str(viewer_path),
            "html": str(html_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
        "qc_status": "not_checked",
        "warnings": [
            "Contact mask uses vertex-to-triangle-surface distance only; penetration uses a parity ray-cast signed inside test and is saved as diagnostic-only because it is reliable only for sufficiently closed meshes.",
            "WiLoR depth alignment is approximate; manual 3D QC is required before using as supervision.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    report_path.write_text("# Stage J WiLoR + SAM-CAD contactmap\n\n" + json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Stage J contactmap QC</title><style>body{font-family:sans-serif;background:#111;color:#eee;margin:24px}code{color:#9ef}pre{white-space:pre-wrap}</style></head><body>"
        f"<h1>Stage J WiLoR + SAM-CAD contactmap QC</h1><p>{cfg['sample_id']}</p>"
        f"<p><a href='{viewer_path.name}'>Open interactive 3D contact viewer</a></p>"
        "<h2>Metrics</h2><pre>" + json.dumps(metrics, indent=2, ensure_ascii=False) + "</pre>"
        "</body></html>"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

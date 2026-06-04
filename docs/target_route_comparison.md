# contactmap_cjq target route comparison

## Final target route after manual QC

Selected route:

```text
RGB-D + intrinsics + object CAD
-> SAM manual click/point visible object mask, no object pose seed
-> FoundationPose RGB-D + CAD + visible mask object 6D pose
-> object pose overlay/render QC
-> WiLoR RGB hand mesh + RGB-D depth alignment
-> FoundationPose posed CAD + WiLoR hand mesh contactmap
-> 2D/3D QC viewer
```

Decision: keep this as the current target route for `contactmap_cjq`.

Current status: manual QC passed for the representative DexYCB single-frame smoke test.

## Why this route is selected

1. It matches the intended input boundary.

- Uses RGB-D camera data, intrinsics, and object CAD.
- Does not use DexYCB GT object pose as a seed.
- Does not use CAD-projected bbox to prompt SAM; mask comes from image click prompt.

2. Object side is stronger than previous local routes.

- FoundationPose directly estimates object 6D pose from RGB-D + CAD + visible mask.
- The earlier CAD-depth translation-only refinement was useful as a debug/fallback path, but cannot solve full 6DoF robustly.
- The earlier GT-pose route is only a validation/debug baseline, not the target pipeline.

3. Hand side is acceptable for smoke test.

- WiLoR gives MANO-style hand mesh from RGB.
- RGB-D depth alignment puts the hand mesh into the same reference camera frame as the object mesh.
- Manual QC has passed for the representative frame, but it remains an approximate frontend.

4. Contactmap output is available and inspectable.

- Bidirectional nearest-vertex distances are saved in NPZ.
- Contact and near-contact masks are saved for both hand and object vertices.
- Interactive 3D viewer exists for manual QC.

## Route comparison

| Route | Object input | Object pose source | Hand source | Contactmap status | Decision |
|---|---|---|---|---|---|
| GT CAD pose + debug hand | RGB-D + CAD + DexYCB GT pose | dataset GT | DexYCB/WiLoR debug | worked as debug | archive/debug only |
| SAM visible mask + CAD-depth translation refinement | RGB-D + CAD + SAM mask | partial depth alignment / translation refinement | WiLoR | worked as fallback | fallback only |
| PCA/ICP / auto RGBD CAD pose seed | RGB-D + CAD/mask | fragile seed/refinement | WiLoR/debug | unstable | not target |
| FoundationPose + WiLoR, current | RGB-D + CAD + SAM visible mask | FoundationPose, no GT pose seed | WiLoR + RGB-D alignment | manual QC passed | selected target |

## Representative output paths

FoundationPose object pose:

```text
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/index.html
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/object_pose_cam.npy
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/foundationpose_posed_cad_object_mesh_ref.npz
```

WiLoR hand:

```text
/home/originflow/project/contactmap_cjq/outputs/wilor_depth_aligned_hand_mesh_single_frame/dexycb_obj005_frame000035_ref932122060857/index.html
/home/originflow/project/contactmap_cjq/outputs/wilor_depth_aligned_hand_mesh_single_frame/dexycb_obj005_frame000035_ref932122060857/wilor_hand_mesh_depth_aligned_ref.npz
```

Contactmap:

```text
/home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/index.html
/home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/scene_3d.html
/home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/contact_map_wilor_sam_cad_ref.npz
```

## Remaining limits

- Current validation is single reference-camera, single-frame smoke test.
- Multi-camera version still needs per-camera masks, calibrated camera extrinsics, and fusion/selection policy.
- Contactmap currently uses nearest-vertex distance; production should upgrade to surface-sampled distance or SDF/UDF-style diagnostics.
- WiLoR depth alignment is approximate; for production labels it should retain manual/automatic QC gates.

## Next implementation target

Freeze this as the `contactmap_cjq` target route and clean the naming/scripts around it:

```text
Stage F: SAM visible mask
Stage K: FoundationPose object pose
Stage I2: WiLoR RGB-D hand mesh
Stage J: FoundationPose + WiLoR contactmap
```

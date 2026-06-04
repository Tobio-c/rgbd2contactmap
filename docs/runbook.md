# Runbook

## 1. Validate input sample

```bash
python scripts/stages/validate_sample_inputs.py --config configs/contactmap_cjq_mug_capture_001.yaml
```

## 2. Run SAM mask

```bash
python scripts/stages/run_stage_f_sam_visible_mask.py   --config configs/contactmap_cjq_mug_capture_001.yaml   --point 640,360   --out-dir outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001
```

如果 SAM 选中区域不合理，调整 `--point`、增加 `--negative-point`，或使用 `--select-idx` 手动选择候选 mask。

## 3. Run FoundationPose

先 dry-run 检查环境：

```bash
python scripts/stages/run_stage_k_foundationpose_object_pose.py   --config configs/contactmap_cjq_mug_capture_001.yaml   --mask-png outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001/sam_mask_visible_raw.png   --out-dir outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001   --dry-run
```

如 manifest 中 env_missing/env_warnings 为空，再去掉 `--dry-run` 正式运行。

## 4. Run WiLoR

```bash
python scripts/stages/run_stage_i2_wilor_depth_aligned_hand_mesh.py   --config configs/contactmap_cjq_mug_capture_001.yaml   --object-mask outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001/sam_mask_visible_raw.png   --object-npz outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001/foundationpose_posed_cad_object_mesh_ref.npz   --out-dir outputs/wilor_depth_aligned_hand_mesh_single_frame/contactmap_cjq_mug_capture_001_frame000001
```

## 5. Run contact map

```bash
python scripts/stages/run_stage_j_wilor_sam_cad_contactmap.py   --config configs/contactmap_cjq_mug_capture_001.yaml   --hand-npz outputs/wilor_depth_aligned_hand_mesh_single_frame/contactmap_cjq_mug_capture_001_frame000001/wilor_hand_mesh_depth_aligned_ref.npz   --object-npz outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001/foundationpose_posed_cad_object_mesh_ref.npz   --out-dir outputs/wilor_foundationpose_contactmap_single_frame/contactmap_cjq_mug_capture_001_frame000001
```

## Expected artifacts

- `manifest.json`: machine-readable run metadata
- `report.md`: human-readable QC summary
- `scene_3d.html`: interactive 3D visualization
- `contact_map_wilor_sam_cad_ref.npz`: contact/near/distance arrays
- `hand_contact_map_wilor_ref.ply`, `object_contact_map_sam_cad_ref.ply`: colored meshes

这些 artifacts 都在 outputs 下，默认不入库。

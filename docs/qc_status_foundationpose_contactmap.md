# contactmap_cjq FoundationPose + WiLoR contactmap QC status

## 结论

当前单帧闭环已跑通：

RGB-D + CAD + SAM visible mask -> FoundationPose object 6D pose -> WiLoR RGB hand mesh + RGB-D depth alignment -> hand/object nearest-vertex contactmap。

状态：sample-level smoke test passed；manual 3D QC passed。尚不能宣称 production-ready 或多相机完整流水线已完成。

## 自动验证结果

1. Stage K FoundationPose object pose

- status: success
- pose: object_pose_cam.npy shape=(4,4)
- rotation det≈1.0，orthogonality error≈2.6e-7
- translation_m: [0.171337, 0.226183, 0.895429]
- object mesh: 10983 vertices, 15728 faces
- visual overlay: SAM mask 与 FoundationPose CAD bbox 大致覆盖同一 mustard bottle，无明显粗大错位
- qc_status: manual_passed

2. Stage I2 WiLoR hand mesh

- hand mesh: 778 vertices, 1538 faces, 21 joints
- selected hand bbox: [308, 367, 455, 477]
- detector confidence: 0.899
- aligned hand median depth: 0.848 m
- visual overlay: hand mesh/joints 大致落在右手区域；仍属近似深度对齐，需要人工 3D QC
- qc_status: manual_passed

3. Stage J contactmap

- contact_threshold_m: 0.01
- near_threshold_m: 0.03
- hand_contact_vertices: 128 / 778
- object_contact_vertices: 959 / 10983
- hand_near_vertices: 531 / 778
- object_near_vertices: 5586 / 10983
- min_distance_m: 0.000784
- 3D viewer: scene_3d.html contains hand/object mesh and contact/near-contact layers
- qc_status: manual_passed

## Key outputs

- /home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/index.html
- /home/originflow/project/contactmap_cjq/outputs/wilor_depth_aligned_hand_mesh_single_frame/dexycb_obj005_frame000035_ref932122060857/index.html
- /home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/index.html
- /home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/scene_3d.html
- /home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/contact_map_wilor_sam_cad_ref.npz

## 当前边界

- 这是单参考相机单帧 smoke test，不是多相机融合 production pipeline。
- object 侧依赖 CAD + FoundationPose；不是 RGB-D-only open-world 物体重建。
- contactmap 是 nearest-vertex distance smoke test，尚未做 surface sampling/SDF/penetration correction。
- 当前代表帧 manual 3D QC 已通过；用于训练监督前仍建议保留逐帧/批量 QC gate。

## 下一步

当前代表帧人工 QC 已通过。后续扩展多帧/多相机时继续使用 Stage J 的 3D viewer 做人工 QC：

/home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/scene_3d.html

该路线已固化为当前 target route；下一步是统一脚本命名、增加一键运行脚本，并扩展多帧/多相机。

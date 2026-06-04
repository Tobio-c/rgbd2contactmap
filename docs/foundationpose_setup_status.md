# FoundationPose setup status

项目已接入 FoundationPose wrapper；本机 FoundationPose conda 环境与官方 scorer/refiner 权重现已通过核心 runtime/import/weights 验证。

## 已完成

```text
FoundationPose 源码:
/home/originflow/project/FoundationPose

conda env:
/home/originflow/miniforge3/envs/foundationpose

contactmap_cjq wrapper:
/home/originflow/project/contactmap_cjq/scripts/stages/run_stage_k_foundationpose_object_pose.py
```

Stage K 输入：

```text
RGB
aligned depth
camera intrinsics
object CAD mesh
SAM click visible mask
```

Stage K 不使用：

```text
DexYCB object GT pose
CAD-projected bbox
object pose seed
```

## 本机环境状态

已安装并验证：

```text
Python 3.11.15
CUDA toolkit / nvcc 12.8.93, installed inside conda env
PyTorch 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU, sm_120
pytorch3d 0.7.9
nvdiffrast 0.4.0
open3d 0.19.0
warp-lang 1.14.0
joblib 1.5.3
trimesh / cv2 / scipy / sklearn / imageio / kornia / omegaconf / pyrender / matplotlib 等
mycpp native extension
FoundationPose estimater import
```

mycpp build output：

```text
/home/originflow/project/FoundationPose/mycpp/build/mycpp.cpython-311-x86_64-linux-gnu.so
```

nvdiffrast 编译修复记录：

```text
问题: conda-forge gcc/g++ 14.3.0 超过 CUDA 12.8 支持上限，nvdiffrast wheel build 失败。
修复: 将 foundationpose env 内 gcc_linux-64 / gxx_linux-64 降到 13.4.0。
当前编译器: x86_64-conda-linux-gnu-c++ 13.4.0
结果: nvdiffrast 0.4.0 successfully installed
```

## 当前 dry-run 结果

```text
mask_pixels: 4850
valid_mask_depth_points: 4850
cad_vertices: 10983
status: blocked_until_foundationpose_weights_available
```

输出：

```text
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/manifest.json
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/report.md
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/visible_object_pointcloud_from_sam_mask_ref.ply
```

## 当前验证结果

验证命令：

```bash
cd /home/originflow/project/FoundationPose
/home/originflow/miniforge3/envs/foundationpose/bin/python check_env.py
```

通过项：

```text
PyTorch: 2.11.0+cu128
CUDA available: True
CUDA device: NVIDIA GeForce RTX 5060 Laptop GPU
import pytorch3d: ok
import nvdiffrast.torch: ok
import trimesh: ok
import open3d: ok
import cv2: ok
import warp: ok
import mycpp: ok
import estimater: ok
pip check: No broken requirements found
```

## 权重状态

已从 `/home/originflow/下载` 解压并放置到正确目录：

```text
/home/originflow/project/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth
/home/originflow/project/FoundationPose/weights/2024-01-11-20-02-45/config.yml
/home/originflow/project/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
/home/originflow/project/FoundationPose/weights/2023-10-28-18-33-37/config.yml
```

验证结果：

```text
Weights (scorer, 2024-01-11-20-02-45): ok
Weights (refiner, 2023-10-28-18-33-37): ok
```

## 剩余阻塞项

```text
无 runtime/weights 阻塞；可以进入 Stage K 正式 FoundationPose 推理。
```

## Stage K 正式推理结果

已运行 FoundationPose 正式 object pose 推理：

```text
status: success
sample: dexycb_obj005_frame000035_ref932122060857
mask_pixels: 4850
valid_mask_depth_points: 4850
object_pose_translation_m: [0.171337, 0.226183, 0.895429]
object_vertices: 10983
object_faces: 15728
qc_status: not_checked
```

输出：

```text
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/object_pose_cam.npy
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/foundationpose_posed_cad_object_mesh_ref.npz
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/foundationpose_posed_cad_object_mesh_ref.ply
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/rgb_sam_mask_foundationpose_bbox_overlay.png
/home/originflow/project/contactmap_cjq/outputs/foundationpose_object_pose_single_frame/dexycb_obj005_frame000035_ref932122060857/index.html
```

## WiLoR + FoundationPose contactmap 结果

已用当前 FoundationPose object mesh 重新生成 WiLoR depth-aligned hand mesh 与 contactmap：

```text
hand_vertices: 778
object_vertices: 10983
contact_threshold_m: 0.01
near_threshold_m: 0.03
min_distance_m: 0.000784
hand_contact_vertices: 128
object_contact_vertices: 959
hand_near_vertices: 531
object_near_vertices: 5586
qc_status: not_checked
```

输出：

```text
/home/originflow/project/contactmap_cjq/outputs/wilor_depth_aligned_hand_mesh_single_frame/dexycb_obj005_frame000035_ref932122060857/index.html
/home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/contact_map_wilor_sam_cad_ref.npz
/home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/scene_3d.html
/home/originflow/project/contactmap_cjq/outputs/wilor_foundationpose_contactmap_single_frame/dexycb_obj005_frame000035_ref932122060857/index.html
```

## 下一步

```text
1. 人工打开 Stage K overlay / Stage J 3D viewer 做 object pose 与 hand-object contact QC；
2. 若 QC 通过，将该 route 固化为 contactmap_cjq 当前主线；
3. 若 QC 不通过，优先调整 SAM mask selection 或 FoundationPose/WiLoR 对齐，不再回退到 GT pose seed。
```

# RGB-D 3D Mesh ContactMap Pipeline

该仓库是一个轻量化团队维护版本：只保留源码、文档和一个小型 RGB-D mug 输入样本；运行生成的 outputs 不入库。

## 目标

输入：单帧或多视角 RGB-D、相机内参、目标物体 CAD mesh、手部图像。

主线流程：

1. Stage F: RGB + 人工点击提示 -> SAM 可见物体 mask
2. Stage K: RGB-D + CAD + SAM mask -> FoundationPose 物体 6D pose
3. Stage I2: RGB-D + WiLoR -> 深度对齐的手 mesh
4. Stage J: hand mesh + posed object mesh -> surface-distance contact map

当前 Stage J 的临时稳定规则：

contact = vertex-to-triangle surface distance <= threshold

penetration / signed distance 仅作为 diagnostic 输出，不参与最终 contact mask。

## 仓库内容

```text
configs/                              # repo-relative sample config
data/samples/contactmap_cjq_mug_capture_001/
  frame_000001/                       # 小型 RGB-D 输入样本
  cad/body1_metric_centered.obj       # 小型 mug CAD mesh，Git LFS 管理
scripts/stages/                       # F/K/I2/J 主线 stage 脚本
scripts/utils/                        # viewer/helper 工具
docs/                                 # 路线说明、QC 状态、安装说明
outputs/                              # 运行时生成；默认不入库
```

## 准备

建议先启用 Git LFS：

```bash
git lfs install
git lfs pull
```

Python 依赖：

```bash
pip install -r requirements.txt
```

外部依赖按本机环境配置，不随仓库上传：

- SAM checkpoint，例如 `/home/originflow/project/contact_pipeline/models/sam/sam_vit_b_01ec64.pth`
- FoundationPose repo 和 weights，例如 `/home/originflow/project/FoundationPose`
- WiLoR repo 和 weights，例如 `/home/originflow/project/WiLoR`

可参考：

```text
docs/foundationpose_setup_status.md
external_refs/EXTERNAL_PATHS.md
```

## 输入样本

默认配置：

```text
configs/contactmap_cjq_mug_capture_001.yaml
```

核心输入：

```text
data/samples/contactmap_cjq_mug_capture_001/frame_000001/cam_CP0BB53000CG/rgb.png
data/samples/contactmap_cjq_mug_capture_001/frame_000001/cam_CP0BB53000CG/depth.png
data/samples/contactmap_cjq_mug_capture_001/frame_000001/cam_CP0BB53000CG/intrinsics.yml
data/samples/contactmap_cjq_mug_capture_001/frame_000001/descriptor.json
data/samples/contactmap_cjq_mug_capture_001/cad/body1_metric_centered.obj
```

## 一键 smoke validation

```bash
python scripts/stages/validate_sample_inputs.py   --config configs/contactmap_cjq_mug_capture_001.yaml
```

期望：打印 JSON，`ok: true`，并确认 rgb/depth/intrinsics/CAD/descriptor 均存在。

## 分阶段运行示例

根据实际画面调整 SAM 点击点。下面点位只是模板：

```bash
python scripts/stages/run_stage_f_sam_visible_mask.py   --config configs/contactmap_cjq_mug_capture_001.yaml   --point 640,360   --out-dir outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001
```

FoundationPose dry-run / precheck：

```bash
python scripts/stages/run_stage_k_foundationpose_object_pose.py   --config configs/contactmap_cjq_mug_capture_001.yaml   --mask-png outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001/sam_mask_visible_raw.png   --out-dir outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001   --dry-run
```

WiLoR hand mesh：

```bash
python scripts/stages/run_stage_i2_wilor_depth_aligned_hand_mesh.py   --config configs/contactmap_cjq_mug_capture_001.yaml   --object-mask outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001/sam_mask_visible_raw.png   --object-npz outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001/foundationpose_posed_cad_object_mesh_ref.npz   --out-dir outputs/wilor_depth_aligned_hand_mesh_single_frame/contactmap_cjq_mug_capture_001_frame000001
```

Contact map：

```bash
python scripts/stages/run_stage_j_wilor_sam_cad_contactmap.py   --config configs/contactmap_cjq_mug_capture_001.yaml   --hand-npz outputs/wilor_depth_aligned_hand_mesh_single_frame/contactmap_cjq_mug_capture_001_frame000001/wilor_hand_mesh_depth_aligned_ref.npz   --object-npz outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001/foundationpose_posed_cad_object_mesh_ref.npz   --out-dir outputs/wilor_foundationpose_contactmap_single_frame/contactmap_cjq_mug_capture_001_frame000001
```

期望输出路径：

```text
outputs/sam_click_visible_mask_single_frame/contactmap_cjq_mug_capture_001_frame000001/
outputs/foundationpose_object_pose_single_frame/contactmap_cjq_mug_capture_001_frame000001/
outputs/wilor_depth_aligned_hand_mesh_single_frame/contactmap_cjq_mug_capture_001_frame000001/
outputs/wilor_foundationpose_contactmap_single_frame/contactmap_cjq_mug_capture_001_frame000001/
```

生成的 `scene_3d.html`、`.npz`、`.ply`、overlay PNG 均为运行产物，不提交到 Git。

## 维护原则

- repo 只放输入小样本、源码、文档。
- outputs 一律不入库。
- 旧路线不入库，只在文档中保留结论。
- 大 CAD / 模型权重走 Git LFS、Release asset 或外部存储。
- 本机绝对路径只出现在示例命令或 external refs，不写死在 config 的必需输入里。

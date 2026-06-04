# Pipeline Overview

当前维护版本聚焦一条主线：

RGB-D + CAD + SAM visible mask + FoundationPose object pose + WiLoR hand mesh -> geometry contact map。

## Stage F: SAM visible mask

输入 RGB 和人工正/负点击点，输出目标物体可见 mask。该阶段不使用 object GT pose，也不使用 CAD 投影 bbox。

## Stage K: FoundationPose object pose

输入 RGB-D、SAM mask、CAD mesh 和相机内参，调用外部 FoundationPose 环境估计物体 6D pose。仓库只保留 wrapper，不上传 FoundationPose 权重和构建产物。

## Stage I2: WiLoR depth-aligned hand mesh

输入 RGB-D 和可选 object mask，调用外部 WiLoR 环境预测手部 MANO mesh，并用 RGB-D 深度做全局深度对齐。

## Stage J: Contact map

输入 hand mesh 和 posed object mesh，计算双向 vertex-to-triangle surface distance：

- hand vertex -> object triangle surface
- object vertex -> hand triangle surface

当前最终 contact rule：

```text
contact = surface_distance <= contact_threshold_m
```

signed / penetration 只作为 diagnostic 字段保留，因为开放/非 watertight hand mesh 上的 inside test 容易产生大面积假阳性。

## 旧路线处理

pre-FoundationPose 的 CAD pose prior、depth refine、DexYCB GT hand/object debug route 已从 clean upload 版本移除。历史结论保留在路线对比/QC 文档中，不再上传旧代码和旧 outputs。

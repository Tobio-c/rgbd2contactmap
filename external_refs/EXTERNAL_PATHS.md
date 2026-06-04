# External Paths Template

This clean repository does not vendor external model repositories, checkpoints, or private datasets.

Recommended local environment variables or CLI args:

```bash
export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth
export FOUNDATIONPOSE_ROOT=/path/to/FoundationPose
export WILOR_ROOT=/path/to/WiLoR
```

Example paths on the original workstation were:

```text
SAM checkpoint: /home/originflow/project/contact_pipeline/models/sam/sam_vit_b_01ec64.pth
FoundationPose: /home/originflow/project/FoundationPose
WiLoR: /home/originflow/project/WiLoR
```

Do not commit model weights, private datasets, conda envs, or local output folders to this repository.

# Implementation smoke validation — 2026-08-05

No formal continuation experiment was started. The following checks were run only to validate the standalone implementation.

## Results

- Pure unit tests: `9 passed`, including repository-relative/environment-driven path resolution.
- Full fixed indices:
  - train: 233,808 active-subtask frames;
  - validation: 22,779 active-subtask frames;
  - train task counts: PreSoakPan 77,402; KettleBoiling 44,437; LoadDishwasher 71,354; RinseSinkBasin 40,615.
- Real GR00T sample contract:
  - native three-view `panda_omron` transform loaded;
  - state remained `[1, 64]`;
  - action remained `[16, 32]`;
  - original RoboCasa transformed state occupied dimensions 0–19;
  - ordinal stage and continuous progress occupied dimensions 20 and 21;
  - both new dimensions were enabled in `state_mask`.
- Parent checkpoint `experiment_cfg/metadata.json` is now used for all state/action normalization, matching `Gr00tPolicy` inference and avoiding per-task statistics.
- Standalone repository environment check resolved every external resource through the ignored local `.env`, verified the pinned GR00T commit, and detected both NVIDIA A40 GPUs without relying on the parent directory layout.
- Two-GPU DDP forward/backward completed with:
  - per-device batch 8;
  - gradient accumulation 8;
  - effective global batch 128;
  - 1,068,806,144 trainable action-side parameters out of 2,724,163,520 total;
  - initial two-step mean training loss 0.32067.
- Checkpoint write, final model write, and local checkpoint resume paths completed after the compatibility fixes below.
- Ground-truth reference-video inference plumbing:
  - decoded `agentview_left` for a real five-stage validation demonstration;
  - produced 100 stride-8 reference nodes with the validated 3,072-D RGB32 descriptors;
  - ported the zero-penalty monotonic subsequence-DTW protocol used by the earlier held-out
    GT-video experiment;
  - validated all five uniform/nonuniform path styles, rolling stride-8 query collection,
    cross-call non-rewinding endpoints, stage confirmation, action conversion, and
    conditioned-policy dispatch;
  - expert-action simulator replay localized stage-0 endpoints exactly at nodes
    `0, 2, 4, 6, 8` for control steps `0, 16, 32, 48, 64`;
  - `19 passed` after adding DTW inference coverage.

## Issues caught by the smoke

1. Loading all model weights directly as BF16 makes GR00T's Beta/Dirichlet flow-time sampler fail. The standalone trainer now follows the official pattern: FP32 checkpoint parameters plus BF16 Trainer autocast.
2. GR00T's custom `DualBrainTrainer.save_model` signature is not compatible with a direct public call. Final saving now uses GR00T's official `safe_save_model_for_hf_trainer`.
3. Current PyTorch safe loading rejects the NumPy RNG state written by the local Hugging Face Trainer checkpoint. The `--resume` path adds only the required NumPy RNG types to PyTorch's safe allowlist.

Large smoke checkpoints were deleted after this record was written; they are not research results and can be regenerated with `scripts/launch_two_gpu.sh`.

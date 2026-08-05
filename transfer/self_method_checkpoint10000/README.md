# checkpoint-10000 inference transfer package

This directory contains only the final inference checkpoint produced by our method on this machine. It does not contain the official GR00T parent checkpoint, RoboCasa365 data, optimizer state, scheduler state, or RNG state.

## Files to copy

Copy this entire directory to the target server:

```text
self_method_checkpoint10000/
├── checkpoint-10000-inference.tar.zst
├── ARCHIVE_SHA256.txt
└── README.md
```

Archive properties:

```text
size:    5,995,314,809 bytes (about 5.6 GiB)
sha256:  4901eed6ff98713935ff954d2e03d42704698e2a3d8f8d0a41ef7718474f563a
format:  tar + zstd
```

## Verify after upload

Run inside this directory:

```bash
sha256sum -c ARCHIVE_SHA256.txt
```

The expected result is:

```text
checkpoint-10000-inference.tar.zst: OK
```

Optionally verify the compressed stream and inspect the five archived paths:

```bash
zstd -t checkpoint-10000-inference.tar.zst
tar -I zstd -tf checkpoint-10000-inference.tar.zst
```

## Extract into a cloned repository

Assuming the repository is `/workspace/flf_progress_vla`:

```bash
mkdir -p /workspace/flf_progress_vla/outputs/mvp_continuation
tar -I zstd -xf checkpoint-10000-inference.tar.zst \
  -C /workspace/flf_progress_vla/outputs/mvp_continuation
```

This creates exactly:

```text
/workspace/flf_progress_vla/outputs/mvp_continuation/checkpoint-10000/
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
└── experiment_cfg/
    └── metadata.json
```

If `MIGRATION_SHA256SUMS.txt` is present in the cloned repository, verify all five extracted files from `/workspace`:

```bash
grep 'checkpoint-10000' flf_progress_vla/MIGRATION_SHA256SUMS.txt | sha256sum -c -
```

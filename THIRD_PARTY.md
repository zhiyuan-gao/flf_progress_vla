# Pinned external dependencies

## RoboCasa Isaac-GR00T fork

- Repository: `https://github.com/robocasa-benchmark/Isaac-GR00T.git`
- Required commit: `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10`
- Package version at that commit: `gr00t 1.1.0`
- License: Apache-2.0 (see the upstream repository)

The GR00T source is deliberately not vendored into this repository. Run
`scripts/bootstrap_gr00t.sh` to clone the exact revision into the ignored
`external/Isaac-GR00T` directory, or point `GR00T_ROOT` at an existing clean
checkout of this commit.

The current validated environment uses Python 3.10.16, PyTorch 2.5.1,
TorchVision 0.20.1 and Transformers 4.51.3. Transformers 5.x is incompatible
with the dataclass configuration in this GR00T revision.

RoboCasa365 datasets and GR00T checkpoints are research artifacts governed by
their respective upstream terms. They are external resources and are never
committed to this repository.

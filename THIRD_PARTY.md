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

## RoboCasa365 data registry

- Repository: `https://github.com/robocasa/robocasa.git`
- Registry revision: `b4684e6ee37d377cc392e98302a6b916d588b415`
- Registry file: `robocasa/models/assets/box_links/box_links_ds.json`
- Selected data: exact target-human/composite snapshots listed in
  [RESOURCE_SETUP.md](RESOURCE_SETUP.md)

## GR00T N1.5 parent weights

- Hugging Face repository: `robocasa/robocasa365_checkpoints`
- Revision: `14895998fe7c8f8f2441cc8957ec2c510302758b`
- Subdirectory:
  `gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000`
- File SHA256 values and download commands: [RESOURCE_SETUP.md](RESOURCE_SETUP.md)

RoboCasa365 datasets and GR00T checkpoints are research artifacts governed by
their respective upstream terms. They are external resources and are never
committed to this repository.

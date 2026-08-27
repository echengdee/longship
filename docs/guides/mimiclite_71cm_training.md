# MimicLite 71 cm climb-turn-sit

This experiment tracks a composite G1 reference against the same `0.71 m`
cargo-floor height and `1.16 m` internal clearance used by the PHP box-truck
Sim2Sim scene.

## Build the reference

```bash
python scripts/training/build_mimiclite_71cm_motion.py
```

The generated any4hdmi dataset is
`third_party/mimiclite-assets/g1-climb-turn-sit-71cm`. It contains the real
OmniRetarget `climb_12` low-clearance ascent through frame 240 followed by a
3 second quintic 180-degree low turn and floor-sit transition. Later source
frames stand about `2.02 m` high and are deliberately excluded because the
measured cargo roof is only `1.87 m` above ground. The manifest records the
phase boundaries and source geometry.

## Longship RL platform integration

The `box71` mjlab terrain is the exact scaled `climb_12` rectangular platform
in XY, with its top raised from `0.704494 m` to `0.710000 m`. A collision roof
at `1.87 m` enforces the measured clearance. Both are added to a plane in every
parallel simulation world. The full side-opening truck remains the transfer
environment.

The canonical experiment is
`src/longship/rl/experiments/mimiclite_g1_71cm_climb_turn_sit.yaml`. It owns the
single-motion data contract, G1 model metadata, MimicLite PPO backend settings,
the `box71` scene, and the output/checkpoint contract. Inspect the exact command
without allocating the GPU:

```bash
PYTHONPATH=src python -m longship.rl.training --root "$PWD" plan \
  src/longship/rl/experiments/mimiclite_g1_71cm_climb_turn_sit.yaml \
  --output outputs/mimiclite/g1_71cm_plan
```

Install the backend environment once:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project environments/rl/mjlab
uv --project environments/rl/mjlab run aa-discover-projects
uv --project environments/rl/mjlab run aa-project enable mimic_lite
```

Start training through the platform (the output directory must be new):

```bash
PYTHONPATH=src python -m longship.rl.training --root "$PWD" run \
  src/longship/rl/experiments/mimiclite_g1_71cm_climb_turn_sit.yaml \
  --output outputs/mimiclite/g1_71cm_$(date +%Y%m%d_%H%M%S)
```

The platform writes `resolved.yaml`, the exact shell-free argument vector and
environment to `command.json`, and keeps upstream Hydra/W&B-disabled output
under `upstream/`. The default uses the Huge PPO module, 32 MJLab environments,
and initializes from `elijahgalahad/mimic_lite/xua2csee`. Change the experiment
YAML rather than the legacy wrapper when adjusting a production run.

`scripts/training/train_mimiclite_71cm.sh` remains a low-level developer
adapter; it is not the canonical RL platform entrypoint.

## Box-truck Sim2Sim

After exporting the trained policy to the MimicLite sim2real ONNX/YAML
contract, place the files under
`third_party/mimiclite-sim2real/checkpoints/mimic-lite/v1_1/` and run:

```bash
modules/longship-sim2real/scripts/sim2sim/run_mimiclite_box_truck.sh
```

The profile reuses Longship's PHP box-truck primitive and places its opening
about `0.30 m` ahead of the reference start, matching `climb_12`.

# MimicLite Sim2Sim

Longship runs MimicLite v1.1 with the upstream `sim2real.Tracking` observation,
history, motion and action pipeline. Only robot I/O and keyboard control are
adapted to Longship's shared Unitree DDS contract.

## Assets

The G1 mode-15 MJCF and `walk1_subject1.npz` reference motion are stored under
`third_party/mimiclite-assets`. The released policy files are intentionally not
in Git and must be copied from the upstream artifact store with its required
`gdrive:` rclone remote:

```bash
modules/longship-sim2real/scripts/sim2sim/sync_mimiclite_artifacts.sh
```

This produces:

```text
third_party/mimiclite-sim2real/checkpoints/mimic-lite/v1_1/policy.yaml
third_party/mimiclite-sim2real/checkpoints/mimic-lite/v1_1/policy.onnx
```

## Validate and run

```bash
python -m longship.rl.sim2sim preflight mimiclite --root "$PWD"
modules/longship-sim2real/scripts/sim2sim/run_mimiclite.sh
```

Controls are `i` to interpolate to the default pose, `]` to enable the policy
and start the reference, `p` to pause/resume, `r` to restart, and `o` to hold.

The profile uses CPU ONNX Runtime by default. Set `policy.provider: cuda` in a
profile override to select the upstream `onnx-gpu` backend.

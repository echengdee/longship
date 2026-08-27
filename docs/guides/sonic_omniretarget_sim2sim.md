# SONIC tracking an OmniRetarget trajectory

This integration reuses Longship's existing simulation runtime. It does not install Drake or
create another Conda environment.

## Runtime

The launcher selects the first Python interpreter that provides:

- MuJoCo
- ONNX Runtime
- CycloneDDS and Unitree SDK2 Python
- ZeroMQ

On the current workstation this is:

```text
/home/qcraft/miniconda3/envs/env_isaaclab511/bin/python
```

The dataset is stored at:

```text
third_party/OmniRetarget_Dataset
```

## Run

```bash
cd /home/qcraft/longship
bash modules/longship-sim2real/scripts/sim2sim/run_sonic_omniretarget.sh
```

Keyboard control remains in the terminal:

- `i`: interpolate to the selected trajectory's first pose
- `]`: enable SONIC tracking; it holds the first frame briefly, then starts playback
- `Ctrl-C`: stop the simulator and controller

The MuJoCo window keeps its own camera and gantry keys. The external trajectory owns the motion,
so locomotion keys are rejected while it is active.

## Data path

The profile `sonic_omniretarget.yaml` selects an upright portion of
`sub3_largebox_003_original.npz`, resamples it to 50 Hz, converts the published MuJoCo joint order
to SONIC's IsaacLab order, aligns the initial heading, and feeds ten future frames to SONIC's
released G1 encoder. The same reference initializes a free dynamic box in the shared MuJoCo
simulator.

## Validation status

The transport and inference path is verified: SONIC runs at approximately 50 Hz and MuJoCo
receives low-level commands through the shared DDS topics. SONIC's own planner and its bundled
`award_v2r_smooth` reference remain stable through this Python ONNX path.

The released SONIC policy does not stably track the OmniRetarget box clip after gantry release.
This is a policy/reference compatibility limitation, not a transport failure. A production box
carry result requires either a SONIC checkpoint trained on this reference distribution or the
object-aware HoloSoma WBT policy trained from the same OmniRetarget data.

# NoMaD Runtime Guidance

The package in this directory owns PyTorch model construction, checkpoint
loading, decoded-tensor image ingress, tensor preprocessing, NoMaD inference,
and offline model-aware topomap selection. Camera capture and decoding, online
goal selection, ROS or robot SDK transport, waypoint selection, controller
scaling, and safety integration remain outside this package until their
interfaces are designed explicitly.

The plugin manifest must keep `contracts`, `exports`, and `supported_targets`
empty until those boundaries are implemented and qualified. Do not turn raw
NoMaD trajectories into Longship commands in this package.

## Image Tensor Contract

There are two layers to the image contract.

`canonicalize_image()` accepts one decoded image tensor in HWC or CHW layout,
RGB or BGR channel order, and either `uint8` or floating-point dtype. It returns
contiguous `float32` RGB CHW values in `[0, 1]`. Byte-range floating-point
inputs must be declared explicitly with `value_range="byte"`; do not infer a
float tensor's range from its content.

`ObservationBuffer` stores four canonical frames with strictly increasing
caller-provided timestamps. It rejects a resolution change until `clear()` is
called. Its snapshot order is oldest to newest:

```text
[t-3, t-2, t-1, t]
```

The buffer owns a copy of each submitted frame because camera backends commonly
reuse capture memory. Timestamp clock selection and frame sampling cadence are
the caller's responsibility.

`NomadPolicy.infer()` remains the narrow canonical tensor interface and accepts
floating-point RGB tensors with values in `[0, 1]`.

Observation context:

```text
single: [4, 3, H, W]
batch:  [B, 4, 3, H, W]
```

Goal image:

```text
single: [3, H, W]
batch:  [B, 3, H, W]
```

Requirements:

- Direct policy inputs are channel-first RGB, not HWC or OpenCV BGR.
- Direct policy inputs must use a floating-point dtype; integer and `uint8`
  tensors are rejected. Use `canonicalize_image()` for decoded camera tensors.
- Observation and goal batch sizes must match.
- Input height and width may vary. The runtime resizes images to `96 x 96` and
  applies ImageNet normalization.
- The runtime moves input tensors to the model device.
- Goal-conditioned navigation uses `goal_mask=False`, which is the default.
  Use `goal_mask=True` only for deliberate goal-masked exploration.

Typical tensor-only camera conversion is:

```python
image_tensor = canonicalize_image(
    decoded_image,
    ImageTensorSpec(layout="hwc", channel_order="bgr"),
)
```

## Output Contract

For batch size `B`, `num_samples=N`, and the released checkpoint:

```text
distance: [B]
actions:  [B, N, 8, 2]
```

Actions are raw NoMaD robot-frame waypoint trajectories. Do not silently pick
a sample, pick a waypoint, apply the old LoCoBot `MAX_V / RATE` scale, or
convert the result to a robot command inside the model runtime. Those choices
belong to the future integration and safety layer.

## Validation

Keep the runtime free of ROS, diffusers, diffusion-policy, torchvision, NumPy,
Pillow, PyYAML, and wandb runtime dependencies. The only non-standard runtime
dependencies are PyTorch and `efficientnet-pytorch==0.7.1`.

The adaptive topomap command may use Pillow through the `offline` optional
dependency. Pillow must not become a dependency of model loading or tensor-only
inference. Generated topomaps are local artifacts and are not committed by
default.

Run tests with:

```bash
cd plugins/policies/visual_navigation/nomad
PYTHONPATH="$PWD" pytest -q tests
```

Run the released checkpoint integration test by also setting:

```bash
NOMAD_CHECKPOINT=/absolute/path/to/nomad.pth
```

Never add model checkpoints or generated inference images to git by default.
The repository's explicitly approved exception is the released checkpoint at
`models/nomad/nomad.pth`, which must remain Git-LFS tracked.

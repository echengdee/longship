# RealSense RGB-D FollowScene provider (experimental)

This plugin is the optional on-device perception edge for FollowPerson V0. One
process owns the RGB and aligned depth streams, publishes an atomic
`longship.follow-scene.v1` observation, and exposes a read-only camera page. It
has no command or actuator channel.

The provider requires separately installed `pyrealsense2`, OpenCV, and NumPy.
Its bundled-OpenCV HOG detector plus short IoU tracking is only a baseline. It
does not provide appearance re-identification and is not qualified for crowds,
long occlusion, cropped people, or unsupervised operation.

The checked-in calibration file is intentionally disabled. Replace its matrix
with a surveyed camera-optical-to-`base_link` rigid transform and mark it
confirmed only after the site checks in the
[FollowPerson guide](../../../docs/guides/follow-person-v0.md).

```bash
python3 plugins/perception/realsense_rgbd_follow/worker.py \
  --calibration /absolute/path/to/reviewed-camera-extrinsic.json \
  --host 127.0.0.1 \
  --port 8780
```

Endpoints:

- `/health` reports frame age and calibration/detector/floor gates;
- `/v1/follow-scene` publishes the atomic contract;
- `/preview.jpg` carries the sequence-paired diagnostic frame; and
- `/` is a read-only browser page.

Bind to loopback unless the robot LAN is trusted or protected by an SSH tunnel.

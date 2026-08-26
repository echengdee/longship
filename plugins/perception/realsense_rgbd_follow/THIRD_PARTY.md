# Third-party boundary

This plugin imports separately installed `pyrealsense2`, OpenCV, and NumPy at
runtime. Longship does not vendor their source, binaries, camera firmware, or
model assets. The baseline person detector is OpenCV's installed default HOG
detector; no detector weights are copied into this repository.

Review the installed versions, licenses, platform support, and target-specific
performance before use. The plugin is an experimental perception provider and
has no actuator authority.

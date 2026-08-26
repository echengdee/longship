from __future__ import annotations

import argparse
import time
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="View an external G1 MJCF without copying or actuating it"
    )
    parser.add_argument("--scene", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise SystemExit(
            "BLOCKED: install Longship's optional mujoco dependency"
        ) from exc
    try:
        model = mujoco.MjModel.from_xml_path(str(args.scene.resolve(strict=True)))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        viewer = mujoco.viewer.launch_passive(model, data)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"BLOCKED: cannot load external G1 scene: {exc}") from exc
    print("Asset inspection only: no control loop or locomotion policy is active.")
    try:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

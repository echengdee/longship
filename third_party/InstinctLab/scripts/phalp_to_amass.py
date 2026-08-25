import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

import joblib
import quaternion as npq

CAM_ROT = npq.from_euler_angles([0.0, -np.pi / 2, 0.0]) * npq.from_euler_angles([np.pi / 2, 0.0, 0.0])
# CAM_ROT = npq.from_euler_angles([0., -np.pi/6, 0.]) * CAM_ROT # Add pitch if you see the video camera is not horizontal
CAM_POS = np.array([0, 0, 0.4])


def main(args):
    """Transforming PHALP motion tracking data to AMASS format pose file"""
    results = joblib.load(args.input)

    num_frames = len(results.keys())
    poses = np.zeros(
        (
            num_frames,  # number of frames
            24,
            3,
        )
    )
    trans = np.zeros(
        (
            num_frames,
            3,
        )
    )
    for frame_idx, key in tqdm.tqdm(enumerate(results.keys())):
        # assuming the keys are in frame order
        frame = results[key]

        if len(frame["size"]) == 0:
            print("No size info in frame", frame_idx)
            poses = poses[:frame_idx]
            trans = trans[:frame_idx]
            break

        img_H, img_W = frame["size"][0]
        fx = args.focal_x
        fy = fx * img_H / img_W

        # root pose in camera frame
        trans_ = frame["camera"][0]
        trans_ = (
            np.array(
                [
                    trans_[0] / fx * trans_[2],
                    trans_[1] / fy * trans_[2],
                    trans_[2],
                ]
            )
            / 1000.0
        )  # mm to m
        root_quat = npq.from_rotation_matrix(frame["smpl"][0]["global_orient"])  # (1, 4)

        # root pose in world frame
        trans_ = (
            npq.rotate_vectors(
                CAM_ROT,
                trans_,
            )
            + CAM_POS
        )
        root_quat = CAM_ROT * root_quat

        trans[frame_idx] = trans_
        body_quat = npq.from_rotation_matrix(frame["smpl"][0]["body_pose"])  # (23, 4)
        poses[frame_idx] = npq.as_rotation_vector(
            np.concatenate(
                [
                    root_quat,
                    body_quat,
                ]
            )
        )

    # plt.plot(np.arange(num_frames) * 1/args.fps, trans[:, 0], label="x")
    # plt.plot(np.arange(num_frames) * 1/args.fps, trans[:, 1], label="y")
    # plt.plot(np.arange(num_frames) * 1/args.fps, trans[:, 2], label="z")
    # plt.plot(np.arange(num_frames) * 1/args.fps, poses[:, 0, 0], label="x")
    # plt.plot(np.arange(num_frames) * 1/args.fps, poses[:, 0, 1], label="y")
    # plt.plot(np.arange(num_frames) * 1/args.fps, poses[:, 0, 2], label="z")
    # plt.legend()
    # plt.show()

    # write to AMASS format pose file
    data = {
        "poses": poses,
        "trans": trans,
        "mocap_framerate": args.fps,
    }
    np.savez(args.output, **data)  # type: ignore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input PHALP motion tracking data file")
    parser.add_argument("--output", type=str, help="Output AMASS format pose file")
    parser.add_argument("--focal_x", type=float, default=0.4, help="Normalized focal length of the camera in x-axis")
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help=(
            "Frame rate of the motion tracking data, write directly to the file because human-mesh-reconstruction"
            " system does not know this info"
        ),
    )

    args = parser.parse_args()
    main(args)

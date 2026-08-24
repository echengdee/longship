"""Command-line checkpoint and random-input smoke test."""

import argparse
import json
from pathlib import Path

import torch

from nomad_runtime import NomadConfig, NomadPolicy, default_checkpoint_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a NoMaD checkpoint and run random tensor inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="NoMaD checkpoint (default: repository LFS asset).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-samples", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    config = NomadConfig()
    policy = NomadPolicy.from_checkpoint(
        args.checkpoint, config=config, device=device, strict=True
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    observations = torch.rand(
        (1, config.observation_frames, 3, *config.image_size),
        device=device,
        generator=generator,
    )
    goal = torch.rand(
        (1, 3, *config.image_size),
        device=device,
        generator=generator,
    )
    output = policy.infer(
        observations,
        goal,
        num_samples=args.num_samples,
        generator=generator,
    )
    summary = {
        "device": str(policy.device),
        "checkpoint": str(args.checkpoint),
        "parameters": sum(
            parameter.numel() for parameter in policy.model.parameters()
        ),
        "distance_shape": list(output.distance.shape),
        "action_shape": list(output.actions.shape),
        "distance": output.distance.cpu().tolist(),
        "actions_finite": bool(torch.isfinite(output.actions).all().item()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

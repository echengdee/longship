import os
import subprocess
from datetime import datetime


def main(args):
    """Automatically play the last n runs of the specified task."""
    # specify directory for logging experiments
    expdir = os.path.join("logs", "instinct_rl", args.expdir)
    expdir = os.path.abspath(expdir)

    # get the last n runs
    if args.n_runs:
        all_runs = os.listdir(expdir)
        all_runs.sort(key=lambda x: datetime.strptime("_".join(x.split("_")[:2]), "%Y%m%d_%H%M%S"))
        runs_to_play = all_runs[-args.n_runs :]
        runs_to_play = ["_".join(x.split("_")[:2]) + ".*" for x in runs_to_play]
        print(f"No run specified, getting the latest {args.n_runs} runs")
    else:
        runs_to_play = args.log_runs

    for run_name in runs_to_play:
        print(f"Playing run: {run_name}")
        # play the run
        subprocess.run(
            [
                "python",
                "scripts/instinct_rl/play.py",
                "--task",
                args.task,
                "--load_run",
                run_name,
            ]
            + extra_args
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expdir", type=str, default="g1_shadowing")
    parser.add_argument("--task", type=str, default="Instinct-Shadowing-Plane-PartBody-MultiReward-G1-Play-v0")
    parser.add_argument("-n", "--n_runs", type=int, default=0, help="number of last runs to play")
    parser.add_argument(
        "-l",
        "--log_runs",
        type=str,
        nargs="+",
        default=[],
    )

    args, extra_args = parser.parse_known_args()
    main(args)

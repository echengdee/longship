import os
import yaml

SUPPORTED_MOTION_ENDINGS = ["poses.npz", "retargeted.npz"]
SUPPORTED_TERRAIN_ENDINGS = ["ply", "stl"]


def main(args):
    """This is a function that automatically scans the terrain directory and log all terrain files
    and the motion files that are matched to them in a YAML file, as the metadata.yaml file for the TerrainMotionCfg.

    The input directory should contains a batch of (nested) folders. Each folder should contain:
        - A **single** terrain file.
        - Multiple (>=1) motion files.

    The output YAML file will be in the following structure:
    terrains:
        - terrain_id: "terrain1"
          terrain_file: "terrain1.ply"
        - terrain_id: "terrain2"
          terrain_file: "terrain2.stl"
    motion_files:
        - terrain_id: "terrain1"
          motion_file: "motion1.poses.npz"
        - terrain_id: "terrain1"
          motion_file: "motion2.retargeted.npz"
        - terrain_id: "terrain2"
          motion_file: "motion1.poses.npz"
        - terrain_id: "terrain2"
          motion_file: "motion2.retargeted.npz"
    """

    terrains = []
    motion_files = []
    terrain_id = 0
    for root, _, files in os.walk(args.path, followlinks=True):
        # NOTE: assuming in a folder, there is only one terrain file
        for f in files:
            if any(f.endswith(ending) for ending in SUPPORTED_TERRAIN_ENDINGS):
                terrain_file = os.path.join(root, f)
                terrain_file = os.path.relpath(terrain_file, args.path)
                terrains.append(
                    {
                        "terrain_id": terrain_id,
                        "terrain_file": terrain_file,
                    }
                )
                terrain_id = terrain_id + 1
        for f in files:
            if any(f.endswith(ending) for ending in SUPPORTED_MOTION_ENDINGS):
                motion_file = os.path.join(root, f)
                motion_file = os.path.relpath(motion_file, args.path)
                # search the motion file in current dir (root) whether it matches any terrain_id
                potential_terrain_ids = [
                    os.path.splitext(f)[0]
                    for f in os.listdir(root)
                    if any(f.endswith(ending) for ending in SUPPORTED_TERRAIN_ENDINGS)
                ]
                if not len(potential_terrain_ids) == 1:
                    print(
                        f"Found multiple terrain files in {root} for motion file {motion_file}. "
                        f"Use the first terrain file {potential_terrain_ids[0]} as the match."
                        "Please ensure that each motion file is matched to exactly one terrain."
                    )
                motion_files.append(
                    {
                        "terrain_id": terrains[-1]["terrain_id"],
                        "motion_file": motion_file,
                    }
                )

    # Create the metadata dictionary
    metadata = {
        "terrains": terrains,
        "motion_files": motion_files,
    }
    # Write the metadata to a YAML file
    if args.output_yaml is None:
        args.output_yaml = os.path.join(args.path, "metadata.yaml")
    with open(args.output_yaml, "w") as yaml_file:
        yaml.safe_dump(metadata, yaml_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate metadata for motion-matched terrain files.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Directory containing terrain files and motion files.",
    )
    parser.add_argument(
        "--output_yaml",
        type=str,
        required=False,
        default=None,
        help="Output YAML file to save the metadata.",
    )

    args = parser.parse_args()
    main(args)

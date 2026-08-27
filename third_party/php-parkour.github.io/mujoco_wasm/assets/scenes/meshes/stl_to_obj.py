#!/usr/bin/env python3
import argparse
from pathlib import Path
import trimesh

def convert(stl_path: Path, obj_path: Path, unify: bool, repair: bool):
    mesh = trimesh.load(stl_path, force='mesh')

    # trimesh.load can return a Scene in some cases; handle it.
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(
            g for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a mesh: {type(mesh)}")

    if repair:
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
        mesh.process(validate=True)

    if unify:
        # Merge close vertices; useful if STL has many duplicates
        mesh.merge_vertices()

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(obj_path, file_type="obj")
    print(f"Converted: {stl_path} -> {obj_path}")

def convert_directory(directory: Path, unify: bool, repair: bool):
    """Convert all STL files in a directory to OBJ format."""
    # Find all STL files (case-insensitive)
    stl_files = list(directory.glob("*.STL")) + list(directory.glob("*.stl"))
    
    if not stl_files:
        print(f"No STL files found in {directory}")
        return
    
    print(f"Found {len(stl_files)} STL file(s) to convert")
    
    for stl_path in stl_files:
        obj_path = stl_path.with_suffix(".obj")
        try:
            convert(stl_path, obj_path, unify=unify, repair=repair)
        except Exception as e:
            print(f"Error converting {stl_path}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Convert STL to OBJ locally.")
    ap.add_argument("input", type=Path, help="Input STL file or directory containing STL files")
    ap.add_argument("output_obj", type=Path, nargs="?", default=None, help="Output OBJ file (only used when input is a single file)")
    ap.add_argument("--unify", action="store_true", help="Merge duplicate/near vertices")
    ap.add_argument("--repair", action="store_true", help="Attempt basic mesh cleanup")
    args = ap.parse_args()

    input_path = args.input
    
    # Check if input is a directory
    if input_path.is_dir():
        convert_directory(input_path, unify=args.unify, repair=args.repair)
    elif input_path.is_file():
        # Single file conversion (original behavior)
        out = args.output_obj
        if out is None:
            out = input_path.with_suffix(".obj")
        convert(input_path, out, unify=args.unify, repair=args.repair)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return 1

if __name__ == "__main__":
    main()

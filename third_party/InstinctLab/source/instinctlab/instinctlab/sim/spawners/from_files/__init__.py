# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Sub-module for spawning assets from mesh files (OBJ, STL, FBX)."""

from .from_files import spawn_from_mesh
from .from_files_cfg import MeshFileCfg

__all__ = ["spawn_from_mesh", "MeshFileCfg"]

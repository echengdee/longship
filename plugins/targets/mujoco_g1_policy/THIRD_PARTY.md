# Third-party record: external Unitree RL Gym G1 provider

Longship does not redistribute the inspected G1 model, meshes, policy,
configuration, or license. The operator supplies an external installation and
locks every activated artifact by content digest.

The machine-readable
[`model-artifacts.experimental.json`](model-artifacts.experimental.json) uses
Longship's governed artifact schema for the three regular files. It declares
the checkpoint license `NOASSERTION`, redistribution `reference_only`, and the
artifacts as gated access, so automatic prefetch remains blocked. Loading
already-authorized local bytes is verification, not a license or redistribution
approval. The MJCF/mesh directory is separately covered by the deterministic
bundle digest below.

Locally inspected record:

- stated project: Unitree RL Gym;
- stated upstream: <https://github.com/unitreerobotics/unitree_rl_gym>;
- stated repository license: BSD-3-Clause;
- external license file SHA-256:
  `aef6394ba1597725a68308167324e675f562e6606027404deb1b9da254c2b9c1`;
- external G1 asset-directory digest under Longship's deterministic
  directory-hash algorithm:
  `f569b1425fc055ca759699f36f94eba97663db547b79e663bafa50560a0c9349`;
- external TorchScript policy SHA-256:
  `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`;
- external policy configuration SHA-256:
  `73044e7d355c61915695c16d6e09eb3efef46eec1e3d708fd3eb9157dfe3bbbb`;
- verified public upstream revision:
  `276801e46c5d433564f24658bac64f254b7d2d4b`.

The three public files at that revision were independently downloaded and
their SHA-256 values matched the inspected local files. The content digests
remain the identities enforced by the runner. A reviewer must re-establish
provenance, license compatibility, and behavioral compatibility before
accepting any other artifact set or redistributing third-party bytes. This
inventory is not legal advice or physical-robot qualification.

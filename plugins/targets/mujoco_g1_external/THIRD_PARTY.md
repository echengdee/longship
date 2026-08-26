# Third-party record: external Unitree MuJoCo assets

Longship does not redistribute the inspected G1 model, meshes, simulator, or
policy. The operator supplies an external installation and explicitly locks it
by content hash before use.

Locally inspected record:

- stated project: Unitree MuJoCo;
- stated upstream: <https://github.com/unitreerobotics/unitree_mujoco>;
- copyright notice: HangZhou YuShu TECHNOLOGY CO.,LTD. / Unitree Robotics;
- stated repository license: BSD-3-Clause;
- external license file SHA-256:
  `a5d73fc4aca9074e3e6fe0b1a0ba763cf9514b2249b7390ed20fe8d53630bf25`;
- external G1 directory digest under Longship's deterministic bundle-hash
  algorithm:
  `9ba04edacbaf9bda13bf847e99e845e9b36b27f7b2141e48ccfc8cae211d1f39`;
The inspected external directory was not treated as proof of an upstream Git
revision. Its bundle digest is therefore the identity checked by the plugin. A
reviewer must re-establish provenance and license compatibility before using a
different bundle or redistributing any third-party bytes.

No locomotion-policy license was established by this inspection. Policy bytes
are not accepted or activated by the asset-only plugin. Dynamic G1 simulation
requires its own artifact, license, interface, digest, and qualification
record.

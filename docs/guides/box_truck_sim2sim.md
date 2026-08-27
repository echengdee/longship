# Box-truck Sim2Sim scene

The `php_box_truck` profile adds a static side-opening electric box truck to the
released PHP MuJoCo scene. It is intended for perception, approach, stepping,
and cargo-entry experiments rather than vehicle dynamics.

Measured dimensions supplied for the scene:

- side-door clear length: `2.01 m`;
- side-door clear height: `1.16 m`;
- cargo-floor height above ground: `0.71 m`.

The cargo-box length (`3.20 m`), width (`1.75 m`), wall thickness (`0.05 m`),
cab, chassis, wheel, and raised-door geometry are engineering approximations
from the reference photograph. They can be changed under
`simulator.box_truck` in `php_box_truck.yaml` without changing code.

Run the scene with:

```bash
./modules/longship-sim2real/scripts/sim2sim/run_php_box_truck.sh
```

The default truck pose is `(3, 0, 0)` with yaw `-90 degrees`. This rotates the
open side toward the robot, placing the cargo threshold roughly `2.125 m` in
front of its initial position. The truck is fixed to the world; all exposed
floor, wall, roof, raised-door, chassis, cab, bumper, and wheel primitives have
MuJoCo collision enabled.

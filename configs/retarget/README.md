# SOMA retarget presets

| Preset | SOMA target | Default output |
|---|---|---|
| `g1.json` | `unitree_g1` | `data/lafan1-retarget-g1/` |
| `a3.json` | `agibot-a3t3` | `data/lafan1-retarget-a3/` |
| `x2_ultra.json` | `agibot_x2ultra` | `data/lafan1-retarget-x2_ultra/` |

All presets default to `data/lafan1-soma-aligned/` as input. Paths in the JSON
resolve from the repository root. Explicit `--input` and `--output` paths resolve
from the caller's working directory. `--batch-size` overrides the JSON value.

```bash
bash scripts/run_local.sh scripts/data/retarget.py --robot g1 --print-config
bash scripts/run_local.sh scripts/data/retarget.py --robot g1 --batch-size 2
```

These JSON files configure the batch job. IK matching, human-to-robot scaling and
postprocessing settings are provided by the pinned SOMA target registry under
`vendor/soma-retargeter/soma_retargeter/assets/robotics/`. X2 additionally applies
the checked-in `soma_override` for terminal-palm IK and its matching calibration
rotation. The launcher creates a temporary SOMA registry override; vendor files
remain unchanged. Details and migration steps: [X2 wrist correction](../../docs/X2_WRIST_FIX.md).

G1's SOMA description is
`unitree/g1/desc/mjcf/g1_29dof_rev_1_0.xml`; its training description is the separate
local `assets/robots/g1/robot.xml` imported from mjlab. They share the 29-joint
schema but are different source assets. Editing one does not update the other.

The published G1 baseline uses pre-retargeted ModelScope CSVs. That path needs
neither SOMA nor `g1.json`, which explains its absence in the original presets.
New G1 retargets go into a separate output folder and must be passed explicitly
to `scripts/data/prepare_lafan.py --source ...`. Their tracking quality requires
separate evaluation; the historical benchmark numbers do not apply automatically.

See [data preparation](../../docs/DATA.md) for BVH adaptation, environment and
asset prerequisites. No datasets are downloaded by the retarget command.

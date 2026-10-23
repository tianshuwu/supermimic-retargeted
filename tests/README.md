# Regression tests

This directory contains reusable pytest tests, including upstream numerical
checks and regressions added for robot adaptation and repository restructuring.
It contains test source, not saved local test results. Temporary fixtures go to
pytest's temporary directory (`.cache/tmp/` with `scripts/run_local.sh`); local
logs belong under `logs/` and are ignored by Git.

Run from the repository root with the development dependencies installed. The
main suite assumes G1; A3/X2 tests launch separate processes to select their
dimensions. First build the G1 assets; they are intentionally absent from Git:

```bash
bash scripts/run_local.sh scripts/setup/build_robot_assets.py --robot g1
# CPU checks with synthetic inputs; no dataset download required
SM_ROBOT=g1 bash scripts/run_local.sh -m pytest tests -m "not gpu and not dataset and not slow" -q -ra
# With CUDA and LAFAN prepared, include environment and bounded training regressions
SM_ROBOT=g1 bash scripts/run_local.sh -m pytest tests -m "not slow" -q -ra
```

| Tests | What they check | Additional prerequisites |
|---|---|---|
| `test_entrypoints.py`, `test_source_tracking.py`, `test_retarget.py` | CLI selection/imports, Git ignore boundaries, retarget configs, G1 CSV units and schema | Main environment; A3/X2 CLI cases skip if profiles are absent |
| `test_local_robot_assets.py`, `test_g1_fk.py` | Local G1 asset independence, MuJoCo/Torch FK, gradients | Built G1; one FK comparison needs CUDA |
| `test_embodiments.py` | A3/X2 FK, velocities and gradients | Corresponding built assets; otherwise skip |
| `test_x2_wrist_tracking.py` | Palm reward gradients, reference modes, calibrated SOMA frame change and old-profile rejection | X2 assets/sources for model cases; synthetic motion, no dataset download |
| `test_hand_tracking_acceptance.py` | Reject apparent left/right parity caused by right-hand degradation, persistent roll error or early termination | CPU; synthetic metric reports |
| `test_preparation.py` | Two spawn workers match serial conversion on synthetic motion | Built G1; no downloaded NPZ baseline needed |
| `test_motion_convert.py` | Real LAFAN resampling and CPU FK against reference calculations | `data/lafan1-retargeted/g1/walk1_subject1.csv` |
| `test_env_smoke.py`, `test_analytic_heads.py`, `test_motion_library_compact.py` | Simulation, replay, analytical rewards/observations, compact storage | CUDA and `data/npz/g1/lafan1/*.npz` |
| `test_world_model.py`, `test_imagination.py` | State math, target returns, bounded training/resume and model fitting | Math checks run on CPU; simulation cases need CUDA and prepared LAFAN |

Markers: `gpu` requires CUDA; `dataset` requires downloaded or prepared LAFAN;
`slow` is the longer world-model fitting check, excluded from the commands above.
`-ra` prints skip reasons. A skipped dependency-dependent test has not passed.
Use the [data guide](../docs/DATA.md) to prepare the real-data prerequisites.

These checks verify implementation behavior. They do not establish converged
tracking quality, reproduce the historical benchmark scores, or validate hardware.
The source-export check in `scripts/dev/check_source_distribution.py` runs the
CPU/no-dataset suite in a clean Git archive using the installed dependencies.

## Maintainer release check

Ordinary users do not need this command. Before publishing source changes,
maintainers can run:

```bash
bash scripts/run_local.sh scripts/dev/check_source_distribution.py
```

It checks that required Python files are tracked by Git and tests a clean archive
of the staged source. This catches files that exist locally but would be absent
from a clone, such as Python modules accidentally excluded by an ignore rule.

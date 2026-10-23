# Command entrypoints

Run commands from the repository root using `bash scripts/run_local.sh <script> ...`
or the already installed `.venv/bin/python`. The launcher never synchronizes dependencies.

| Directory | Scripts | Purpose |
|---|---|---|
| `setup/` | `fetch_upstreams.py`, `build_robot_assets.py` | Fetch pinned external sources; generate local robot assets |
| `data/` | `prepare_lafan_bvh.py`, `align_lafan_bone_frames.py`, `retarget.py` | Adapt BVH and retarget to G1/A3/X2 |
| `data/` | `prepare_lafan.py`, `csv_to_npz_batch.py` | Prepare filtered references; lower-level CSV conversion with per-clip FPS support |
| `data/` | `download_bones_gitcode.sh`, `select_bones.py`, `extract_bones_subset.py` | Optional BONES download, deterministic selection and extraction |
| `data/` | `filter_bones_seed.py`, `finalize_manifest.py`, `build_library.py` | Feasibility filtering, accepted clip lists and packed training libraries |
| `training/` | `train.py` | Training CLI, including configuration preview and bounded smoke preset |
| `evaluation/` | `eval_benchmark.py`, `play.py`, `render_policy.py` | Deterministic evaluation, interactive playback and reference-overlay video |
| `dev/` | `check_source_distribution.py` | Check an exported Git-index source tree before release |

The root retains only two cross-purpose launchers: `run_local.sh` chooses the
existing Python environment; `robot.py ROBOT <script-or-module> ...` selects the
robot before importing modules that depend on its joint dimensions.

`align_lafan_bone_frames.py` runs with `vendor/soma-retargeter/.venv/bin/python`.
`retarget.py` runs with the main interpreter and invokes that isolated SOMA
environment itself. `download_bones_gitcode.sh` runs with Bash.

Former root-level script paths moved into the directories above. Update external
shell scripts to use the new paths. The installed `sm-train` command and
`python -m supermimic.scripts.train` remain available. Numerical training code
stays together in `src/supermimic/`; these directories organize command entrypoints.

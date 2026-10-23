# Data preparation

All commands run from the repository root. Large raw/processed data are ignored by Git. Manifest files contain names and selection metadata, not motion arrays.

## Directory roles

| Path | Contents |
|---|---|
| `data/lafan1-source-repo/`, `data/lafan1-source/` | Source checkout/archive and extracted original BVH |
| `data/lafan1-retargeted/` | Downloaded retargeted dataset checkout, including G1 CSVs and provider files |
| `data/lafan1-named/`, `data/lafan1-soma-aligned/` | Intermediate BVH skeleton/frame adaptation |
| `data/lafan1-retarget-<robot>/` | Locally generated SOMA CSVs |
| `data/npz/<robot>/<dataset>/` | Training/evaluation motion arrays and per-run filtered manifests |
| `data/library/` | Packed motion libraries for larger datasets |
| `data/manifests/` | Locally generated selections; published lists live in top-level `manifests/` |
| `logs/data/` | Download/conversion audits and extraction hash-verification records |

Thus `data/` holds the entire data preparation lifecycle, including validation
motions; it is not exclusively ready-to-train data. Checkpoints and training
configuration belong in `logs/<run>/`. Historical public evaluation numbers live
in [`docs/results/`](results/README.md). Operational scripts belong in `scripts/data/`.

## LAFAN1

Obtain the original BVH files from the Ubisoft source and the 40 G1 retargeted CSVs from ModelScope; links and revisions are in [UPSTREAM.md](../UPSTREAM.md). Unpack them locally. G1 uses the existing G1 CSVs; A3/X2 use the same selected original BVHs, renamed and frame-aligned for SOMA.

```bash
# G1: replace the source directory with your downloaded CSV folder
bash scripts/run_local.sh scripts/data/prepare_lafan.py --robot g1 \
  --source data/lafan1-retargeted/g1 --dest data/npz/g1/lafan1 --input-fps 30

# Shared raw BVH adapter: replace --input with your unpacked Ubisoft BVH directory
bash scripts/run_local.sh scripts/data/prepare_lafan_bvh.py \
  --input data/lafan1-source --output data/lafan1-named
vendor/soma-retargeter/.venv/bin/python scripts/data/align_lafan_bone_frames.py \
  --input data/lafan1-named --output data/lafan1-soma-aligned --pelvis-pivot

# Repeat for --robot x2_ultra
bash scripts/run_local.sh scripts/data/retarget.py --robot a3 \
  --input data/lafan1-soma-aligned --output data/lafan1-retarget-a3 --batch-size 2
bash scripts/run_local.sh scripts/data/prepare_lafan.py --robot a3 \
  --source data/lafan1-retarget-a3 --dest data/npz/a3/lafan1 --input-fps 30
```

The LAFAN adapter first renames joints simultaneously, then uses Left Rx(pi)/Right Ry(pi) frame gauges and an anatomical pelvis-pivot correction. A name-only BVH conversion was insufficient. The alignment script checks non-root world-joint position invariance. `prepare_lafan.py` performs bounded leg/contact correction and records repair magnitudes; this QC is not proof that a motion is dynamically feasible.

### Optional G1 retargeting from BVH

The original G1 path uses the downloaded ModelScope CSVs, so it does not need
SOMA. To generate new G1 CSVs, install the isolated SOMA environment as described
in the README, then fetch its G1 robot description/meshes (Git LFS is required):

```bash
git -C vendor/soma-retargeter lfs pull --include='soma_retargeter/assets/robotics/unitree/g1/**' --exclude=''
# Run the shared BVH rename/alignment commands above first.
bash scripts/run_local.sh scripts/data/retarget.py --robot g1 --print-config
bash scripts/run_local.sh scripts/data/retarget.py --robot g1 --batch-size 2
bash scripts/run_local.sh scripts/data/prepare_lafan.py --robot g1 \
  --source data/lafan1-retarget-g1 --dest data/npz/g1/lafan1-soma --input-fps 30
```

`configs/retarget/{g1,a3,x2_ultra}.json` select the corresponding SOMA target.
Omitted input/output/batch options use these presets; explicit CLI options take
precedence. Preset paths resolve from the repository root. Training the new G1
references requires `--data.npz-dir data/npz/g1/lafan1-soma`.

SOMA uses its own pinned G1 MJCF for IK. Training and NPZ FK use the local mjlab
copy under `assets/robots/g1/`. The converter accepts both SOMA's plain G1 joint
names and BONES' `_dof` names, checking the 29-joint order and converting
centimetres/degrees into metres/radians. The headerless ModelScope format remains
supported. New retargets are separate references; the published G1 baseline
numbers apply to the original ModelScope inputs.

## BONES-SEED

Use the dataset provider or the optional anonymous mirror downloader:

```bash
bash scripts/data/download_bones_gitcode.sh
```

The G1 and SOMA archives together require about 69 GB compressed, plus extraction/processed-data space. The script does not request a Hugging Face token. Mirror availability can change; dataset license terms still apply. `extract_bones_subset.py` verifies archive SHA256 before extracting selected files.

```bash
# Generate nested 2k/10k selections and a capture-group-separated validation set
bash scripts/run_local.sh scripts/data/select_bones.py
# Alternatively, use the published experiment selections
mkdir -p data/manifests/bones
cp manifests/bones/* data/manifests/bones/

# G1 subset
bash scripts/run_local.sh scripts/data/extract_bones_subset.py --format g1 \
  --manifest data/manifests/bones/train_2000.csv --output data/bones-subsets/g1_2000
bash scripts/run_local.sh scripts/data/prepare_lafan.py --robot g1 --source data/bones-subsets/g1_2000 \
  --dest data/npz/g1/bones_2000 --manifest data/manifests/bones/train_2000.csv \
  --input-fps 120 --skip-invalid --include-lafan
bash scripts/run_local.sh scripts/data/filter_bones_seed.py --npz-dir data/npz/g1/bones_2000

# A3/X2: extract SOMA, then retarget (example A3 10k)
bash scripts/run_local.sh scripts/data/extract_bones_subset.py --format soma_uniform \
  --manifest data/manifests/bones/train_10000.csv --output data/bones-subsets/soma_10000
bash scripts/run_local.sh scripts/data/retarget.py --robot a3 \
  --input data/bones-subsets/soma_10000 --output data/bones-retarget-a3-10000
bash scripts/run_local.sh scripts/data/prepare_lafan.py --robot a3 --source data/bones-retarget-a3-10000 \
  --dest data/npz/a3/bones_10000 --manifest data/manifests/bones/train_10000.csv \
  --input-fps 120 --skip-invalid --include-lafan
```

Use `manifest_filtered.csv` to exclude rejected files and retain the 40 validated LAFAN clips. The upstream G1 feasibility filter writes its own manifest; re-add validated LAFAN names to that manifest before packing. The helper below merges accepted entries with the original LAFAN curriculum and writes a final manifest:

```bash
bash scripts/run_local.sh scripts/data/finalize_manifest.py --robot a3 --count 10000
bash scripts/run_local.sh scripts/robot.py a3 scripts/data/build_library.py \
  --npz-dir data/npz/a3/bones_10000 --manifest data/npz/a3/bones_10000/manifest_filtered.csv \
  --out data/library/a3-bones-10000.pt
```

For G1/X2 replace the robot and count with `g1`/`x2_ultra` and `2000`. A directory can contain rejected G1 NPZ files: loading the filtered manifest/packed library is essential. Prepare `heldout_200.csv` the same way **without** `--include-lafan`; evaluate only accepted filenames from that validation manifest with `--clip-list`.

Original/mirrored takes must not cross the split. This run selected original takes, grouped by capture identity, balanced categories, and used clips of 2–60 seconds. Because validation is used for selecting checkpoints, report it as validation, not an untouched final test.

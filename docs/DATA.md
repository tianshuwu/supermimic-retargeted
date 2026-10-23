# Data preparation

All commands run from the repository root. Large raw/processed data are ignored by Git. Manifest files contain names and selection metadata, not motion arrays.

## LAFAN1

Obtain the original BVH files from the Ubisoft source and the 40 G1 retargeted CSVs from ModelScope; links and revisions are in [UPSTREAM.md](../UPSTREAM.md). Unpack them locally. G1 uses the existing G1 CSVs; A3/X2 use the same selected original BVHs, renamed and frame-aligned for SOMA.

```bash
# G1: replace the source directory with your downloaded CSV folder
uv run python scripts/prepare_lafan.py --robot g1 \
  --source data/lafan1-retargeted/g1 --dest data/npz/g1/lafan1 --input-fps 30

# Shared raw BVH adapter: replace --input with your unpacked Ubisoft BVH directory
uv run python scripts/prepare_lafan_bvh.py \
  --input data/lafan1-source --output data/lafan1-named
uv run --project vendor/soma-retargeter python scripts/align_lafan_bone_frames.py \
  --input data/lafan1-named --output data/lafan1-soma-aligned --pelvis-pivot

# Repeat for --robot x2_ultra
uv run python scripts/retarget.py --robot a3 \
  --input data/lafan1-soma-aligned --output data/lafan1-retarget-a3 --batch-size 2
uv run python scripts/prepare_lafan.py --robot a3 \
  --source data/lafan1-retarget-a3 --dest data/npz/a3/lafan1 --input-fps 30
```

The LAFAN adapter first renames joints simultaneously, then uses Left Rx(pi)/Right Ry(pi) frame gauges and an anatomical pelvis-pivot correction. A name-only BVH conversion was insufficient. The alignment script checks non-root world-joint position invariance. `prepare_lafan.py` performs bounded leg/contact correction and records repair magnitudes; this QC is not proof that a motion is dynamically feasible.

## BONES-SEED

Use the dataset provider or the optional anonymous mirror downloader:

```bash
bash scripts/download_bones_gitcode.sh
```

The G1 and SOMA archives together require about 69 GB compressed, plus extraction/processed-data space. The script does not request a Hugging Face token. Mirror availability can change; dataset license terms still apply. `extract_bones_subset.py` verifies archive SHA256 before extracting selected files.

```bash
# Generate nested 2k/10k selections and a capture-group-separated validation set
uv run python scripts/select_bones.py
# Alternatively, use the published experiment selections
mkdir -p data/manifests/bones
cp manifests/bones/* data/manifests/bones/

# G1 subset
uv run python scripts/extract_bones_subset.py --format g1 \
  --manifest data/manifests/bones/train_2000.csv --output data/bones-subsets/g1_2000
uv run python scripts/prepare_lafan.py --robot g1 --source data/bones-subsets/g1_2000 \
  --dest data/npz/g1/bones_2000 --manifest data/manifests/bones/train_2000.csv \
  --input-fps 120 --skip-invalid --include-lafan
uv run python scripts/data/filter_bones_seed.py --npz-dir data/npz/g1/bones_2000

# A3/X2: extract SOMA, then retarget (example A3 10k)
uv run python scripts/extract_bones_subset.py --format soma_uniform \
  --manifest data/manifests/bones/train_10000.csv --output data/bones-subsets/soma_10000
uv run python scripts/retarget.py --robot a3 \
  --input data/bones-subsets/soma_10000 --output data/bones-retarget-a3-10000
uv run python scripts/prepare_lafan.py --robot a3 --source data/bones-retarget-a3-10000 \
  --dest data/npz/a3/bones_10000 --manifest data/manifests/bones/train_10000.csv \
  --input-fps 120 --skip-invalid --include-lafan
```

Use `manifest_filtered.csv` to exclude rejected files and retain the 40 validated LAFAN clips. The upstream G1 feasibility filter writes its own manifest; re-add validated LAFAN names to that manifest before packing. The helper below merges accepted entries with the original LAFAN curriculum and writes a final manifest:

```bash
uv run python scripts/finalize_manifest.py --robot a3 --count 10000
uv run python scripts/robot.py a3 scripts/data/build_library.py \
  --npz-dir data/npz/a3/bones_10000 --manifest data/npz/a3/bones_10000/manifest_filtered.csv \
  --out data/library/a3-bones-10000.pt
```

For G1/X2 replace the robot and count with `g1`/`x2_ultra` and `2000`. A directory can contain rejected G1 NPZ files: loading the filtered manifest/packed library is essential. Prepare `heldout_200.csv` the same way **without** `--include-lafan`; evaluate only accepted filenames from that validation manifest with `--clip-list`.

Original/mirrored takes must not cross the split. This run selected original takes, grouped by capture identity, balanced categories, and used clips of 2–60 seconds. Because validation is used for selecting checkpoints, report it as validation, not an untouched final test.

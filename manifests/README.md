# Published motion selections

These files list **which motions to use**. They contain identifiers and selection
metadata, not BVH/CSV trajectories or NPZ training arrays.

| Files | Contents and use |
|---|---|
| `lafan1.csv` | 40 clip names (`filename` column), used by reference preparation |
| `lafan1.txt` | The same 40 names, usable as evaluation `--clip-list` or training `--env.replay-clip-list` |
| `bones/train_2000.*`, `bones/train_10000.*` | Nested candidate BONES selections, before per-robot feasibility filtering |
| `bones/heldout_200.*` | Capture-group-separated validation candidates; this split participated in checkpoint selection |

BONES CSVs include source archive paths, category, package, capture group,
duration and split. TXT files contain one clip name per line. Dataset names and
selection metadata retain the [dataset-source terms](../UPSTREAM.md).

`scripts/data/select_bones.py` writes new selections into the ignored
`data/manifests/bones/`. Preparation/filtering writes `manifest_filtered.csv`
beside the generated NPZ files. Those per-run accepted lists can differ by robot
and are distinct from these published candidate lists. Training an expanded
dataset must use its filtered manifest or packed library; a candidate list does
not certify successful retargeting or feasibility.

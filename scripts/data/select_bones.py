"""Create nested, deterministic BONES expansion lists with capture-group holdout."""

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data/bones-seed-gitcode/metadata/seed_metadata_v004.csv",
    )
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "upstream_filter", ROOT / "scripts/data/filter_bones_seed.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def digest(text):
        return hashlib.sha256(("supermimic-bones-v1:" + text).encode()).hexdigest()

    buckets = {"train": defaultdict(list), "heldout": defaultdict(list)}
    excluded = Counter()
    with args.metadata.open() as f:
        for row in csv.DictReader(f):
            name = row["filename"]
            frames = int(row["move_duration_frames"])
            duration = (frames - 1) / 120
            # Select originals first. Mirrors are augmentations, never independent evaluation samples.
            if row["is_mirror"].lower() == "true":
                excluded["mirror"] += 1
                continue
            reason = module.name_filter(name)
            if reason:
                excluded["upstream_name_filter"] += 1
                continue
            if not 2 <= duration <= 60:
                excluded["duration_outside_2_60s"] += 1
                continue
            # A shared capture take and its actors cannot straddle train/heldout.
            group = row.get("take_org_name") or row.get("take_name") or name
            split = "heldout" if int(digest(group)[:8], 16) % 10 == 0 else "train"
            row["capture_group"] = group
            row["duration_s"] = duration
            row["split"] = split
            buckets[split][row["category"]].append(row)

    def balanced(categories):
        groups = [
            sorted(rows, key=lambda r: digest(r["filename"]))
            for _, rows in sorted(categories.items())
        ]
        result = []
        for i in range(max(map(len, groups), default=0)):
            result.extend(rows[i] for rows in groups if i < len(rows))
        return result

    out = ROOT / "data/manifests/bones"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "metadata": str(args.metadata),
        "selection_seed": "supermimic-bones-v1",
        "input_fps": 120,
        "excluded": dict(excluded),
        "note": "Planning lists; every robot still requires retargeting and per-clip physical feasibility QC. LAFAN quality precedes expansion. Capture-group heldout is for generalization; LAFAN benchmark is training-set tracking.",
    }
    for split, count in [("train", 2000), ("train", 10000), ("heldout", 200)]:
        rows = balanced(buckets[split])[:count]
        name = f"{split}_{count}"
        fields = [
            "filename",
            "move_g1_path",
            "move_soma_uniform_path",
            "category",
            "package",
            "capture_group",
            "duration_s",
            "split",
        ]
        with (out / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        (out / (name + ".txt")).write_text("".join(r["filename"] + "\n" for r in rows))
        summary[name] = {
            "clips": len(rows),
            "hours": sum(r["duration_s"] for r in rows) / 3600,
            "categories": dict(Counter(r["category"] for r in rows)),
        }
    (out / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

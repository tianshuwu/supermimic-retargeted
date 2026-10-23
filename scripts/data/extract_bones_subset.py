"""Validate a completed Git LFS object, then safely extract only selected motions."""

import argparse
import csv
import fcntl
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["g1", "soma_uniform"], required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".extraction.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    previous = args.output / "extraction.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if (
            old.get("manifest_sha256") == manifest_sha
            and not old.get("missing")
            and old.get("file_sizes")
            and all(
                (args.output / name).is_file() and (args.output / name).stat().st_size == size
                for name, size in old["file_sizes"].items()
            )
        ):
            print("Verified prior subset extraction:", old["extracted"], "files")
            raise SystemExit(0)
    repo = ROOT / "data/bones-seed-gitcode"
    specs = {
        "g1": ("52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a", 23499973647),
        "soma_uniform": (
            "0c1fd91628d2644d0d7c0b735ae64faa5fa79834280bc1ce7860a871e55f912c",
            45188088297,
        ),
    }
    oid, size = specs[args.format]
    archive = repo / (args.format + ".tar.gz")
    if not archive.exists() or archive.stat().st_size != size:
        archive = repo / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid
    if not archive.exists() or archive.stat().st_size != size:
        raise SystemExit("Archive is not fully downloaded; leaving current transfer untouched.")
    cache = ROOT / "logs/data" / ("verified-" + oid + ".json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "path": str(archive),
        "size": size,
        "mtime_ns": archive.stat().st_mtime_ns,
        "sha256": oid,
    }
    if not cache.exists() or json.loads(cache.read_text()) != identity:
        h = hashlib.sha256()
        with archive.open("rb") as f:
            for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != oid:
            raise SystemExit("Archive SHA256 mismatch")
        cache.write_text(json.dumps(identity) + "\n")
    key = "move_g1_path" if args.format == "g1" else "move_soma_uniform_path"
    with args.manifest.open() as f:
        wanted = {PurePosixPath(row[key]).name: row["filename"] for row in csv.DictReader(f)}
    args.output = args.output.resolve()
    args.output.relative_to(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    found = set()
    sizes = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            pure = PurePosixPath(member.name)
            if pure.name not in wanted:
                continue
            if not member.isfile() or pure.is_absolute() or ".." in pure.parts:
                raise ValueError("Unsafe archive member " + member.name)
            target = args.output / pure.name
            if not target.exists() or target.stat().st_size != member.size:
                temporary = target.with_suffix(target.suffix + ".tmp")
                with tar.extractfile(member) as source, temporary.open("wb") as dest:
                    shutil.copyfileobj(source, dest, 1024 * 1024)
                temporary.replace(target)
            found.add(pure.name)
            sizes[pure.name] = member.size
            if len(found) == len(wanted):
                break
    missing = sorted(set(wanted) - found)
    report = {
        "manifest_sha256": manifest_sha,
        "file_sizes": sizes,
        "format": args.format,
        "archive_sha256": oid,
        "manifest": str(args.manifest),
        "extracted": len(found),
        "missing": missing,
        "output": str(args.output),
    }
    (args.output / "extraction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

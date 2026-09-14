"""Fetch pinned CellScientist HDF5 files, verify SHA-256, and arrange runtime paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "release_manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, record):
    return (path.is_file() and path.stat().st_size == record["size_bytes"]
            and sha256_file(path) == record["sha256"])


def download(root, manifest, datasets, fetch=None):
    if fetch is None:
        from huggingface_hub import hf_hub_download
        fetch = hf_hub_download
    root = root.resolve()
    records = []
    for record in manifest["files"]:
        if record["dataset"] not in datasets:
            continue
        target = root / record["runtime_path"]
        if target.exists() and not verified(target, record):
            raise ValueError(f"Existing file has a different checksum: {target}; choose a new --root to keep it")
        if not target.exists():
            downloaded = Path(fetch(
                repo_id=manifest["repo_id"], repo_type="dataset",
                revision=manifest["revision"], filename=record["hub_path"],
                local_dir=str(root / ".hub-download"),
            ))
            if not verified(downloaded, record):
                raise ValueError(f"Downloaded file failed size/SHA-256 verification: {record['hub_path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(downloaded, target)
        records.append({**record, "local_path": str(target)})
        print(f"Verified: {record['runtime_path']}")
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "download_manifest.json"
    existing = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
    previous = existing.get("files", []) if existing.get("revision") == manifest["revision"] else []
    merged = {record["runtime_path"]: record for record in previous + records}
    payload = {"repo_id": manifest["repo_id"], "revision": manifest["revision"], "files": list(merged.values())}
    temporary = receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/hdf5"))
    parser.add_argument("--datasets", nargs="+", choices=("BBBC036", "BBBC047", "CPG0016"), default=["BBBC036", "BBBC047"])
    parser.add_argument("--list", action="store_true", help="Show files, sizes and hashes")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.list:
        selected = [item for item in manifest["files"] if item["dataset"] in args.datasets]
        print(json.dumps({**manifest, "files": selected, "total_bytes": sum(item["size_bytes"] for item in selected)}, indent=2))
        return 0
    download(args.root, manifest, args.datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m cellscientist preflight --config configs/bbbc036_047_formal.json
python -m cellscientist inspect --config configs/bbbc036_047_formal.json
python -m cellscientist freeze \
  --config configs/bbbc036_047_formal.json \
  --lock configs/bbbc036_047_formal.lock.json
python -m cellscientist run-matrix \
  --config configs/bbbc036_047_formal.json \
  --lock configs/bbbc036_047_formal.lock.json \
  --jobs "${JOBS:-1}"

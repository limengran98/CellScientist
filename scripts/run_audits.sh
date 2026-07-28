#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m cellscientist lca-audit --output audit_outputs/lca_audit.json
python -m cellscientist routing-audit \
  --config configs/bbbc036_047_formal.json \
  --output audit_outputs/routing_audit.json

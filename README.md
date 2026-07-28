# CellScientist

This is the clean public implementation accompanying **CellScientist**. It
contains the final discrepancy-conditioned controller, protected task contract,
typed component addresses, deterministic atomic realization, persistent history,
and two method audits. It supports the four BBBC036/BBBC047 task settings:
plate- and SMILES-grouped splits for each dataset.

The repository intentionally excludes historical experiments, result files,
baseline and ablation controllers, external comparison code, credentials, and
private service endpoints.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the locations of your local BBBC HDF5 files and an OpenAI-compatible LLM
endpoint. No credential or endpoint is stored in this repository.

```bash
export CELLSCIENTIST_DATA_ROOT=/path/to/bbbc_hdf5
export CELLSCIENTIST_API_BASE=https://your-openai-compatible-endpoint/v1
export CELLSCIENTIST_API_KEY=your_key
```

The required data layout and HDF5 schema are documented in
[`data/README.md`](data/README.md).

## Reproduce the BBBC exploration

The release uses folds 1--3 for fitting, deterministically partitions fold 4
into group-disjoint feedback and selection subsets, and keeps fold 5 untouched
for final reporting. The configuration registers the five exploration seeds
before execution. A lock records the code and data hashes used in the run.

```bash
./scripts/run_exploration.sh
```

For a single task/seed, after creating the lock:

```bash
python -m cellscientist run \
  --config configs/bbbc036_047_formal.json \
  --lock configs/bbbc036_047_formal.lock.json \
  --task BBBC036_smiles \
  --seed 11
```

On a CUDA machine, set `JOBS` to the number of task/seed processes appropriate
for available GPU memory. The released configuration is CUDA fail-closed: it
will not silently replace the registered backend with CPU execution.

## Run method audits

```bash
./scripts/run_audits.sh
```

`lca-audit` evaluates registered protected-field, interface, output, runtime,
and repair-budget checks. `routing-audit` evaluates typed discrepancy routing
and local repair on 15 registered held-out component faults. Supplying the LLM
environment variables exercises the constrained LLM route; otherwise the audit
uses its registered deterministic top-ranked route and records that mode.

## Scope

The public code is a method release, not a benchmark bundle. It does not
redistribute BBBC data, trained artifacts, numerical result files, private
endpoints, or any comparator implementation. The paper and final rebuttal
snapshot are preserved one directory above this source tree.

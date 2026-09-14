# Reproducing and comparing runs

## Start from a fixed source and dataset version

Record the repository commit with `git rev-parse HEAD`. Install the project
in an isolated environment and capture the installed package versions:

```bash
python -m pip install -e ".[data]"
python -m pip freeze > environment.txt
python scripts/download_data.py --root data/hdf5
```

The downloader uses the revision, file sizes and SHA-256 hashes in
[`data/release_manifest.json`](../data/release_manifest.json). It arranges the
configured runtime paths and writes `download_manifest.json` inside the data
root. The manifest covers BBBC036, BBBC047 and CPG0016; the default download
selects the four BBBC inputs (974,777,218 bytes, about 0.97 GB).

## Configure the experiment

The BBBC base protocol specifies four task/split settings, seeds
`11, 22, 33, 44, 55`, ten evaluated candidates, and budget checkpoints
`1, 3, 5, 10`. The default predictor backend is CUDA PyTorch and the LLM model
ID is `gemini-3-pro-preview`. See [LLM backends](llm_backends.md) for the paper's
Qwen reproduction backbone and custom APIs.

For a new backbone or compute backend, use `scripts/configure_run.py` with a
new experiment name. Configure the served model ID, provider/deployment,
decoding settings and endpoint before freezing. Keep the configuration, lock,
environment capture and serving details with the run records.

## Inspect, freeze and execute

Follow the commands in the [README](../README.md#exploration):

1. `preflight --check-llm` checks the selected compute backend, credential
   settings and an allow-listed health response. An unsuccessful check exits
   with status 1 and reports actionable errors.
2. `inspect` checks the HDF5 schema, nonempty partitions and group separation.
3. `freeze` hashes the resolved configuration, input files and Python package
   source files into a protocol lock.
4. `run` or `run-matrix` verifies that lock before evaluating candidates.

The lock stores resolved data paths, so create a lock on each machine after
placing the data. Match file hashes, source commit and experiment settings
when comparing machines. Dependency versions, Python/platform information
and CUDA details are recorded in `run_manifest.json`; retain `environment.txt`
to reconstruct the installed environment. API deployment versions and
hardware can affect trajectories even with the same requested decoding values.

## What is fitted, selected and reported

| Partition | Use |
| :--- | :--- |
| Folds 1–3 | Train candidate predictors and fit preprocessing/projections |
| Fold 4 feedback groups | Compute revision diagnostics |
| Fold 4 selection groups | Score candidate retention alongside feedback |
| Fold 5 | Compute held-out metrics for the retained candidates |

The partition seed is `20260724`. Fold 4 is split deterministically by the
task's plate or SMILES group, and protected partitions are checked for group
overlap. Retention maximizes the mean feedback and selection global PCC.
The evaluator also records response metrics; the configured primary metric
determines selection. Checkpoint records identify the retained candidate at
each budget, which makes comparisons at equal candidate counts explicit.

## Read the artifacts

| Artifact | Contents |
| :--- | :--- |
| Protocol lock | Resolved configuration, file hashes, source hashes and lock hash |
| `run_manifest.json` | Task, seed, protocol/source hashes and runtime package versions |
| `run_result.json` | Selected candidate, held-out metrics, trajectory, HRT and checkpoints |
| `llm_trace.jsonl` | Request/response records, retries, usage and decision-validation events |
| `run_failure.json` | Run provenance and execution error for a failed trajectory |

Formal execution reuses an existing completed result when its frozen run key
matches. Use a new experiment name and output root for an independent repeat.
Compare task-matched seeds and the same checkpoint budget, and examine fallback
and retry records when interpreting a backbone comparison.

## Run the regression suite

```bash
python -m unittest discover -s tests -v
```

The suite uses generated HDF5 fixtures and a local mock Chat Completions
server. It exercises optional authentication, request variants, malformed
responses, partition leakage checks, lock invalidation and a complete frozen
CPU trajectory. These are software checks; biological benchmark results come
from running the configured data/model protocol.

The two component audits are also available as standalone commands:

```bash
python -m cellscientist lca-audit --output audit_outputs/lca_audit.json
python -m cellscientist routing-audit --output audit_outputs/routing_audit.json
```

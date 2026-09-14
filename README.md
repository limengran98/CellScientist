<a id="top"></a>

<div align="center">

<h1>CellScientist</h1>
<h3>Model Revision by Diagnostic Routing for Morphological Perturbation Prediction</h3>

<p><b>Turn execution feedback into traceable model improvements.</b><br>
A protocol-constrained workflow that links design choices, local revisions and validation outcomes.</p>

<p>
  <a href="pyproject.toml"><img alt="Python 3.10 or later" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT code license" src="https://img.shields.io/badge/code-MIT-4B8B72"></a>
  <a href="https://huggingface.co/datasets/Boom5426/CellScientist"><img alt="Data on Hugging Face" src="https://img.shields.io/badge/data-Hugging%20Face-E0A16B?logo=huggingface&logoColor=black"></a>
</p>

<p>
  <a href="https://cellscientist-research.phuonganh49123.chatgpt.site"><b>Project page</b></a> ·
  <a href="https://cellscientist-research.phuonganh49123.chatgpt.site/assets/CellScientist.pdf"><b>Paper</b></a> ·
  <a href="#quick-start"><b>Quick start</b></a> ·
  <a href="#data"><b>Data</b></a> ·
  <a href="#exploration"><b>Run exploration</b></a> ·
  <a href="#extension"><b>Extend</b></a> ·
  <a href="#method"><b>Method</b></a> ·
  <a href="#outputs"><b>Inspect outputs</b></a> ·
  <a href="#citation"><b>Cite</b></a>
</p>

</div>

**CellScientist** uses execution and validation feedback to guide local revisions
of cellular perturbation-response models. A fixed task contract protects data
partitions and evaluation semantics, while a persistent history records which
components changed, why they changed and how each candidate performed.

The **BBBC036/BBBC047 protocol** runs CellScientist across plate and SMILES
splits with a shared candidate language, paired seeds and fixed evaluation
budgets. Configuration locks, component audits and recorded trajectories make
each run inspectable and provide a starting point for new models and datasets.

<p align="center">
  <a href="docs/assets/refinement-trajectory.pdf"><img src="docs/assets/refinement-trajectory.png" alt="BBBC047 manuscript case study: successive FiLM and reference-conditioned model revisions include both improvements and regressions; Ref-RGAF is retained with the highest reported validation PCC in the illustrated trajectory." width="920"></a>
  <br>
  <sub>From the manuscript: the BBBC047 model-revision trajectory · <a href="docs/assets/refinement-trajectory.pdf">Vector figure ↗</a></sub>
</p>

The manuscript case study follows successive architecture revisions and retains
the strongest validation result. The [controlled BBBC workflow](#method) makes
revision decisions comparable through typed component addresses and a shared
candidate language.

<a id="quick-start"></a>

## 🚀 Quick start

**Python 3.10+.** Clone the repository and install it in an isolated environment:

```bash
git clone https://github.com/limengran98/CellScientist.git
cd CellScientist
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data]"
python -m cellscientist --help
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. Run the commands below from the repository root.

### Start with the self-contained audits

These self-contained checks run synthetic contract and component-fault cases
on the CPU:

```bash
python -m cellscientist lca-audit --output audit_outputs/lca_audit.json
python -m cellscientist routing-audit --output audit_outputs/routing_audit.json
```

| Audit | What it checks | Output |
| :--- | :--- | :--- |
| **LCA** | Protected fields, interfaces, outputs, runtime failures and repair budgets | Case-level decisions and summary counts |
| **Routing** | Localization and local repair of 15 registered faults across five component addresses | Per-case routing, repair outcomes and the routing mode used |

The default routing audit evaluates the deterministic top-ranked route.
Pass `--config configs/bbbc036_047_formal.json` to evaluate LLM routing using
your configured endpoint. Both modes record the routing mode in the output.

<a id="data"></a>

## 🧬 Data

**[Download the CellScientist data release on Hugging Face →](https://huggingface.co/datasets/Boom5426/CellScientist)**

The release contains six preprocessed HDF5 files under
[`Cell_Morphology_data/`](https://huggingface.co/datasets/Boom5426/CellScientist/tree/main/Cell_Morphology_data).

| Dataset | Plate split | SMILES split |
| :--- | :--- | :--- |
| **BBBC036** | `BBBC036_plate_split.h5` | `BBBC036_smiles_split.h5` |
| **BBBC047** | `BBBC047_plate_split.h5` | `BBBC047_smiles_split.h5` |
| **CPG0016** | `cpg0016_plate_split.h5` | `cpg0016_smiles_split.h5` |

Download the BBBC protocol inputs and arrange their runtime paths automatically:

```bash
python scripts/download_data.py --root data/hdf5
```

The script uses the pinned Hugging Face revision and SHA-256 hashes in
[`data/release_manifest.json`](data/release_manifest.json). Add `--list` to
inspect the download plan; use `--datasets CPG0016` to select CPG0016.
Verified files are reused on subsequent runs. The BBBC layout is:

```text
data/hdf5/
├── BBBC036/
│   ├── BBBC036_plate_split.h5
│   └── BBBC036_smiles_split.h5
└── BBBC047/
    ├── BBBC047_plate_split.h5
    └── BBBC047_smiles_split.h5
```

Each runtime file uses a `combined` HDF5 group with `morphology_pre`,
`morphology_post`, `dose`, `smiles`, `plate_id` and `split_id` datasets.
`split_id` assigns folds 1–5. The [`inspect` command](#exploration) validates
the schema and reports partition sizes and group separation.
See [the input specification](data/README.md) for details.

<a id="exploration"></a>

## Reproduce the BBBC workflow

The paper uses **Gemini 3 Pro** as its default LLM backbone and
**Qwen2.5-0.5B-Instruct** for the open-weight reproduction of the controlled audit.
The [BBBC configuration](configs/bbbc036_047_formal.json) selects
`gemini-3-pro-preview` and the `torch_cuda` predictor backend.

**Other LLM APIs are supported through the OpenAI-compatible Chat Completions
interface.** Set the endpoint and its served model ID to use a hosted provider,
a gateway or a local model server. The [LLM guide](docs/llm_backends.md) gives
Qwen and custom-model configurations, optional local authentication and request
parameter settings. Predictor fitting supports `torch_cuda`, `torch_cpu` and
`sklearn` backends.

### 1. Set your data and endpoint

```bash
export CELLSCIENTIST_DATA_ROOT="$PWD/data/hdf5"
export CELLSCIENTIST_API_BASE=https://your-openai-compatible-endpoint/v1
export CELLSCIENTIST_API_KEY=your_key
```

<details>
<summary><b>Windows PowerShell equivalents</b></summary>

```powershell
$env:CELLSCIENTIST_DATA_ROOT = (Resolve-Path "data/hdf5").Path
$env:CELLSCIENTIST_API_BASE = "https://your-openai-compatible-endpoint/v1"
$env:CELLSCIENTIST_API_KEY = "your_key"
```

</details>

Endpoint credentials are read from the configured environment variable.

### 2. Inspect the inputs and freeze the run

```bash
python -m cellscientist preflight --config configs/bbbc036_047_formal.json --check-llm
python -m cellscientist inspect --config configs/bbbc036_047_formal.json
python -m cellscientist freeze --config configs/bbbc036_047_formal.json --lock configs/bbbc036_047_formal.lock.json
```

`preflight` checks the selected compute backend and LLM settings;
`--check-llm` sends a small request to validate the endpoint's response.
`inspect` checks the HDF5 inputs and protected partitions. `freeze` records
the resolved configuration, source hashes and data-file hashes; formal runs
verify the lock before execution.

### 3. Run one task and seed

```bash
python -m cellscientist run --config configs/bbbc036_047_formal.json --lock configs/bbbc036_047_formal.lock.json --task BBBC036_smiles --seed 11
```

Available task IDs are `BBBC036_plate`, `BBBC036_smiles`, `BBBC047_plate` and
`BBBC047_smiles`. The configuration registers seeds `11, 22, 33, 44, 55` and
a budget of 10 evaluated candidates, with reporting checkpoints at
`1, 3, 5, 10`.

<details>
<summary><b>Run the full registered matrix</b></summary>

After freezing the configuration, run all four tasks and five seeds:

```bash
python -m cellscientist run-matrix --config configs/bbbc036_047_formal.json --lock configs/bbbc036_047_formal.lock.json --jobs 1
```

Alternatively, the Bash wrapper performs preflight, inspection, freezing and
the matrix run in sequence:

```bash
bash scripts/run_exploration.sh
```

Choose `--jobs` according to available GPU memory. The wrapper reads the
equivalent setting from `JOBS`, which defaults to `1`.

</details>

### Evaluation protocol

| Data partition | Role |
| :--- | :--- |
| **Folds 1–3** | Fit candidate predictors |
| **Fold 4: feedback subset** | Produce diagnostics that guide the next revision |
| **Fold 4: selection subset** | Score candidates for retention alongside feedback scores |
| **Fold 5** | Report the selected candidates after search |

Fold 4 is deterministically divided into group-disjoint feedback and selection
subsets using the task's plate or SMILES groups. The registered retention rule
maximizes the mean feedback and selection **global PCC**. Fold 5 provides
held-out reporting after the search and retention decisions.

<a id="method"></a>

## How CellScientist revises a model

```text
Fixed task contract + recorded history
                   ↓
      Execution and validation diagnostics
                   ↓
       Route to a typed component address
                   ↓
      Apply and check a legal local revision
                   ↓
          Evaluate → record → retain
                   └───────────────↺
```

| Component | Role in the workflow | Implementation |
| :--- | :--- | :--- |
| **HRT — History-Aware Revision Tracking** | Preserve candidate states, component dependencies and transition provenance | [`hrt.py`](cellscientist/hrt.py), [`runner.py`](cellscientist/runner.py) |
| **LCA — Limited-Change Application** | Check protected task semantics and admissible local changes | [`schemas.py`](cellscientist/schemas.py), [`search_space.py`](cellscientist/search_space.py), [`lca_audit.py`](cellscientist/lca_audit.py) |
| **PDR — Performance-Discrepancy Refinement** | Use diagnostics and history to choose the component to revise | [`controllers.py`](cellscientist/controllers.py), [`routing_audit.py`](cellscientist/routing_audit.py) |

The public controller addresses **conditioning, target, representation,
decoder and regularization** within a 768-candidate space. Candidates specify
input conditioning, direct or delta targets, input and output projection
dimensions, and regularization strength. The LLM selects from permitted
address–candidate pairs; deterministic code realizes the proposed local change.

<a id="extension"></a>

## Extend to another model or dataset

Create a named run configuration while preserving the base protocol:

```bash
python scripts/configure_run.py --name my_llm --model your-served-model-id --output configs/my_llm.json
```

Set `CELLSCIENTIST_API_BASE` and `CELLSCIENTIST_API_KEY` for that endpoint, then
use `configs/my_llm.json` in the same `preflight → inspect → freeze → run`
sequence. The helper creates separate result and cache paths for the new run.
Add `--local-api` for an API that accepts requests without a key, or
`--backend torch_cpu` / `--backend sklearn` for CPU fitting.

| Extension | Starting point |
| :--- | :--- |
| Change the LLM or API endpoint | [LLM backends and examples](docs/llm_backends.md) |
| Add a dataset and define its split groups | [Input schema](data/README.md) and [extension guide](docs/extending.md) |
| Add a candidate option or component address | [Candidate and routing contracts](docs/extending.md#candidate-language-and-routing) |
| Reproduce and compare runs | [Run records and verification](docs/reproducibility.md) |

<a id="outputs"></a>

## Inspect the trajectory and selected candidate

Runs are saved under the configured output root:

```text
outputs/exploration/<task>/standard_h0/cellscientist/seed-<seed>/
├── run_manifest.json   # Protocol, source and runtime provenance
├── run_result.json     # Selection, metrics, checkpoints, HRT and trajectory
└── llm_trace.jsonl     # Recorded LLM requests and responses, when present
```

Failed runs write `run_failure.json`. Fitted-candidate caching uses
`outputs/cache/` by default. To inspect the example run:

```python
import json
from pathlib import Path

path = Path(
    "outputs/exploration/BBBC036_smiles/standard_h0/"
    "cellscientist/seed-11/run_result.json"
)
result = json.loads(path.read_text(encoding="utf-8"))
print("Selected candidate:", result["selected_candidate_id"])
print("Held-out metrics:", result["test_metrics"])
print("Evaluated candidates:", result["evaluated_candidates"])
```

`trajectory` records successive candidates and feedback;
`hrt` records the protected task state and component history;
`budget_checkpoints` reports retained candidates at the registered budgets.
Keep these records and the lock together when comparing runs. The
[reproducibility guide](docs/reproducibility.md) explains provenance,
checkpoint selection, environment capture and the regression suite:

```bash
python -m unittest discover -s tests -v
```

<details>
<summary><b>Repository map</b></summary>

```text
cellscientist/   Controller, candidate space, evaluator, provenance and audits
configs/        BBBC036/BBBC047 protocol and named run configurations
data/           HDF5 schema and pinned download manifest
scripts/        Data download, configuration, exploration and audit tools
tests/          API, data, lock and end-to-end regression checks
docs/           Reproduction, LLM and extension guides
docs/assets/    Manuscript trajectory figure for this README
project-page/   Public research page, interactive charts, figures and preprint
```

Useful entry points: [CLI](cellscientist/cli.py) ·
[data loader](cellscientist/data.py) ·
[protocol and locks](cellscientist/protocol.py) ·
[predictors](cellscientist/models.py) ·
[evaluation](cellscientist/evaluator.py).

</details>

<a id="citation"></a>

## Citation, license and support

CellScientist accompanies **CellScientist: Model Revision by Diagnostic Routing for Morphological Perturbation Prediction**.
Read the [paper](https://cellscientist-research.phuonganh49123.chatgpt.site/assets/CellScientist.pdf), explore the
[project page](https://cellscientist-research.phuonganh49123.chatgpt.site), or preview its [source](project-page/README.md).

```bibtex
@misc{li2026cellscientist,
  title = {CellScientist: Model Revision by Diagnostic Routing
           for Morphological Perturbation Prediction},
  author = {Li, Mengran and Li, Bo and Wang, Jiaying and
            Xing, Wenbin and Zhang, Chengyang and Wu, Jinlin and
            Lei, Zhen and Luo, Jiebo and Li, Stan Z. and Zang, Zelin},
  year = {2026},
  howpublished = {Preprint},
  url = {https://github.com/limengran98/CellScientist}
}
```

For software use, cite this repository and record the exact commit and run
configuration. The following entry identifies the current software release:

```bibtex
@misc{cellscientist_software,
  author       = {{CellScientist Authors}},
  title        = {{CellScientist}},
  year         = {2026},
  howpublished = {\url{https://github.com/limengran98/CellScientist}},
  note         = {Software, version 1.0.0}
}
```

Cite the original datasets used in your analysis as well. Code is released
under the [MIT License](LICENSE). The companion
[Hugging Face data card](https://huggingface.co/datasets/Boom5426/CellScientist)
declares Apache-2.0 for that release; consult the source datasets' terms for
the underlying measurements.

For questions or reproducible bug reports,
[open an issue](https://github.com/limengran98/CellScientist/issues) with the
commit, task ID, configuration and relevant error trace.

---

<p align="center">
  <b>A selected model, with an inspectable path to it.</b><br>
  <sub><a href="#top">Back to top ↑</a></sub>
</p>

# CellScientist

CellScientist is an autonomous AI agent framework designed for Virtual Cell Modeling (VCM). It employs a Dual-Space Bilevel Optimization strategy to align symbolic scientific hypotheses with computational code implementation.

The system operates through a structured Task Hypergraph, performing evolutionary optimization to discover robust biological models.

## 🛠️ Installation

```bash
conda create --name CellScientist python=3.11.14
conda activate CellScientist
cd CellScientist
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f [https://download.pytorch.org/whl/cu118/torch_stable.html](https://download.pytorch.org/whl/cu118/torch_stable.html)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f [https://data.pyg.org/whl/torch-2.0.1+cu118.html](https://data.pyg.org/whl/torch-2.0.1+cu118.html)
pip install -r requirements.txt

```

## 📂 Project Structure

```
CellScientist/
├── Design_Analysis/          # [Phase 1] Exploration & Hypergraph Initialization
│   ├── cellscientist_phase_1.py
│   └── design_analysis_config.json
│
├── Generate_Execution/       # [Phase 2] Top-Down Instantiation & Code Generation
│   ├── cellscientist_phase_2.py
│   └── generate_execution_config.json
│
├── Review_Feedback/          # [Phase 3] Bottom-Up Refinement
│   ├── cellscientist_phase_3.py
│   ├── review_feedback_config.json
│   └── CodeEvo/              # Evolution History & Artifacts
│
├── pipeline_config.json      # ⭐ Unified pipeline-level configuration (recommended)
├── run_cellscientist.py      # 🚀 Unified pipeline runner
├── requirements.txt
└── README.md

```

* **Design_Analysis/** – handles design and analytical logic
* **Generate_Execution/** – manages generation and execution processes
* **Review_Feedback/** – reviews and iteratives optimization process
* **llm_providers.json** – defines available LLM configurations
* **requirements.txt** – Python dependencies
* **run_cellscientist.py** – The master script that validates configurations and executes Phase 1, 2, and 3 sequentially in isolated environments.


## Data

### Cell Morphology

- **[cpg0016 & cpg0003]** (Cell Painting Gallery)  
  - **Registry**: [AWS Open Data Registry – Cell Painting Gallery](https://registry.opendata.aws/cellpainting-gallery/)
  - **cpg0016**: Contains compound perturbation profiles from the JUMP-CP project.   🔗 [Browse on S3](https://cellpainting-gallery.s3.amazonaws.com/index.html#cpg0016-jump-assembled/source_all/workspace/profiles_assembled/COMPOUND/v1.0/)
  - **cpg0003**: Includes the Rosetta dataset with two key subsets:  🔗 [Browse on S3](https://cellpainting-gallery.s3.amazonaws.com/index.html#cpg0003-rosetta/broad/workspace/preprocessed_data/)
    - `CDRP-BBBC047-Bray`  
    - `CDRPBIO-BBBC036-Bray`  

> 💡 The preprocessed Cell Morphology dataset used in this paper is also publicly available on Hugging Face:  
> 📦 [**CellScientist Dataset** @ HuggingFace](https://huggingface.co/datasets/Boom5426/CellScientist)





## ⚙️ Experiment Settings & Environment

### Hardware & Software Infrastructure

Experiments are conducted on high-performance nodes tailored for graph learning.

* **CPU:** Dual Intel Xeon Platinum 8336C @ 2.30GHz
* **GPU:** NVIDIA RTX 5880 Ada Generation (48GB VRAM)
* **Memory:** 512 GB DDR4 ECC
* **Software:** Python 3.11.14, PyTorch 2.0.1+cu118, PyG 2.3.0, CUDA 11.8

### Hyperparameters (Key Configurations)

The Dual-Space Bilevel Optimization is controlled via hierarchical configs:

* **LLM Engine:** Gemini 3 Pro (Temp: 0.5 - 0.7)
* **Design Phase:** 4 parallel hypothesis branches; Max 3 self-correction fix rounds.
* **Execution Phase:** Global timeout 100h; Step timeout 5h; Max 5 debugging rounds.
* **Review Phase:** Max 10 optimization iterations; Optimized via Pearson Correlation Coefficient (PCC).

### Cost Efficiency

CellScientist minimizes cost through a **Contextual Memory** mechanism that reduces token load by ~60% in later iterations.

* **Average Run (3-5 iterations):** $1.00 - $2.00 USD
* **Complex Run (10 iterations):** < $5.00 USD

## 🚀 Usage

### Method I: The Unified Pipeline

```bash
python run_cellscientist.py

```

* Reads pipeline_config.json (if present)
* Automatically merges shared parameters into each phase config
* Ensures consistent dataset, paths, GPU, and LLM settings across all phases

### Method II: Manual Phase Execution

You can also run individual phases manually if you need to debug a specific step.

**Phase 1: Design & Analysis**

```bash
cd Design_Analysis
python cellscientist_phase_1.py design_analysis_config.json

```

**Phase 2: Generation & Execution**

```bash
cd Generate_Execution
python cellscientist_phase_2.py --config generate_execution_config.json run --use-idea

```

**Phase 3: Review & Optimization**

```bash
cd Review_Feedback
python cellscientist_phase_3.py --config review_feedback_config.json

```

## 📊 Outputs

All experiment artifacts are automatically organized in `../results/<dataset_name>/`:

* **`pipeline_summary.json`**: The global scoreboard containing success rates, budget usage, and performance metrics across all phases.
* **`logs/<timestamp>/`**:
* Contains detailed execution logs (`phase1.log`, `phase2.log`, `phase3.log`) for full traceability.
* **`advanced_metrics/`**: Analysis of mechanism diversity and code complexity.


* **`design_analysis/`**: Generated scientific hypotheses and initial data artifacts.
* **`generate_execution/`**: Source code instantiation, execution logs, and intermediate notebooks.
* **`review_feedback/`**: Final optimized model, evolutionary history artifacts, and optimization process logs.
* Optimization history (`optimization_history.md`, `optimization_tree.txt`).
* **`notebook_best.ipynb`**: The final, best-performing model code validated by the system.
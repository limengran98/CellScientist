# CellScientist

A lightweight toolkit for cell-related analysis and intelligent workflow generation using large language models (LLMs).


## 🛠️ Installation

```bash
conda create --name CellScientist python=3.11.14
conda activate CellScientist
git clone https://github.com/limengran98/CellScientist.git
cd CellScientist
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
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
├── run_cellscientist.py      # 🚀 Unified Pipeline Orchestrator
├── llm_providers.json        # LLM API Configurations
├── requirements.txt          # Python Dependencies
└── README.md
```

* **Design_Analysis/** – handles design and analytical logic
* **Generate_Execution/** – manages generation and execution processes
* **Review_Feedback/** – reviews and iteratives optimization process
* **llm_providers.json** – defines available LLM configurations
* **requirements.txt** – Python dependencies
* **run_cellscientist.py** – The master script that validates configurations and executes Phase 1, 2, and 3 sequentially in isolated environments.


## 🚀 Usage

### Method I: The Unified Pipeline
```bash
python run_cellscientist.py
```

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
python cellscientist_phase_2.py --config generate_execution_config.json run

```

**Phase 3: Review & Optimization**

```bash
cd Review_Feedback
python cellscientist_phase_3.py --config review_feedback_config.json

```


## 📊 Outputs

Results are stored in the `../results/<dataset_name>/` directory (relative to the project root):

* **`design_analysis/`**: Hypotheses, initial H5 data artifacts.
* **`generate_execution/`**: Generated Notebooks (`.ipynb`) and execution logs.
* **`review_feedback/`**:
* `notebook_best.ipynb`: The final optimized model.
* `optimization_history.md`: Detailed log of the evolutionary process.
* `optimization_tree.txt`: Visual ASCII tree of the decision process 
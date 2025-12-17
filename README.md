# CellScientist

A lightweight toolkit for cell-related analysis and intelligent workflow generation using large language models (LLMs).


## Installation

```bash
conda create --name CellScientist python=3.11.14
conda activate CellScientist
git clone https://github.com/limengran98/CellScientist.git
cd CellScientist
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install -r requirements.txt
```

## Project Structure

```
CellScientist/
├── Design_Analysis/
├── Generate_Execution/
├── Review_Feedback/
    ├── CodeEvo/
├── llm_providers.json
├── requirements.txt
└── README.md
```

* **Design_Analysis/** – handles design and analytical logic
* **Generate_Execution/** – manages generation and execution processes
* **Review_Feedback/** – reviews and iteratives optimization process
* **llm_providers.json** – defines available LLM configurations
* **requirements.txt** – Python dependencies


## Quick Start


```bash
python run_cellscientist.py
```

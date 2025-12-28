# CodeEvo Agent 🚀

### 1. Run Immediately
Execute in terminal:
```bash
python main_agent.py --config config.json --prompt prompts/code_optimizer.yaml

python main_agent.py --config config-mol.json --prompt prompts/code_optimizer-mol.yaml
````


### 2. Configuration

Open `config.json` and update just two fields:

1.  **Target Path** (`target_project` -\> `root_dir`):
    Enter the absolute path of the codebase you want to optimize.
2.  **API Key** (`providers` -\> `yizhan` -\> `api_key`):
    Enter your LLM API Key.

### 3. Outputs

After execution, check the `workspace/` folder in the current directory.

  - **New Project**: `workspace/ProjectName_Timestamp/` (This is the complete, modified, and runnable project copy).
  - **Logic & Ideas**: `workspace/.../agent_idea.json`.

#!/usr/bin/env python3
# run_cellscientist.py
import os
import sys
import json
import subprocess
import time
from typing import Dict, Any, List

# =============================================================================
# ⚙️ Path Configuration Mapping (Based on your directory tree)
# =============================================================================

# Define folder name, script name, and config filename for each phase
# Assumes config files (*_config.json) are located inside their respective subfolders
PHASE_MAP = {
    "Phase 1": {
        "folder": "Design_Analysis",
        "script": "cellscientist_phase_1.py",
        "config": "design_analysis_config.json",
        "cmd_args": []  # Phase 1 only needs the config filename as an argument
    },
    "Phase 2": {
        "folder": "Generate_Execution",
        "script": "cellscientist_phase_2.py",
        "config": "generate_execution_config.json",
        "cmd_args": ["run"] # Phase 2 requires the 'run' subcommand
    },
    "Phase 3": {
        "folder": "Review_Feedback",
        "script": "cellscientist_phase_3.py",
        "config": "review_feedback_config.json",
        "cmd_args": [] # Phase 3 runs by default
    }
}

# =============================================================================
# Import Utility Libraries (Rich for styling)
# =============================================================================
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
except ImportError:
    console = None

# =============================================================================
# Helper Functions
# =============================================================================

def get_config_path(phase_info: Dict) -> str:
    """Construct full path: ./Folder/config.json"""
    return os.path.join(phase_info["folder"], phase_info["config"])

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"❌ Error: Config file not found: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_nested(data: Dict, keys: List[str], default="N/A"):
    """Safely retrieve nested dictionary fields"""
    val = data
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val

def validate_configs() -> str:
    """Pre-load all configs to ensure 'dataset_name' consistency"""
    dataset_names = {}
    
    # Iterate through all phases to load Config
    for name, info in PHASE_MAP.items():
        path = get_config_path(info)
        cfg = load_json(path)
        ds = cfg.get("dataset_name", "MISSING")
        dataset_names[name] = ds
        
        # Store loaded Config back into the dictionary for later printing
        info["_loaded_cfg"] = cfg

    # Check consistency
    unique = set(dataset_names.values())
    if len(unique) > 1:
        if console:
            console.print(Panel("[bold red]CRITICAL ERROR: 'dataset_name' mismatch![/]", border_style="red"))
            for k, v in dataset_names.items():
                console.print(f"  - {k}: [red]{v}[/]")
        else:
            print("CRITICAL ERROR: 'dataset_name' mismatch!")
            print(dataset_names)
        sys.exit(1)
    
    return list(unique)[0]

def print_execution_plan(dataset_name: str):
    """Print a stylized execution plan table"""
    if not console:
        print(f"Plan for dataset: {dataset_name}")
        return

    table = Table(title=f"🧬 CellScientist Pipeline (Target: [bold green]{dataset_name}[/])")
    table.add_column("Phase", style="cyan", no_wrap=True)
    table.add_column("Directory", style="blue")
    table.add_column("Model", style="magenta")
    table.add_column("Key Params", style="white")

    # Phase 1 Info
    p1 = PHASE_MAP["Phase 1"]
    c1 = p1["_loaded_cfg"]
    p1_model = get_nested(c1, ["phases", "task_analysis", "llm_notebook", "llm", "model"])
    p1_runs = get_nested(c1, ["phases", "task_analysis", "llm_notebook", "multi", "num_runs"])
    table.add_row("1. Design", p1["folder"], str(p1_model), f"Runs: {p1_runs}")

    # Phase 2 Info
    p2 = PHASE_MAP["Phase 2"]
    c2 = p2["_loaded_cfg"]
    p2_model = get_nested(c2, ["llm", "model"])
    p2_iters = get_nested(c2, ["experiment", "max_iterations"])
    table.add_row("2. Generate", p2["folder"], str(p2_model), f"Max Iters: {p2_iters}")

    # Phase 3 Info
    p3 = PHASE_MAP["Phase 3"]
    c3 = p3["_loaded_cfg"]
    p3_model = get_nested(c3, ["llm", "model"])
    p3_metric = get_nested(c3, ["review", "target_metric"])
    table.add_row("3. Review", p3["folder"], str(p3_model), f"Target: {p3_metric}")

    console.print(table)
    console.print("")

# =============================================================================
# Main Logic
# =============================================================================

def main():
    # 1. Environment Check
    if not os.path.exists("Design_Analysis"):
        print("❌ Error: Run this script from the 'CellScientist' root directory.")
        sys.exit(1)

    # 2. Validate Config Consistency
    ds_name = validate_configs()

    # 3. Display Plan
    print_execution_plan(ds_name)

    # 4. Countdown Confirmation
    print(f"🚀 Pipeline starting for [{ds_name}] in 3 seconds...")
    time.sleep(3)

    # 5. Sequential Execution
    for name, info in PHASE_MAP.items():
        folder = info["folder"]
        script = info["script"]
        config = info["config"]
        extra_args = info["cmd_args"]

        # Construct command
        # Note: Since we cd into the subdirectory, use the filename directly for config, no folder prefix needed
        cmd = ["python", script, "--config", config] if name != "Phase 1" else ["python", script, config]
        
        # Phase 2 Special Handling: Add extra 'run' argument
        if extra_args:
            cmd.extend(extra_args)

        if console:
            console.rule(f"[bold blue]Running {name}[/]")
            console.print(f"📂 Context: [underline]{folder}[/]")
            console.print(f"💻 Command: [dim]{' '.join(cmd)}[/]\n")
        else:
            print(f"\n=== Running {name} (Dir: {folder}) ===")

        start_ts = time.time()
        
        try:
            # [CRITICAL] cwd=folder: Switch to the corresponding subfolder to run the script
            # This ensures internal relative paths (e.g., from prompts import...) work correctly
            subprocess.run(cmd, cwd=folder, check=True)
            
        except subprocess.CalledProcessError as e:
            if console:
                console.print(Panel(f"[bold red]❌ Pipeline Failed at {name}[/]\nExit Code: {e.returncode}", border_style="red"))
            else:
                print(f"❌ Pipeline Failed at {name} with exit code {e.returncode}")
            sys.exit(e.returncode)
        
        duration = time.time() - start_ts
        print(f"✅ {name} Completed ({duration:.1f}s)\n")

    if console:
        console.print(Panel("[bold green]🏆 CellScientist Workflow Completed Successfully![/]", border_style="green"))
    else:
        print("\n🏆 CellScientist Workflow Completed Successfully!")

if __name__ == "__main__":
    main()
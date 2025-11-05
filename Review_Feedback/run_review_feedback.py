#!/usr/bin/env python3
# run_review_feedback.py
# Main entry point for the "Review_Feedback" hyperedge.
# This phase reads the manifest from "Design_Analysis",
# runs an expert review, and exports the best artifact.

import os, sys
from pathlib import Path
import json

# [NEW] Import the new config loader and review agent
from config_loader import load_app_config
from review_agent import expert_review_hypergraph, select_and_export_reference

THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    # [MODIFIED] Default config and prompts path
    cfg_path_arg = sys.argv[1] if len(sys.argv) > 1 else str(THIS_DIR / 'review_config.json')
    cfg_path = Path(cfg_path_arg).resolve()
    prompts_path = str(THIS_DIR / 'prompts')
    
    if len(sys.argv) > 1:
        print(f"ℹ️  Using custom review config path: {cfg_path}")
        print(f"ℹ️  Assuming prompts dir: {prompts_path}")

    # 1. Load the review config
    cfg = load_app_config(str(cfg_path), prompts_path)
    
    review_cfg = cfg.get("review", {})
    hypergraph_path_str = review_cfg.get("hypergraph_manifest_path")
    if not hypergraph_path_str:
        raise ValueError("Config error: 'review.hypergraph_manifest_path' is missing in review_config.json")

    # Resolve the path to the hypergraph.json relative to this config
    hypergraph_path = (THIS_DIR / hypergraph_path_str).resolve()
    if not hypergraph_path.exists():
        raise FileNotFoundError(f"Hypergraph manifest not found at: {hypergraph_path}\n"
                                f"Please run the Design_Analysis phase first.")

    threshold = float(review_cfg.get("threshold", 0.7))
    
    print('--- [HYPEREDGE: Review_Feedback] ---', flush=True)
    print('🗂  Config:', cfg_path, flush=True)
    print('📂 Prompts:', prompts_path, flush=True)
    print('📉 Reading Manifest:', hypergraph_path, flush=True)
    
    # 2. Run the Expert Review
    # This function will read the hypergraph.json, add "expert_review"
    # sections to it, and save the updated file.
    print(f"🧪 [REVIEW] Running expert review. Threshold={threshold}")
    expert_review_hypergraph(
        hypergraph_path=str(hypergraph_path),
        review_config=review_cfg, # Pass the whole review config
        prompts=cfg.get("prompts", {}) # Pass the loaded prompts
    )

    # 3. Run the Reference Selection
    # This function reads the *updated* hypergraph.json (with scores)
    # and exports the best notebook and H5 file.
    print(f"🏆 [EXPORT] Selecting and exporting best reference artifact...")
    select_and_export_reference(
        hypergraph_path=str(hypergraph_path),
        review_config=review_cfg
    )
    
    print('--- [HYPEREDGE: Review_Feedback] Complete ---', flush=True)

if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    main()
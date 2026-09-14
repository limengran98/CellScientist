# Extending CellScientist

## LLM and compute backends

Use the [configuration helper and LLM examples](llm_backends.md) to select
another served model. `llm` controls the decision policy; `model.backend`
controls candidate fitting (`torch_cuda`, `torch_cpu` or `sklearn`). Named
configurations retain the dataset contract and place results in separate
directories. Freeze the completed configuration before a formal run.

## A new morphology dataset

Prepare the [HDF5 schema](../data/README.md), then add task entries to a named
configuration's `data.tasks`. For example:

```json
{
  "dataset": "MyDataset",
  "split": "smiles",
  "path": "MyDataset/MyDataset_smiles_split.h5",
  "group_key": "smiles"
}
```

The task ID becomes `MyDataset_smiles`. A plate split uses `"split": "plate"`
and `"group_key": "plate_id"`. Paths resolve under the configured data root.
The loader accepts dataset names from the configuration, so adding a dataset
with the same feature/response schema is a configuration change.

For the companion CPG0016 files, download with:

```bash
python scripts/download_data.py --root data/hdf5 --datasets CPG0016
```

Use `CPG0016/cpg0016_plate_split.h5` and
`CPG0016/cpg0016_smiles_split.h5` as the task paths in your named configuration.
Run `inspect` to check the schema and group contract before freezing and
running the new task. The loader reads arrays into memory, and projection
fitting creates additional arrays; size RAM and GPU memory for the chosen
dataset and backend. Sequential matrix execution uses `--jobs 1`.

For a different response schema, adapt `TaskData` and `load_task_data` in
[`data.py`](../cellscientist/data.py), predictor input/output handling in
[`models.py`](../cellscientist/models.py), and metrics in
[`evaluator.py`](../cellscientist/evaluator.py) together. Preserve the roles
of training, feedback, selection and held-out reporting in the new contract.

## Candidate language and routing

| Extension point | Files and responsibilities |
| :--- | :--- |
| Candidate fields and allowed values | [`schemas.py`](../cellscientist/schemas.py): validation, serialization and stable IDs |
| Component-local edits | [`search_space.py`](../cellscientist/search_space.py): candidates, neighbors and edit legality |
| Executable predictors | [`models.py`](../cellscientist/models.py): fitting, prediction and cached representation |
| Diagnostic routing | [`controllers.py`](../cellscientist/controllers.py): address ranking, legal pair construction and selection |
| History and dependency records | [`hrt.py`](../cellscientist/hrt.py): typed state and revision provenance |
| Audit cases | [`lca_audit.py`](../cellscientist/lca_audit.py), [`routing_audit.py`](../cellscientist/routing_audit.py) |

When adding an option, give it a stable candidate ID and implement its fitting
semantics before exposing it to the controller. When adding a component
address, update its legal neighbors, diagnostics, history dependencies and
audit cases together. Rebuild the lock after a source change.

## Validate an extension

Run the regression suite, the two audits, and one small task/seed trajectory.
Check that the proposed address matches its candidate, fitted transforms use
the training partition, retention follows the configured validation metric,
and result records contain the selected candidate and held-out metrics.
For a new LLM, inspect the request trace for valid decisions, retries and
fallbacks before running the full matrix.

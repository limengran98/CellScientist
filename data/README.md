# Input data and HDF5 schema

The preprocessed datasets are hosted at
[Boom5426/CellScientist on Hugging Face](https://huggingface.co/datasets/Boom5426/CellScientist).
[`release_manifest.json`](release_manifest.json) records their pinned revision,
file sizes, SHA-256 hashes and runtime paths.

```bash
python scripts/download_data.py --root data/hdf5
```

The default selects BBBC036 and BBBC047. Add `--datasets CPG0016` for CPG0016,
or `--datasets BBBC036 BBBC047 CPG0016` for all three. `--list` displays the
download plan. The script verifies each file, arranges the directories, reuses
matching files and saves a download receipt. Downloads use
[`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/guides/download)
with the pinned commit revision.

Set `CELLSCIENTIST_DATA_ROOT` to the resulting `data/hdf5` directory:

```text
data/hdf5/
├── BBBC036/
│   ├── BBBC036_plate_split.h5
│   └── BBBC036_smiles_split.h5
├── BBBC047/
│   ├── BBBC047_plate_split.h5
│   └── BBBC047_smiles_split.h5
└── CPG0016/                         # selected with --datasets CPG0016
    ├── cpg0016_plate_split.h5
    └── cpg0016_smiles_split.h5
```

## Runtime schema

Each file contains a `combined` group. Rows refer to the same observations
across all arrays; morphology features share the same order before and after
perturbation.

| Dataset within `combined` | Shape | Meaning |
| :--- | :--- | :--- |
| `morphology_pre` | `(N, D)` | Reference morphology features |
| `morphology_post` | `(N, D)` | Perturbed morphology targets |
| `dose` | `(N,)` or `(N, 1)` | Numeric perturbation dose |
| `smiles` | `(N,)` | SMILES strings, UTF-8 or byte strings |
| `plate_id` | `(N,)` | Plate group identifiers |
| `split_id` | `(N,)` | Integer fold assignment, 1–5 for the BBBC protocol |

The BBBC protocol trains on folds 1–3, partitions fold 4 into group-disjoint
feedback and selection subsets, and evaluates retained candidates on fold 5.
Plate tasks protect `plate_id` groups; SMILES tasks protect `smiles` groups.

```bash
python -m cellscientist inspect --config configs/bbbc036_047_formal.json
```

Inspection reports sample/feature counts, partition sizes and group counts,
and checks dimensions, required fields, fold assignments and partition/group
overlap. New datasets with this schema can be added through `data.tasks` as
described in the [extension guide](../docs/extending.md).

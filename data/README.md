# BBBC input layout

This release intentionally does not redistribute dataset files. Set
`CELLSCIENTIST_DATA_ROOT` to a directory containing the four preprocessed HDF5
files below:

```text
BBBC036/BBBC036_plate_split.h5
BBBC036/BBBC036_smiles_split.h5
BBBC047/BBBC047_plate_split.h5
BBBC047/BBBC047_smiles_split.h5
```

Each file must contain a `combined` HDF5 group with the datasets
`morphology_pre`, `morphology_post`, `dose`, `smiles`, `plate_id`, and
`split_id`. `split_id` uses folds 1--5. The release trains on folds 1--3,
splits fold 4 into group-disjoint feedback and selection partitions, and holds
fold 5 out until final reporting.

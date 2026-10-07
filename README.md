# DP-LLPR early test package

This repository contains an early prototype for testing a configuration-level
last-layer prediction rigidity (LLPR) workflow with DeePMD-kit PyTorch models.
It is intended for technical inspection and testing and is not yet a validated
or production-ready implementation.

## Files

- `extract_dp_features.py`: extracts DeePMD last-layer features from reference
  structures and writes a reference NPZ file.
- `compute_dp_llpr.py`: reads the reference NPZ and a LAMMPS dump trajectory,
  calculates rigidity scores, and prints selected timesteps.
- `DP-LLPR-LIMITATIONS.md`: describes the current scope and known limitations.

The model and test data are distributed separately. To reproduce the supplied
test, copy the contents of the private test-data package into this directory.
The resulting layout should contain `frozen_model.pth`, `dp_features.npz`,
`all.lammpstrj`, and `training_data/` alongside the two scripts.

## Requirements

- Python 3.12
- DeePMD-kit 3.1.1 with the PyTorch backend
- NumPy
- PyTorch

## Step 1: extract reference features

```bash
python extract_dp_features.py \
    --model frozen_model.pth \
    --dataset training_data \
    --set set.000 \
    --frames all \
    --batch-size 32 \
    --device auto \
    --output dp_features.npz
```

The supplied `dp_features.npz` was generated from the included private test
model and reference structures. Running this step again should regenerate the
reference feature matrix.

## Step 2: compute candidate rigidity

```bash
python compute_dp_llpr.py \
    --model frozen_model.pth \
    --reference dp_features.npz \
    --candidates all.lammpstrj \
    --device auto \
    --batch-size 32 \
    --min-frame-gap 10 \
    --output dp-llpr-scores.npz
```

The program prints reference and candidate statistics followed by the selected
candidate frame indices, LAMMPS timesteps, and rigidity values.

Useful selection options:

```text
--max-rigidity    select candidates below a rigidity threshold
--top-k           keep at most this many candidates
--min-frame-gap   minimum frame-index separation between selections
```

The current trajectory reader expects a LAMMPS custom dump containing
`id type x y z` columns. Each `BOX BOUNDS` row must contain three numeric values
in the LAMMPS restricted-triclinic bounds layout. The `xy`, `xz`, and `yz` tilt
factors may all be zero, so an untilted orthogonal cell is supported when the
third column is explicitly present. The standard two-column orthogonal
`BOX BOUNDS` format is not yet supported.

The LAMMPS type numbers must map to the DeePMD type order as
`1 -> type_map[0]`, `2 -> type_map[1]`, and so on. The model and reference NPZ
must have been generated from the same trained model and feature construction.

See `DP-LLPR-LIMITATIONS.md` before interpreting the scores.

# DP-LLPR early test version: scope and limitations

This project is an independent early-stage adaptation of a configuration-level
LLPR workflow for DeePMD-kit PyTorch models. It is provided for inspection and
testing and is not yet a validated or production-ready implementation.

The currently tested descriptor path is DeePMD `se_e2_a`. The current method uses
last-layer representations to screen candidate configurations from a LAMMPS
trajectory. Low rigidity is intended to serve as a potential indicator of
insufficient training-data coverage; it is not proof of extrapolation, physical
instability, or a large DFT error.

The following points still require further work and validation:

- feature scaling and the choice of regularization;
- the relationship between rigidity and DeePMD energy/force losses;
- support for the standard two-column orthogonal `BOX BOUNDS` format and broader
  triclinic-box formats (the current reader requires three numeric values per
  row in the restricted-triclinic layout, although all tilt factors may be zero);
- broader support for DeePMD dataset layouts and automatic merging of multiple
  `set.*` directories;
- stronger consistency checks between the reference features and the model;
- force-aware LLPR;
- automatic export of selected structures to `coord.xyz`;
- DFT energy and force validation of selected configurations.

This repository does not claim original authorship of the LLPR method. Please
consult the original COSMO/LLPR publications and software for the authoritative
methodological source.

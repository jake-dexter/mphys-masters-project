# Muon Electric Dipole Moment – Sensitivity Optimisation

This repository contains the code and analysis for my MPhys project investigating methods to improve sensitivity to a possible muon electric dipole moment (EDM) in the Fermilab muon g-2 experiment.

The repository is primarily used for version tracking, exploratory analysis, and maintaining a record of the computational work carried out throughout the project.

## Project Overview

A non-zero muon EDM would provide evidence for additional sources of CP violation beyond the Standard Model. In the Fermilab muon g-2 experiment, such an EDM would manifest as a small tilt in the muon spin precession, which can be inferred from the angular and energy distribution of decay positrons.

This project studies how sensitivity to this signal can be improved by applying weighting schemes over the positron angle-energy phase space. Rather than treating all regions of phase space equally, the analysis constructs weights that enhance regions carrying greater sensitivity to the EDM signal while accounting for statistical uncertainty.

## Main Components

The project includes work on:

* defining and evaluating a Figure of Merit (FOM) for EDM sensitivity;
* testing empirical weighting schemes;
* performing numerical optimisation using CMA-ES;
* deriving and evaluating an analytic stationarity weight;
* comparing analytic and numerical weighting schemes;
* extracting the EDM tilt parameter using fitted observables;
* performing a log-likelihood cross-check;
* studying possible detector acceptance effects.

## Repository Contents

The repository currently contains a mixture of Python scripts, Jupyter notebooks, stored weight maps, and final dissertation figures.

Main files and folders include:

* `Functions.py` – shared analysis functions used across the project.
* `main_funcs.py` – additional helper functions for the main analysis workflow.
* `weighting.py` – code associated with weighting calculations.
* `extract_weights.py` – script for extracting or processing optimised weights.
* `interpolate_weights.py` – script for constructing interpolated weight maps across fractional energy.
* `main.ipynb` – main exploratory analysis notebook.
* `acceptance-weighting.ipynb` – studies including detector acceptance effects.
* `iterative-weight.ipynb` – investigation of iterative/self-consistent weighting behaviour.
* `smoothing_with_iterative.ipynb` – tests involving smoothing and iterative weighting behaviour.
* `extracting_delta.ipynb` – extraction of the EDM tilt parameter from observables.
* `log-likelihood.ipynb` – likelihood-based EDM extraction cross-check.
* `Weight_Maps2/` – stored numerical weight maps.
* `partial weights/` – intermediate or partial weighting outputs.
* `final diss plots/` – figures used in the final dissertation.
* `phd interview/` – material prepared for related presentation/interview work.
* `analytic_weights.npz` – stored analytic weighting results.
* `numerical_weights.npz` – stored numerical weighting results.
* `total_interpolated_weights.npz` – interpolated two-dimensional weighting map.

## Data

The simulation data files are not included in this repository due to size.

They should be stored locally in:

```text
data/With EDM/
data/Without EDM/
```

The data can be downloaded from:

```text
https://hep.ph.liv.ac.uk/~jprice/Project2025-26/
```

The analysis assumes that the `.pkl` files are placed in the directory structure above.

## Dependencies

The analysis is written in Python and uses standard scientific computing packages, including:

* NumPy
* SciPy
* Matplotlib
* pickle
* pathlib
* nevergrad
* Jupyter Notebook

Some notebooks or scripts may require additional plotting or utility packages depending on the specific analysis being run.

## Typical Workflow

The repository was developed as an exploratory research project, so the notebooks and scripts are not arranged as a single automated pipeline. A typical workflow is:

1. Download the simulation data and place it in the local `data/` directory.
2. Load the probability density data for the with-EDM and without-EDM samples.
3. Compute weighted and unweighted observables.
4. Evaluate the Figure of Merit for different weighting schemes.
5. Generate numerical weights using CMA-ES or load stored weight maps.
6. Compare numerical and analytic weighting schemes.
7. Extract the EDM tilt parameter from the fitted observables.
8. Perform likelihood-based cross-checks where required.

The main analysis is split across scripts and notebooks rather than being run from a single command.

## Key Results

The final analysis found that:

* optimised weighting schemes improve EDM sensitivity relative to the unweighted case;
* the CMA-ES and analytic weighting schemes show close structural agreement;
* the analytic and numerical weights have a cosine similarity of approximately 0.996 at a representative energy slice;
* the analytic weighting gives an improvement of approximately 49% in the Figure of Merit;
* the weighted observable allows the injected EDM tilt parameter to be recovered through a linear calibration.

## Dissertation

The full dissertation is included in this repository:

[Dissertation.pdf](./Masters_Project.pdf)

## Notes

This repository is intended to support the dissertation analysis rather than provide a polished standalone software package. Paths, notebooks, and scripts may therefore reflect the exploratory development of the project.

## Author

Jake Dexter
MPhys Physics
University of Liverpool

## License

This repository is released under the MIT License.

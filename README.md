# sesl - Structural Eigenmode Source Localisation

A Python package for source localisation using the structural eigenmode approach.  

## Reference

If you use this code in your research, please cite:

>Siu, P. H., Karoly, P. J., Mansour, S. L., Soto-Breceda, A., Kuhlmann, L., Cook, M. J., & Grayden, D. B.
>“Structural Eigenmodes of the Brain to Improve the Source Localization of EEG: Application to Epileptiform Activity.” 
>Advanced Science (2026): e16802. https://doi.org/10.1002/advs.202516802

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
[![DOI](https://img.shields.io/badge/DOI-10.1002/advs.202516802-blue)](https://doi.org/10.1002/advs.202516802)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

The purpose of this project is to provide a Python package for source localisation using the structural eigenmode approach. This method allows for the generation of structural eigenmodes (either geometric or connectome-based) and solves the source localisation problem effectively.  
For more details, please refer to the original paper.

## Features

- ✅ Generates structural eigenmodes (geometric or connectome)
- ✅ Solves the source localisation problem using the structural eigenmode approach
- 🚧 Would be best to remove dependencies on tvb

## Installation

### Requirements

- Python 3.10
- mne, lapy, and more, see `requirements_sesl.txt` for an abbreviated list.
- Working set of dependencies is in `requirements.txt`.

### Steps

```bash
# Clone the repository
git clone https://github.com/Spokhim/sesl.git
cd sesl

# Create and activate a virtual environment
python -m venv .sesl
source .sesl/bin/activate  # On Windows: .sesl\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
Useful functions can be found in: 
- `dynsim_fns.py`: Functions with tvb dependencies
- `gem_solver.py`: Functions for solving the GEM problem
- `plot_fns.py`: Functions for plotting results
- `useful_fns.py`: Miscellaneous utility functions

## Project Structure

```
sesl/
├── External_Resources/          # Holds external resources such as data files
│   └── HCP_data/                # Holds HCP data files
├── Ignore/                      # Holds miscellaneous files to be used in the pipeline such as forward models
│   └── Simulations/             # Holds simulation files
├── dynsim_fns.py                # Functions with TVB dependencies
├── gem_solver.py                # GEM problem solvers
├── plot_fns.py                  # Plotting utilities
├── useful_fns.py                # Miscellaneous utilities
├── requirements.txt
├── sample_sesl_pipeline.ipynb   # Example notebook for using the package
└── STC_Analysis.ipynb           # Example notebook for running analysis from original paper
```

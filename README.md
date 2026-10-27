# Empirical calibration of Bayesian predictive intervals for cross-species, cross-endpoint oral toxicity translation

**Study type:** proof-of-concept methodological evaluation  
**Analysis set:** 289 rat–human compound pairs  
**Status:** research code and analysis outputs; use the manuscript for the final study description once the manuscript is finalized.

This repository contains the analysis scripts, configuration, derived analysis dataset, tabular outputs, and figures for the study evaluating prediction-interval calibration when translating rat oral LD50 values to human oral TDLo values.

> **Reproducibility note.** The pipeline and derived 289-row analysis set have been checked against the supplied derived dataset. The SHA-256 recorded in `results/run_manifest.json` does not match the SHA-256 of the source `dataset.txt` copy used during the local rebuild. The reason has not been established; therefore, do not treat the manifest hash as verified for that copy. See [Data provenance](#data-provenance).

## Repository layout

```text
code/
  build_dataset.py       # Build the 289-compound analysis set and selection comparison
  run_validation.py     # Repeated-split validation
  run_tail.py            # Tail-holdout stress test
  run_diagnostics.py    # Model and posterior predictive diagnostics
  analyze.py             # Summaries, coverage and uncertainty analyses
  make_figures.py        # Generate study figures
  lib.py                 # Shared methods
  config.json            # Analysis settings and seeds
results/
  analysis_set_289.csv
  dataset_summary.json
  diagnostics.json
  null_mace.json
  results.json
  run_manifest.json
  selection_comparison.csv
  tail_info.json
figures/                 # Figures generated for the study
```

## Data provenance

The source data are from the Acute Toxicity FSL dataset, available at:

- Source repository: https://github.com/LuJiangTHU/Acute_Toxicity_FSL
- Source file: `data/dataset.txt`
- Source repository states the data are available under CC BY 4.0; verify the source license and attribution requirements when redistributing.

The full source file is **not bundled** here. Download `data/dataset.txt` from the source repository and place it at `code/dataset.txt`. The file is intentionally excluded from version control to avoid duplicating a large third-party dataset. The derived `results/analysis_set_289.csv` is included for inspection and comparison.

The `dataset_sha256` in `results/run_manifest.json` is the hash recorded in the original analysis manifest. It differed from the hash of the uploaded source copy used in a local rebuild. This discrepancy is disclosed rather than silently overwritten. Before publication, record the exact source-file hash for the final rerun and update the manifest with a documented change.

## Environment

The recorded environment is listed in `results/run_manifest.json`. The pipeline was authored for Python 3.12 and uses NumPy, SciPy, pandas, Matplotlib, and RDKit. Install the dependencies from `requirements.txt` in a clean virtual environment.

Example using a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If an exact RDKit wheel is unavailable for your platform, install RDKit using its official installation instructions and record the installed version.

## Reproduce the pipeline

From the repository root:

```bash
mkdir -p out
cd code
# First download the source data to code/dataset.txt, as described above.
python build_dataset.py
python run_validation.py
python run_tail.py
python run_diagnostics.py
python analyze.py
python make_figures.py
```

Scripts use paths relative to `code/`; derived data and result files are written to `../out/`, and figures are written to `../figures/`. Compare the regenerated outputs with the committed `results/` and `figures/` after the full pipeline completes. Some stages may be computationally intensive. Do not interpret an interrupted run as a successful reproduction.

The main settings, priors, split counts, bootstrap count, and base seed are in `code/config.json`. The recorded seed scheme and package versions are in `results/run_manifest.json`.

## Results and interpretation

The included outputs are intended to make the reported analysis inspectable. Prediction intervals should be interpreted in the context of the dataset selection, cross-species/cross-endpoint translation, repeated-split design, and the limitations described in the manuscript. This is a proof-of-concept calibration study, not a clinically validated toxicology model.

## Citation

Please cite the associated manuscript once its bibliographic details are available. Until publication, a citation can be added here with the title, authors, journal, DOI or preprint URL. Do not invent a DOI or publication status.

## License

No repository-wide license has been assigned yet. The upstream source dataset has its own stated license and attribution requirements; those do not automatically determine the license for the study code. Add the authors' chosen code license before describing the code as open source.

## Contact

For questions about the analysis, contact the corresponding author listed in the associated manuscript.

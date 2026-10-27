# Reproducibility status at repository assembly

- The 289-row derived analysis dataset reconstructed from the supplied source copy matched the separately supplied `analysis_set_289.csv` exactly in row/column order and values.
- The recorded source-dataset SHA-256 in `results/run_manifest.json` did not match the SHA-256 of the uploaded source copy used for the local rebuild. This remains unexplained and must be reconciled in the final release.
- The user reports that the remaining statistical stages have completed and that the numerical audit agrees with the manuscript. The original supplied analysis archive includes result JSON/CSV files and figures; the complete final rerun logs and intermediate validation arrays were not present in that archive at repository assembly.
- Do not claim that this exact packaged snapshot was independently regenerated end-to-end unless the full pipeline has been run from a clean environment using the identified source file.

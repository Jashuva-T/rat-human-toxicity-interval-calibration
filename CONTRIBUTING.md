# Contribution and reproducibility notes

- Keep changes to analysis code reviewable and describe their impact on reported results.
- Do not commit the full third-party `dataset.txt`; follow the data-source terms and use the documented download step.
- For any result-changing code or configuration change, rerun the full pipeline and update affected result files and figures together.
- Record Python and package versions, source-data SHA-256, random seeds, and the command sequence used for the final run.
- Do not replace a discrepant result in a manuscript or table by hand. Trace it to the relevant code and regenerate the affected outputs.
- Before a public release, resolve the source-data hash discrepancy in the README and run manifest, and choose an explicit license for original code.

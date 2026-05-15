# Bootstrap CI Summary

For uncertainty estimation, 95% confidence intervals were computed by nonparametric bootstrap resampling of the held-out test set. We sampled test cases with replacement for 2000 bootstrap replicates using random seed 2026. For each replicate, accuracy, precision, sensitivity, specificity, F1-score, and ROC-AUC were recomputed. Intervals were defined by the 2.5th and 97.5th percentiles of the bootstrap distribution. Bootstrap samples containing only one class were excluded for ROC-AUC only.

## Notes
- All available normalized test prediction CSVs were processed.

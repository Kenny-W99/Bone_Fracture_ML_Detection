# Journal Strengthening Report

## 1. What Was Implemented

A reproducible strengthening package was added under `experiments/journal_strengthening/`. It includes repository audit, CNN prediction normalization/generation, validation-based threshold tuning, bootstrap confidence intervals, guarded multi-seed training, guarded VLM repeated trials, external-validation scaffold, ResNet50 error/subgroup analysis, publication figure generation, and a safe runner.

## 2. What Was Successfully Run

Safe analyses run by the wrapper are recorded in `logs/run_all_strengthening.log` when the wrapper is used. Generated CSVs and summaries listed below indicate completed steps.

## 3. What Could Not Be Run and Why

- Multi-seed training is not run unless `RUN_MULTISEED=1` is set.
- Repeated VLM API calls are not run unless `RUN_VLM=1` is set and the provider API key is available in the environment.
- External validation is not run unless an external dataset exists and label mapping is validated.

## 4. New Result Files Generated

- `experiments/journal_strengthening/00_repo_audit.md`: present
- `experiments/journal_strengthening/prediction_generation_notes.md`: present
- `experiments/journal_strengthening/results/threshold_tuning_results.csv`: present
- `experiments/journal_strengthening/results/threshold_curves_by_model.csv`: present
- `experiments/journal_strengthening/results/bootstrap_ci_results.csv`: present
- `experiments/journal_strengthening/error_analysis_summary.md`: present
- `experiments/journal_strengthening/figure_generation_notes.md`: present
- `experiments/journal_strengthening/resnet50_multiseed_summary.md`: present
- `experiments/journal_strengthening/vlm_repeated_trials_summary.md`: present
- `experiments/journal_strengthening/external_validation_summary.md`: present

## 5. Commands That Were Run

Primary intended command:

```bash
bash experiments/journal_strengthening/run_all_strengthening.sh
```

Optional commands:

```bash
RUN_MULTISEED=1 bash experiments/journal_strengthening/run_all_strengthening.sh
RUN_VLM=1 VLM_MODEL=gpt-4o-mini bash experiments/journal_strengthening/run_all_strengthening.sh
```

## 6. Recommended Tables to Add

- `tables/threshold_tuning_table.tex`
- `tables/bootstrap_ci_table.tex`
- Add `tables/resnet50_multiseed_table.tex`, `tables/vlm_repeated_trials_table.tex`, and `tables/external_validation_table.tex` only after those experiments are actually run.

## 7. Recommended Figures to Add

- `figures/resnet50_threshold_tuning_curve.png`
- `figures/bootstrap_ci_f1_point_range.png`
- Add multi-seed, repeated-VLM, and external-validation figures only when real outputs exist.

## 8. Suggested Methods Text

Decision thresholds for CNNs were tuned using validation-set predictions only. Thresholds from 0.01 to 0.99 were evaluated, and selected thresholds were then applied once to the held-out test set. Test-set uncertainty was estimated using nonparametric bootstrap resampling with replacement across test images, using 2,000 replicates and the 2.5th and 97.5th percentiles as 95% confidence intervals.

## 9. Suggested Results Text

Newly computed ResNet50 validation-selected threshold was 0.57; test F1=0.6781, sensitivity=0.7315, specificity=0.9089.
Newly computed ResNet50 tuned-threshold F1 95% CI was 0.6781 [0.6041, 0.7458].

## 10. Suggested Discussion Text

Validation-selected thresholds provide a more transparent operating-point analysis than a fixed 0.50 cutoff, especially in a medical-imaging screening context where sensitivity and specificity trade-offs must be stated explicitly. These results should be interpreted as research-only evidence on FracAtlas rather than clinical diagnostic performance.

## 11. Suggested Limitations Text

The experiments remain limited by the size and source distribution of FracAtlas. External validation, repeated VLM trials, and multi-seed retraining should be reported only after they are completed with real data and documented compute/API settings. The models are not intended for clinical use.

## 12. Suggested Updated Abstract Paragraph

We further assessed operating-point selection and statistical uncertainty by tuning CNN decision thresholds on the validation set and estimating 95% confidence intervals on the held-out test set using bootstrap resampling. These additions quantify performance variability and make the sensitivity-specificity trade-off explicit without using test data for model selection.

## 13. Suggested Updated Title

Transfer Learning and Vision-Language Model Baselines for Research-Only Bone Fracture Classification in FracAtlas X-rays

## 14. Recommended Journal-Submission Strategy

Lead with the validated CNN results, threshold tuning, bootstrap intervals, and error analysis. Present the VLM baseline as exploratory, not as a clinical comparator. Treat external validation and multi-seed stability as high-priority additions before submission if time and data availability permit.

## Result Provenance

Threshold tuning and bootstrap intervals generated by this package are newly computed from available prediction CSVs. Existing CNN test predictions and VLM baseline summaries are pre-existing project outputs unless explicitly regenerated by the scripts. No metrics are fabricated.

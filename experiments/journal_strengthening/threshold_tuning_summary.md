# Threshold Tuning Summary

Decision thresholds were selected using validation predictions only and then evaluated once on the held-out test set.

## Selected Thresholds
- ResNet50 / fixed_0_50: threshold 0.50; test F1 0.6534, recall 0.7593, specificity 0.8792.
- ResNet50 / max_val_f1: threshold 0.57; test F1 0.6781, recall 0.7315, specificity 0.9089.
- ResNet50 / max_youden_j: threshold 0.54; test F1 0.6751, recall 0.7407, specificity 0.9030.
- ResNet50 / high_sensitivity_precision_constrained: threshold 0.19; test F1 0.4480, recall 0.9167, specificity 0.5347.
- MobileNetV2 / fixed_0_50: threshold 0.50; test F1 0.6311, recall 0.7130, specificity 0.8832.
- MobileNetV2 / max_val_f1: threshold 0.69; test F1 0.6354, recall 0.5648, specificity 0.9545.
- MobileNetV2 / max_youden_j: threshold 0.35; test F1 0.5804, recall 0.7685, specificity 0.8119.
- MobileNetV2 / high_sensitivity_precision_constrained: threshold 0.13; test F1 0.4626, recall 0.9167, specificity 0.5624.
- DenseNet121 / fixed_0_50: threshold 0.50; test F1 0.5923, recall 0.6389, specificity 0.8891.
- DenseNet121 / max_val_f1: threshold 0.61; test F1 0.6238, recall 0.5833, specificity 0.9386.
- DenseNet121 / max_youden_j: threshold 0.35; test F1 0.5329, recall 0.7870, specificity 0.7505.
- DenseNet121 / high_sensitivity_precision_constrained: threshold 0.14; test F1 0.3905, recall 0.9167, specificity 0.4059.
- Custom CNN / fixed_0_50: threshold 0.50; test F1 0.4669, recall 0.5556, specificity 0.8238.
- Custom CNN / max_val_f1: threshold 0.50; test F1 0.4669, recall 0.5556, specificity 0.8238.
- Custom CNN / max_youden_j: threshold 0.45; test F1 0.4479, recall 0.6574, specificity 0.7267.
- Custom CNN / high_sensitivity_precision_constrained: threshold 0.20; test F1 0.3333, recall 0.9537, specificity 0.1941.

## High-Sensitivity Rule Notes
- No high-sensitivity fallback was needed.

## Recommended Paper Reporting
- ResNet50: report `max_val_f1` threshold 0.57 as the primary tuned-threshold result; keep fixed 0.50 as a comparator.
- MobileNetV2: report `max_val_f1` threshold 0.69 as the primary tuned-threshold result; keep fixed 0.50 as a comparator.
- DenseNet121: report `max_val_f1` threshold 0.61 as the primary tuned-threshold result; keep fixed 0.50 as a comparator.
- Custom CNN: report `max_val_f1` threshold 0.50 as the primary tuned-threshold result; keep fixed 0.50 as a comparator.

Validation-based threshold tuning is stronger than using a fixed 0.50 cutoff because neural-network probabilities are often not calibrated to a universal operating point. Selecting the threshold on validation data makes the sensitivity-specificity trade-off explicit while preserving the test set for unbiased final evaluation.

# Error Analysis Summary

ResNet50 error analysis used threshold `0.57`. This is the validation-selected max-F1 threshold when available; otherwise 0.50.

- False negatives: 29
- False positives: 46
- True positives: 79
- True negatives: 459
- Subgroup metrics were computed using FracAtlas test metadata columns.

Recommended Grad-CAM examples for the discussion should be sampled from the saved false-negative and false-positive CSVs, prioritizing high-confidence errors and clinically interpretable metadata groups where available.

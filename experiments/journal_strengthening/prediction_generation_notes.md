# CNN Prediction Generation Notes

- Generated `val` predictions for `resnet50` from checkpoint: `experiments/journal_strengthening/predictions/resnet50_val_predictions.csv`
- Normalized existing test predictions for `resnet50`: `experiments/journal_strengthening/predictions/resnet50_test_predictions.csv`
- Generated `val` predictions for `mobilenetv2` from checkpoint: `experiments/journal_strengthening/predictions/mobilenetv2_val_predictions.csv`
- Normalized existing test predictions for `mobilenetv2`: `experiments/journal_strengthening/predictions/mobilenetv2_test_predictions.csv`
- Generated `val` predictions for `densenet121` from checkpoint: `experiments/journal_strengthening/predictions/densenet121_val_predictions.csv`
- Normalized existing test predictions for `densenet121`: `experiments/journal_strengthening/predictions/densenet121_test_predictions.csv`
- Generated `val` predictions for `custom_cnn` from checkpoint: `experiments/journal_strengthening/predictions/custom_cnn_val_predictions.csv`
- Normalized existing test predictions for `custom_cnn`: `experiments/journal_strengthening/predictions/custom_cnn_test_predictions.csv`

To regenerate manually:

```bash
python experiments/journal_strengthening/generate_cnn_predictions.py --force
```

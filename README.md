# Bone Fracture ML Detection

A PyTorch-based machine learning project for binary bone fracture detection using the **FracAtlas X-ray dataset**. This project compares a custom CNN baseline, ImageNet-pretrained transfer learning models, Grad-CAM explainability outputs, and a small zero-shot vision-language model (VLM) pilot baseline for X-ray fracture classification.

> **Important:** This project is for educational and research purposes only. It is **not** intended for clinical diagnosis or medical decision-making.

## Latest Update: Journal-Strengthening Package

This repository now includes a reproducible journal-strengthening experiment package under:

```text
experiments/journal_strengthening/
```

The package addresses several manuscript-review weaknesses by adding:

- Validation-set decision-threshold tuning for CNN models
- Bootstrap 95% confidence intervals for test-set metrics
- Normalized validation/test prediction CSVs
- ResNet50 error analysis and subgroup analysis
- Paper-ready LaTeX tables and figures
- Guarded scaffolds for multi-seed ResNet50 training, repeated VLM trials, and external validation

The safe non-expensive analysis can be run with:

```bash
bash experiments/journal_strengthening/run_all_strengthening.sh
```

Expensive or paid experiments are opt-in only:

```bash
RUN_MULTISEED=1 bash experiments/journal_strengthening/run_all_strengthening.sh
RUN_VLM=1 VLM_MODEL=gpt-4o-mini bash experiments/journal_strengthening/run_all_strengthening.sh
```

No API keys, private `.env` files, raw image folders, or newly generated model checkpoints are committed.

## Paper and Reproducibility

- The final compiled manuscript PDF is stored in `paper/final_manuscript.pdf`.
- The expected manuscript source path is `paper/final_manuscript.tex`. If the full exported LaTeX source is available separately, replace the placeholder file before journal submission.
- Journal-strengthening results, scripts, tables, and figures are stored in `experiments/journal_strengthening/`.
- The full raw FracAtlas image dataset and model checkpoints are not fully committed because of repository size limits.
- Users can reproduce the reported experiments by downloading FracAtlas, placing the images under `data/FracAtlas/images/`, and running the provided training, evaluation, and journal-strengthening scripts.

## Project Overview

This project builds a complete machine learning pipeline for bone fracture classification from musculoskeletal X-ray images. The main goal is to study how well task-specific CNN models can detect fractures, and how their performance compares with a small pilot baseline using a general-purpose multimodal LLM / VLM.

The project includes:

- Dataset loading from CSV metadata
- Train / validation / test splitting
- X-ray image preprocessing
- Medical-image-safe data augmentation
- Custom CNN and transfer learning models
- Training and evaluation scripts
- Clinical-style metrics, including recall / sensitivity and false negatives
- Grad-CAM visualization for interpretability
- Optional VLM baseline using image-input APIs such as OpenAI, Gemini, or Anthropic
- Reproducibility utilities and saved experiment outputs
- Journal-strengthening analyses for threshold tuning, confidence intervals, error analysis, and paper-ready tables/figures

## Motivation

Bone fracture detection is an important medical imaging task. In a real screening context, missing a fracture can delay treatment and potentially harm patients. Because of this, this project does not only report accuracy. It also reports recall, specificity, F1-score, ROC-AUC, and confusion matrix values.

For this task, **recall / sensitivity** is especially important because a false negative means the model missed an actual fracture.

## Dataset

This project uses the **FracAtlas X-ray dataset**, which contains musculoskeletal radiographs with fracture-related labels.

The expected dataset structure is:

```text
data/FracAtlas/
├── images/
├── dataset.csv
├── train.csv
├── val.csv
└── test.csv
```

The main CSV columns used by this project are:

| Column | Description |
|---|---|
| `image_id` | Image filename |
| `fractured` | Binary label, where `1` means fracture and `0` means no fracture |

Current full metadata split used in this project:

| Split | Number of Images |
|---|---:|
| Train | 2,858 |
| Validation | 612 |
| Test | 613 |
| Total | 4,083 |

### Repository Image Subset Note

The full CSV metadata contains 4,083 image records, but the GitHub repository may only include a subset of image files because of repository size limits. For quick runnable demos in GitHub Codespaces, use:

```text
data/FracAtlas/dataset_available.csv
data/FracAtlas/train_available.csv
data/FracAtlas/val_available.csv
data/FracAtlas/test_available.csv
```

These files are generated from the images currently available under `data/FracAtlas/images/`. To reproduce the full CNN experiment, download the complete FracAtlas image set and place all images under:

```text
data/FracAtlas/images/
```

## Task Definition

This is a binary image classification task.

Given one X-ray image, the model predicts:

| Label | Meaning |
|---:|---|
| `0` | No fracture |
| `1` | Fracture |

The CNN models output one logit. During training, the project uses `BCEWithLogitsLoss`, which is appropriate for binary classification.

During inference, the output logit is converted into a probability using sigmoid:

```text
probability = sigmoid(logit)
```

A default threshold of `0.5` is used:

```text
probability >= 0.5 → fracture
probability < 0.5  → no fracture
```

For the journal-strengthening analyses, additional thresholds are selected using validation-set predictions only and then evaluated on the held-out test set. This avoids selecting thresholds on test data while making the sensitivity-specificity trade-off explicit.

## Model Architectures

This project compares four CNN-based models and one optional VLM pilot baseline.

### 1. Custom CNN

A simple convolutional neural network built from scratch. It is used as a baseline model.

### 2. MobileNetV2

A lightweight ImageNet-pretrained model. It is useful for efficient inference and smaller model size.

### 3. ResNet50

A deeper residual network with skip connections. In the current CNN experiments, ResNet50 achieved the strongest overall performance.

### 4. DenseNet121

A densely connected CNN architecture that is commonly used in medical imaging research because it can reuse features effectively.

For the pretrained CNN models, the original ImageNet classification head is replaced with a single-output binary classification layer.

### 5. Vision-Language Model Baseline

The VLM baseline is a zero-shot comparison using a general-purpose multimodal model. The VLM is **not fine-tuned** on FracAtlas. Instead, each X-ray image is sent with a standardized prompt asking the model to classify the image as fracture-positive or no-fracture and return structured JSON.

This VLM experiment is included only as a research comparison. It should not be interpreted as medical diagnosis.

## Preprocessing and Augmentation

The data pipeline is designed for X-ray images, where preserving anatomical structure is important.

### Validation and Test Preprocessing

For validation and test images, the pipeline applies:

1. Load image with PIL
2. Convert image to RGB
3. Pad image to a square shape while preserving aspect ratio
4. Resize to `224 x 224`
5. Convert to PyTorch tensor
6. Normalize using ImageNet mean and standard deviation

Padding before resizing is used to avoid stretching or distorting bone structures.

### Training Augmentation

For training images, the project applies realistic augmentations:

- Small rotation
- Small affine translation
- Brightness and contrast jitter
- Optional horizontal flip

The pipeline intentionally avoids vertical flipping and random cropping because these transformations can be unrealistic or harmful for medical X-ray interpretation.

## Repository Structure

```text
Bone_Fracture_ML_Detection/
├── data/
│   └── FracAtlas/
│       ├── images/
│       ├── dataset.csv
│       ├── dataset_available.csv
│       ├── train.csv
│       ├── train_available.csv
│       ├── val.csv
│       ├── val_available.csv
│       ├── test.csv
│       ├── test_available.csv
│       └── test_vlm_balanced_40.csv
│
├── outputs/
│   ├── results_summary.csv
│   ├── results_summary_with_vlm.csv
│   └── other training / evaluation outputs
│
├── paper/
│   ├── final_manuscript.pdf
│   ├── final_manuscript.tex
│   └── old_versions/
│
├── experiments/
│   └── journal_strengthening/
│       ├── README.md
│       ├── run_all_strengthening.sh
│       ├── threshold_tuning.py
│       ├── bootstrap_ci.py
│       ├── error_analysis.py
│       ├── results/
│       ├── tables/
│       ├── figures/
│       └── JOURNAL_STRENGTHENING_REPORT.md
│
├── research_context/
├── shared/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── evaluate_vlm.py
│   ├── gradcam.py
│   ├── inspect_dataset.py
│   ├── make_vlm_subset.py
│   ├── models.py
│   ├── split_dataset.py
│   ├── train.py
│   ├── vlm_baseline.py
│   └── append_vlm_to_summary.py
│
├── README_VLM_BASELINE.md
├── requirements.txt
├── requirements-vlm.txt
├── .env.example
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Kenny-W99/Bone_Fracture_ML_Detection.git
cd Bone_Fracture_ML_Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

For macOS / Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

Install the core dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available in your environment, install the core packages manually:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib pillow
```

For the optional VLM baseline, install the VLM dependencies:

```bash
pip install -r requirements-vlm.txt
```

If you are using a CUDA GPU, install the correct PyTorch version from the official PyTorch website. On Apple Silicon, the scripts can use Apple MPS when available.

## API Key Setup for VLM Baseline

The VLM baseline requires an API key only if you run a real provider such as OpenAI, Gemini, or Anthropic. The mock provider does not require an API key.

Create a local `.env` file from the example file:

```bash
cp .env.example .env
```

Add your own key to `.env`. For example:

```bash
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

Then load the environment variables in the terminal:

```bash
set -a
source .env
set +a
```

The `.env` file is ignored by Git and should never be committed.

The repository includes a safe `.env.example` with empty placeholders only.

## How to Run

### 1. Inspect the Dataset

Before training, check that the dataset and image paths are correct:

```bash
python src/inspect_dataset.py \
  --csv_path data/FracAtlas/dataset.csv \
  --image_dir data/FracAtlas/images
```

This script helps verify:

- CSV columns
- Label distribution
- Image filename column
- Whether the image files exist on disk

### 2. Split the Dataset

If you need to regenerate the train / validation / test CSV files, run:

```bash
python src/split_dataset.py \
  --csv_path data/FracAtlas/dataset.csv \
  --output_dir data/FracAtlas \
  --label_column fractured \
  --train_size 0.70 \
  --val_size 0.15 \
  --test_size 0.15 \
  --seed 42
```

This creates:

```text
train.csv
val.csv
test.csv
```

### 3. Train a CNN Model

Example: train ResNet50.

```bash
python src/train.py \
  --model resnet50 \
  --train_csv data/FracAtlas/train.csv \
  --val_csv data/FracAtlas/val.csv \
  --test_csv data/FracAtlas/test.csv \
  --image_dir data/FracAtlas/images \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_workers 0 \
  --use_pos_weight \
  --output_dir outputs/resnet50_5ep
```

Supported CNN model names:

```text
custom_cnn
mobilenet_v2
resnet50
densenet121
```

Useful arguments:

| Argument | Description |
|---|---|
| `--model` | Model architecture |
| `--train_csv` | Training CSV path |
| `--val_csv` | Validation CSV path |
| `--test_csv` | Test CSV path |
| `--image_dir` | Image folder path |
| `--epochs` | Number of training epochs |
| `--batch_size` | Batch size |
| `--lr` | Learning rate |
| `--num_workers` | DataLoader workers |
| `--output_dir` | Output folder |
| `--use_pos_weight` | Use class weighting for imbalance |
| `--seed` | Random seed |

The training script saves outputs such as:

```text
outputs/<run_name>/
├── best_model.pth
├── metrics.csv
├── run_metadata.json
├── requirements.lock
└── figures/
```

### 4. Evaluate a CNN Model

After training, evaluate a checkpoint on the test set:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/evaluate.py \
  --checkpoint outputs/resnet50_5ep/best_model.pth \
  --model resnet50 \
  --test_csv data/FracAtlas/test.csv \
  --image_dir data/FracAtlas/images \
  --batch_size 16 \
  --num_workers 0 \
  --output_dir outputs/resnet50_5ep/evaluation
```

The evaluation script reports:

- Accuracy
- Precision
- Recall / sensitivity
- Specificity
- F1-score
- ROC-AUC
- True positives
- False positives
- True negatives
- False negatives

It also saves:

```text
outputs/<run_name>/evaluation/
├── predictions.csv
├── test_metrics.csv
└── figures/
    ├── confusion_matrix.png
    └── roc_curve.png
```

### 5. Generate Grad-CAM Visualizations

Grad-CAM is used to visualize which regions of the X-ray influenced the model prediction.

Example:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/gradcam.py \
  --checkpoint outputs/resnet50_5ep/best_model.pth \
  --model resnet50 \
  --test_csv data/FracAtlas/test.csv \
  --image_dir data/FracAtlas/images \
  --predictions_csv outputs/resnet50_5ep/evaluation/predictions.csv \
  --output_dir outputs/resnet50_5ep/gradcam \
  --num_examples 5
```

Grad-CAM helps check whether the model is focusing on meaningful bone regions instead of unrelated artifacts.

### 6. Run the VLM Baseline

First, create a small VLM evaluation subset from the available test images:

```bash
python src/make_vlm_subset.py \
  --input_csv data/FracAtlas/test_available.csv \
  --output_csv data/FracAtlas/test_vlm_balanced_40.csv \
  --per_class 20
```

Because the repository image subset may contain only a small number of fracture-positive test images, the resulting file may be smaller than 40 images.

Run a real OpenAI VLM baseline:

```bash
python src/vlm_baseline.py \
  --provider openai \
  --model gpt-4o-mini \
  --test_csv data/FracAtlas/test_vlm_balanced_40.csv \
  --image_dir data/FracAtlas/images \
  --prompt_mode simple \
  --sleep_sec 1 \
  --output_csv outputs/vlm/openai_simple_available24_fixed.csv
```

Evaluate the VLM predictions:

```bash
python src/evaluate_vlm.py \
  --predictions_csv outputs/vlm/openai_simple_available24_fixed.csv \
  --output_dir outputs/vlm/openai_simple_available24_fixed_eval \
  --title "OpenAI VLM Simple Available 24"
```

Other prompt modes can be tested with:

```text
simple
conservative
sensitive
```

Example command for conservative prompting:

```bash
python src/vlm_baseline.py \
  --provider openai \
  --model gpt-4o-mini \
  --test_csv data/FracAtlas/test_vlm_balanced_40.csv \
  --image_dir data/FracAtlas/images \
  --prompt_mode conservative \
  --sleep_sec 1 \
  --output_csv outputs/vlm/openai_conservative_available24.csv
```

Example command for sensitive prompting:

```bash
python src/vlm_baseline.py \
  --provider openai \
  --model gpt-4o-mini \
  --test_csv data/FracAtlas/test_vlm_balanced_40.csv \
  --image_dir data/FracAtlas/images \
  --prompt_mode sensitive \
  --sleep_sec 1 \
  --output_csv outputs/vlm/openai_sensitive_available24.csv
```

Create a compact VLM prompt summary:

```bash
cat > outputs/vlm/vlm_prompt_summary.csv <<'CSV'
model,prompt_mode,n_total_rows,n_valid_predictions,n_failed_predictions,accuracy,precision,recall_sensitivity,specificity,f1_score,roc_auc,true_positives,false_positives,true_negatives,false_negatives
CSV

tail -n +2 outputs/vlm/openai_simple_available24_fixed_eval/test_metrics.csv >> outputs/vlm/vlm_prompt_summary.csv
tail -n +2 outputs/vlm/openai_conservative_available24_eval/test_metrics.csv >> outputs/vlm/vlm_prompt_summary.csv
tail -n +2 outputs/vlm/openai_sensitive_available24_eval/test_metrics.csv >> outputs/vlm/vlm_prompt_summary.csv

cat outputs/vlm/vlm_prompt_summary.csv
```

### 7. Run Journal-Strengthening Analyses

After CNN checkpoints and prediction CSVs are available, run:

```bash
bash experiments/journal_strengthening/run_all_strengthening.sh
```

This safe runner performs:

1. Repository audit
2. CNN validation/test prediction normalization or generation
3. Validation-only threshold tuning
4. Bootstrap 95% confidence intervals
5. ResNet50 error and subgroup analysis
6. Paper figure generation
7. Final report generation

Important guarded steps:

- Multi-seed training is skipped unless `RUN_MULTISEED=1`.
- Paid VLM repeated trials are skipped unless `RUN_VLM=1`.
- External validation is skipped unless external data and validated labels are present.

Main outputs:

```text
experiments/journal_strengthening/results/
experiments/journal_strengthening/tables/
experiments/journal_strengthening/figures/
experiments/journal_strengthening/error_analysis/
experiments/journal_strengthening/JOURNAL_STRENGTHENING_REPORT.md
```

## Results

### CNN Test-Set Results

The following table summarizes the CNN results on the held-out test set.

| Model | Epochs | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 10 | 0.7765 | 0.4027 | 0.5556 | 0.8238 | 0.4669 | 0.7613 | 60 | 89 | 416 | 48 |
| MobileNetV2 | 5 | 0.8532 | 0.5662 | 0.7130 | 0.8832 | 0.6311 | 0.8709 | 77 | 59 | 446 | 31 |
| ResNet50 | 5 | 0.8581 | 0.5734 | 0.7593 | 0.8792 | 0.6534 | 0.8915 | 82 | 61 | 444 | 26 |
| DenseNet121 | 5 | 0.8450 | 0.5520 | 0.6389 | 0.8891 | 0.5923 | 0.8558 | 69 | 56 | 449 | 39 |

### VLM Pilot Baseline Results

The original VLM pilot used the available-image subset in the GitHub repository environment. Raw VLM API output CSVs are not committed because `outputs/vlm/` is ignored. A non-sensitive full-test summary CSV is committed under `experiments/journal_strengthening/results/` for the 613-image held-out FracAtlas test split.

| Model | Prompt | n | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o-mini | simple | 24 | 0.8750 | 0.6667 | 0.5000 | 0.9500 | 0.5714 | 0.6188 | 2 | 1 | 19 | 2 |
| GPT-4o-mini | conservative | 24 | 0.8750 | 0.6667 | 0.5000 | 0.9500 | 0.5714 | 0.7250 | 2 | 1 | 19 | 2 |
| GPT-4o-mini | sensitive | 24 | 0.6667 | 0.2500 | 0.5000 | 0.7000 | 0.3333 | 0.6000 | 2 | 6 | 14 | 2 |

Full-test clean VLM summary from `experiments/journal_strengthening/results/vlm_fulltest613_clean_summary.csv`:

| Model | Prompt | n | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o-mini | simple | 613 | 0.8336 | 0.6250 | 0.1389 | 0.9822 | 0.2273 | 0.5702 | 15 | 9 | 496 | 93 |
| GPT-4o-mini | conservative | 613 | 0.8320 | 0.6923 | 0.0833 | 0.9921 | 0.1488 | 0.5386 | 9 | 4 | 501 | 99 |
| GPT-4o-mini | sensitive | 613 | 0.8059 | 0.4396 | 0.3704 | 0.8990 | 0.4020 | 0.6357 | 40 | 51 | 454 | 68 |

### Journal-Strengthening Results

Validation-only threshold tuning improved the ResNet50 operating point compared with the fixed `0.50` threshold.

| ResNet50 Threshold Rule | Threshold | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 0.50 | 0.50 | 0.8581 | 0.5734 | 0.7593 | 0.8792 | 0.6534 | 0.8915 | 82 | 61 | 444 | 26 |
| Max validation F1 | 0.57 | 0.8777 | 0.6320 | 0.7315 | 0.9089 | 0.6781 | 0.8915 | 79 | 46 | 459 | 29 |
| Max Youden J | 0.54 | 0.8744 | 0.6202 | 0.7407 | 0.9030 | 0.6751 | 0.8915 | 80 | 49 | 456 | 28 |
| High-sensitivity constrained | 0.19 | 0.6020 | 0.2964 | 0.9167 | 0.5347 | 0.4480 | 0.8915 | 99 | 235 | 270 | 9 |

Bootstrap 95% confidence intervals were computed with 2,000 test-set resamples. For ResNet50 at the validation-selected max-F1 threshold:

| Metric | Estimate [95% CI] |
|---|---:|
| Accuracy | 0.8777 [0.8499, 0.9038] |
| Precision | 0.6320 [0.5431, 0.7180] |
| Recall / Sensitivity | 0.7315 [0.6422, 0.8144] |
| Specificity | 0.9089 [0.8823, 0.9346] |
| F1-score | 0.6781 [0.6041, 0.7458] |
| ROC-AUC | 0.8915 [0.8496, 0.9264] |

Full tables are saved in:

```text
experiments/journal_strengthening/tables/threshold_tuning_table.tex
experiments/journal_strengthening/tables/bootstrap_ci_table.tex
```

## Result Interpretation

Among the CNN models, **ResNet50** achieved the best overall performance.

It had:

- Highest accuracy: `0.8581`
- Highest recall: `0.7593`
- Highest F1-score: `0.6534`
- Highest ROC-AUC: `0.8915`
- Lowest false negatives: `26`

This is important because false negatives are especially concerning in fracture detection. A false negative means the model missed an actual fracture.

The pretrained transfer learning models performed better than the custom CNN baseline. This makes sense because the dataset is relatively small compared with large-scale natural image datasets, and pretrained CNN backbones already contain useful visual features such as edges, textures, and shapes.

The VLM baseline showed that a general-purpose zero-shot VLM can produce structured fracture / no-fracture predictions, but it did not outperform the task-specific CNN models on fracture recall. In the full-test clean evaluation, the sensitive prompt improved recall relative to simple and conservative prompts, but it remained below the CNN models and increased false positives.

The journal-strengthening analysis shows that ResNet50 performance depends on the selected operating threshold. Reporting both the fixed `0.50` threshold and validation-selected thresholds gives a clearer view of the trade-off between missed fractures and false alarms.

## Why Recall Matters

In this project, recall is one of the most important metrics.

```text
Recall = TP / (TP + FN)
```

A higher recall means the model catches more actual fracture cases. In medical screening tasks, this is important because missing a fracture can be more harmful than incorrectly flagging a normal image.

False negatives in the current CNN experiments:

| Model | False Negatives |
|---|---:|
| Custom CNN | 48 |
| MobileNetV2 | 31 |
| ResNet50 | 26 |
| DenseNet121 | 39 |

ResNet50 had the lowest number of false negatives among the current CNN models.

At the ResNet50 validation-selected max-F1 threshold (`0.57`), the model reduced false positives from 61 to 46, while false negatives increased from 26 to 29. A high-sensitivity threshold (`0.19`) reduced false negatives to 9, but at the cost of 235 false positives. These trade-offs should be discussed as operating-point choices rather than as separate trained models.

## Grad-CAM Explainability

This project includes Grad-CAM support for interpretability.

Grad-CAM produces heatmaps that show which parts of the image contributed most to the model's prediction. This is useful for medical imaging because it helps answer questions such as:

- Is the model looking at the bone region?
- Is the model focusing near the possible fracture?
- Is the model relying on irrelevant artifacts?
- Why did the model miss a fracture?
- Why did the model produce a false positive?

Grad-CAM does not prove that the model is clinically reliable, but it is useful for debugging and interpretation.

## Reproducibility

The training script includes reproducibility support, including:

- Random seed setting
- Saved training arguments
- Saved dataset CSV hashes
- Saved Git commit information when available
- Saved environment information
- Saved `requirements.lock`
- Saved `run_metadata.json`
- Saved metrics CSV files
- Saved confusion matrix, ROC curve, and Grad-CAM figures

These files help track how each experiment was produced.

## Limitations

This project has several important limitations:

1. **Not clinically validated**

   The model is not approved for medical diagnosis and should not be used for real patient care.

2. **Binary classification only**

   The model only predicts fracture or no fracture. It does not classify fracture type, anatomical region, severity, or treatment urgency.

3. **No supervised localization**

   Grad-CAM provides visual explanations, but the model is still trained as an image-level classifier.

4. **Dataset limitations**

   The model may learn patterns specific to the FracAtlas dataset and may not generalize to other hospitals, imaging devices, or patient populations.

5. **Threshold tuning is validation-based, not clinical calibration**

   The journal-strengthening package tunes thresholds on the validation set, but this is not the same as clinical calibration. A real deployment would require prospective validation and clinically chosen operating points.

6. **False negatives still exist**

   Even the best current model still misses some fracture cases.

7. **VLM stability remains limited**

   Full-test VLM summary metrics are included, but raw VLM API outputs are not committed. Repeated paid API trials are guarded and have not been run unless `RUN_VLM=1` is explicitly set.

8. **VLMs are not medical diagnostic systems**

   General-purpose multimodal models may produce plausible explanations, but they are not validated for medical image diagnosis in this project.

## Future Work

Possible improvements include:

- Train all CNN models for more epochs under the same experimental setup
- Run the guarded multi-seed ResNet50 stability experiment
- Tune learning rate, batch size, optimizer, and weight decay
- Calibrate probabilities and choose operating thresholds with clinical input
- Add more model architectures such as EfficientNet or ConvNeXt
- Use ensemble models
- Add external validation using another X-ray dataset
- Use bounding box or segmentation annotations for localization-aware evaluation
- Compare Grad-CAM heatmaps with ground-truth fracture annotations
- Run repeated VLM trials across prompt modes to quantify API-output stability
- Compare multiple VLM providers and prompt strategies under the same test split
- Add a simple web demo for uploading an X-ray and viewing predictions
- Add automated experiment tracking with TensorBoard or Weights & Biases

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow
- OpenAI / Gemini / Anthropic APIs for optional VLM experiments

## References

- Abedeen, I., et al. “FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs.” *Scientific Data*, 2023.
- FracAtlas Dataset on Figshare.
- PyTorch Documentation.
- Torchvision Models Documentation.
- Selvaraju, R. R., et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.”
- OpenAI API documentation for image inputs.
- Google Gemini API documentation for image understanding.
- Anthropic Claude documentation for vision inputs.

## Disclaimer

This project is for educational and research purposes only. It is not intended for clinical diagnosis or medical decision-making. Any real-world medical application would require expert review, external validation, regulatory approval, and clinical testing.

## Author

**Kenneth / Zijian Wang**

Computer Science student interested in artificial intelligence, machine learning, medical imaging, and applied deep learning systems.

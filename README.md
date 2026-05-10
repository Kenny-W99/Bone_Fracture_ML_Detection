# Bone Fracture ML Detection

A PyTorch-based machine learning project for binary bone fracture detection using the **FracAtlas X-ray dataset**. This project compares several CNN-based models, including a custom CNN baseline and ImageNet-pretrained transfer learning models, to classify X-ray images as either **fracture** or **no fracture**.

## Project Overview

This project builds a complete machine learning pipeline for bone fracture classification from X-ray images. The goal is to evaluate whether convolutional neural networks can detect fractures from musculoskeletal radiographs.

The project includes:

- Dataset loading from CSV metadata
- Train / validation / test splitting
- X-ray image preprocessing
- Medical-image-safe data augmentation
- Multiple CNN model architectures
- Training and evaluation scripts
- Model comparison using clinical-style metrics
- Grad-CAM visualization for model interpretability

This project is designed for educational and research purposes. It is not intended for clinical diagnosis.

## Motivation

Bone fracture detection is an important medical imaging task. In real clinical settings, missing a fracture can delay treatment and potentially harm patients. Because of this, this project does not only report accuracy. It also reports recall, specificity, F1-score, ROC-AUC, and confusion matrix values.

For this task, **recall** is especially important because a false negative means the model missed an actual fracture.

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

Current split used in this project:

| Split | Number of Images |
|---|---:|
| Train | 2,858 |
| Validation | 612 |
| Test | 613 |
| Total | 4,083 |

## Task Definition

This is a binary image classification task.

Given one X-ray image, the model predicts:

| Label | Meaning |
|---:|---|
| `0` | No fracture |
| `1` | Fracture |

The model outputs one logit. During training, the project uses `BCEWithLogitsLoss`, which is appropriate for binary classification.

During inference, the output logit is converted into a probability using sigmoid:

```text
probability = sigmoid(logit)
```

A default threshold of `0.5` is used:

```text
probability >= 0.5 → fracture
probability < 0.5  → no fracture
```

## Model Architectures

This project compares four CNN-based models.

### 1. Custom CNN

A simple convolutional neural network built from scratch. It is used as a baseline model.

### 2. MobileNetV2

A lightweight ImageNet-pretrained model. It is useful for efficient inference and smaller model size.

### 3. ResNet50

A deeper residual network with skip connections. In the current experiments, ResNet50 achieved the strongest overall performance.

### 4. DenseNet121

A densely connected CNN architecture that is commonly used in medical imaging research because it can reuse features effectively.

For the pretrained models, the original ImageNet classification head is replaced with a single-output binary classification layer.

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
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── outputs/
│   ├── results_summary.csv
│   └── other training / evaluation outputs
│
├── research_context/
├── shared/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── inspect_dataset.py
│   ├── models.py
│   ├── split_dataset.py
│   └── train.py
│
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

Install dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib pillow
```

If you are using a CUDA GPU, install the correct PyTorch version from the official PyTorch website.

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

### 3. Train a Model

Example: train ResNet50.

```bash
python src/train.py \
  --model resnet50 \
  --train_csv data/FracAtlas/train.csv \
  --val_csv data/FracAtlas/val.csv \
  --test_csv data/FracAtlas/test.csv \
  --image_dir data/FracAtlas/images \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir outputs/resnet50_run \
  --use_pos_weight
```

Supported model names:

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

### 4. Evaluate a Model

After training, evaluate a checkpoint on the test set:

```bash
python src/evaluate.py \
  --checkpoint outputs/resnet50_run/best_model.pth \
  --model resnet50 \
  --test_csv data/FracAtlas/test.csv \
  --image_dir data/FracAtlas/images \
  --output_dir outputs/resnet50_run/evaluation
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
python src/gradcam.py \
  --checkpoint outputs/resnet50_run/best_model.pth \
  --model resnet50 \
  --test_csv data/FracAtlas/test.csv \
  --image_dir data/FracAtlas/images \
  --predictions_csv outputs/resnet50_run/evaluation/predictions.csv \
  --output_dir outputs/resnet50_run/gradcam \
  --num_examples 5
```

Grad-CAM helps check whether the model is focusing on meaningful bone regions instead of unrelated artifacts.

## Results

The following table summarizes the current test-set results.

| Model | Epochs | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 10 | 0.7765 | 0.4027 | 0.5556 | 0.8238 | 0.4669 | 0.7613 | 60 | 89 | 416 | 48 |
| MobileNetV2 | 5 | 0.8532 | 0.5662 | 0.7130 | 0.8832 | 0.6311 | 0.8709 | 77 | 59 | 446 | 31 |
| ResNet50 | 5 | 0.8581 | 0.5734 | 0.7593 | 0.8792 | 0.6534 | 0.8915 | 82 | 61 | 444 | 26 |
| DenseNet121 | 5 | 0.8450 | 0.5520 | 0.6389 | 0.8891 | 0.5923 | 0.8558 | 69 | 56 | 449 | 39 |

## Result Interpretation

Among the tested models, **ResNet50** achieved the best overall performance.

It had:

- Highest accuracy: `0.8581`
- Highest recall: `0.7593`
- Highest F1-score: `0.6534`
- Highest ROC-AUC: `0.8915`
- Lowest false negatives: `26`

This is important because false negatives are especially concerning in fracture detection. A false negative means the model missed an actual fracture.

The pretrained transfer learning models performed better than the custom CNN baseline. This makes sense because the dataset is relatively small compared with large-scale natural image datasets, and pretrained CNN backbones already contain useful visual features such as edges, textures, and shapes.

## Why Recall Matters

In this project, recall is one of the most important metrics.

```text
Recall = TP / (TP + FN)
```

A higher recall means the model catches more actual fracture cases. In medical screening tasks, this is important because missing a fracture can be more harmful than incorrectly flagging a normal image.

False negatives in the current experiments:

| Model | False Negatives |
|---|---:|
| Custom CNN | 48 |
| MobileNetV2 | 31 |
| ResNet50 | 26 |
| DenseNet121 | 39 |

ResNet50 had the lowest number of false negatives among the current models.

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

5. **Threshold not optimized**

   The current prediction threshold is `0.5`. In a medical screening setting, threshold tuning may be needed to reduce false negatives.

6. **False negatives still exist**

   Even the best current model still misses some fracture cases.

## Future Work

Possible improvements include:

- Train all models for more epochs under the same experimental setup
- Tune learning rate, batch size, optimizer, and weight decay
- Optimize the classification threshold to reduce false negatives
- Add more model architectures such as EfficientNet or ConvNeXt
- Use ensemble models
- Add external validation using another X-ray dataset
- Use bounding box or segmentation annotations for localization-aware evaluation
- Compare Grad-CAM heatmaps with ground-truth fracture annotations
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

## References

- Abedeen, I., et al. “FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs.” *Scientific Data*, 2023.
- FracAtlas Dataset on Figshare.
- PyTorch Documentation.
- Torchvision Models Documentation.
- Selvaraju, R. R., et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.”

## Disclaimer

This project is for educational and research purposes only. It is not intended for clinical diagnosis or medical decision-making. Any real-world medical application would require expert review, external validation, regulatory approval, and clinical testing.

## Author

**Kenneth / Zijian Wang**

Computer Science student interested in artificial intelligence, machine learning, medical imaging, and applied deep learning systems.

Note: The full CSV metadata contains 4,083 image records, but the repository may only include a subset of image files due to repository size limits. For a quick runnable demo, use `dataset_available.csv`, `train_available.csv`, `val_available.csv`, and `test_available.csv`, which are generated from the images currently available in `data/FracAtlas/images/`. To reproduce full training, download the complete FracAtlas image set and place all images under `data/FracAtlas/images/`.
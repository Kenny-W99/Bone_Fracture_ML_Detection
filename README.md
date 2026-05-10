# Bone Fracture ML Detection

A PyTorch-based machine learning project for binary bone fracture detection using the **FracAtlas X-ray dataset**. This project compares multiple convolutional neural network architectures, including a custom CNN baseline and ImageNet-pretrained transfer learning models, to classify musculoskeletal X-ray images as **fracture** or **no fracture**.

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Task Definition](#task-definition)
- [Model Architectures](#model-architectures)
- [Preprocessing and Augmentation](#preprocessing-and-augmentation)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [1. Inspect the Dataset](#1-inspect-the-dataset)
  - [2. Split the Dataset](#2-split-the-dataset)
  - [3. Train a Model](#3-train-a-model)
  - [4. Evaluate a Trained Model](#4-evaluate-a-trained-model)
  - [5. Generate Grad-CAM Visualizations](#5-generate-grad-cam-visualizations)
- [Results](#results)
- [Interpretation of Results](#interpretation-of-results)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Technologies Used](#technologies-used)
- [References](#references)
- [Disclaimer](#disclaimer)
- [Author](#author)

## Project Overview

This project develops and evaluates machine learning models for **binary bone fracture classification** from X-ray images. The goal is to determine whether an input musculoskeletal radiograph contains a fracture.

The project includes a complete PyTorch pipeline:

- Dataset loading from CSV metadata
- Train / validation / test splitting
- Medical-image-aware preprocessing
- Data augmentation
- Training with class imbalance handling
- Evaluation with clinically meaningful metrics
- Comparison of multiple CNN architectures
- Grad-CAM visualization for interpretability

The main models compared in this project are:

1. Custom CNN
2. MobileNetV2
3. ResNet50
4. DenseNet121

Among the current experiments, **ResNet50** achieved the strongest overall performance based on F1-score, recall, and ROC-AUC.

## Motivation

Bone fracture detection is an important medical imaging task. In real clinical settings, missing a fracture can delay treatment and lead to further injury, chronic pain, or long-term complications. Although this project is not intended for clinical use, it explores how deep learning can assist fracture classification from X-ray images.

For this task, **recall**, also known as sensitivity, is especially important. A false negative means the model predicts “no fracture” when a fracture is actually present. In medical imaging, false negatives are often more dangerous than false positives because a missed diagnosis can affect patient care.

Therefore, this project reports not only accuracy, but also:

- Precision
- Recall / sensitivity
- Specificity
- F1-score
- ROC-AUC
- Confusion matrix values

## Dataset

This project uses the **FracAtlas** dataset, a musculoskeletal radiograph dataset for fracture classification, localization, and segmentation.

The dataset is stored under:

```text
data/FracAtlas/
├── images/
├── dataset.csv
├── train.csv
├── val.csv
└── test.csv
```

The main CSV columns used in this project are:

| Column | Description |
|---|---|
| `image_id` | Filename of the X-ray image |
| `fractured` | Binary label: `1` for fracture, `0` for no fracture |

Current dataset split:

| Split | Number of Images |
|---|---:|
| Train | 2,858 |
| Validation | 612 |
| Test | 613 |
| Total | 4,083 |

## Task Definition

This is a **binary image classification** task.

Given an X-ray image, the model predicts:

| Label | Meaning |
|---:|---|
| `0` | No fracture |
| `1` | Fracture |

The model outputs a single logit. During training, the project uses `BCEWithLogitsLoss`, which combines a sigmoid layer and binary cross-entropy loss in a numerically stable way.

During inference, the output logit is converted to a probability using the sigmoid function:

```text
probability = sigmoid(logit)
```

A default threshold of `0.5` is used:

```text
probability >= 0.5 → fracture
probability < 0.5  → no fracture
```

## Model Architectures

This project compares four CNN-based architectures.

### 1. Custom CNN

The custom CNN is a simple baseline model built from scratch. It is intentionally lightweight and serves as a lower-bound comparison against pretrained transfer learning models.

General structure:

```text
Conv2D → ReLU → MaxPool
Conv2D → ReLU → MaxPool
Conv2D → ReLU → MaxPool
Conv2D → ReLU
Adaptive Average Pooling
Dropout
Linear output layer
```

### 2. MobileNetV2

MobileNetV2 is a lightweight convolutional neural network designed for efficient inference. It is useful when model size and speed matter.

In this project, the ImageNet-pretrained MobileNetV2 classification head is replaced with a binary output layer.

### 3. ResNet50

ResNet50 is a deep residual network that uses skip connections to make training deeper networks easier. It is a strong general-purpose computer vision backbone and performed best in the current experiments.

The original ImageNet classification layer is replaced with a single-output binary classification head.

### 4. DenseNet121

DenseNet121 uses dense connections between layers, allowing feature reuse throughout the network. DenseNet-style architectures are commonly used in medical imaging research because they can learn useful features from limited data.

The original classifier is replaced with a binary classification output layer.

## Preprocessing and Augmentation

Medical images require careful preprocessing because unrealistic transformations can damage important anatomical information.

### Validation and Test Preprocessing

For validation and test images, the pipeline is deterministic:

1. Load image using PIL
2. Convert image to RGB
3. Pad image to a square shape while preserving aspect ratio
4. Resize to `224 x 224`
5. Convert to PyTorch tensor
6. Normalize using ImageNet mean and standard deviation

The aspect-ratio-preserving step is important because directly resizing rectangular X-ray images to a square can stretch or distort bone structures.

### Training Augmentation

For training images, the project applies realistic augmentations:

- Small random rotation
- Small affine translation
- Brightness and contrast jitter
- Optional horizontal flip
- ImageNet normalization

The project intentionally avoids:

- Vertical flipping
- Random resized cropping
- Heavy geometric distortion

These transformations are avoided because medical X-rays should remain anatomically realistic, and cropping may remove the fracture region.

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
│   ├── custom_cnn_test/
│   ├── mobilenet_v2_5ep/
│   ├── resnet50_repro_5ep_eval.log
│   ├── resnet50_repro_5ep_train.log
│   └── results_summary.csv
│
├── research_context/
│
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

Activate the environment:

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

If using a CUDA GPU, install the correct PyTorch version from the official PyTorch installation page:

```text
https://pytorch.org/get-started/locally/
```

## How to Run

### 1. Inspect the Dataset

Before training, inspect the dataset structure and label distribution:

```bash
python src/inspect_dataset.py \
  --csv_path data/FracAtlas/dataset.csv \
  --image_dir data/FracAtlas/images
```

This helps confirm that:

- The CSV file can be loaded
- Image paths are correct
- Labels are available
- The class distribution is reasonable

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

Useful training options:

| Argument | Description |
|---|---|
| `--model` | Model architecture |
| `--train_csv` | Path to training CSV |
| `--val_csv` | Path to validation CSV |
| `--test_csv` | Path to test CSV |
| `--image_dir` | Path to image folder |
| `--epochs` | Number of training epochs |
| `--batch_size` | Batch size |
| `--lr` | Learning rate |
| `--output_dir` | Folder for checkpoints, metrics, and figures |
| `--use_pos_weight` | Uses positive class weighting for class imbalance |
| `--seed` | Random seed for reproducibility |

The training script saves:

```text
outputs/<run_name>/
├── best_model.pth
├── metrics.csv
├── run_metadata.json
├── requirements.lock
└── figures/
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── precision_curve.png
    ├── recall_curve.png
    ├── f1_curve.png
    ├── specificity_curve.png
    └── roc_auc_curve.png
```

### 4. Evaluate a Trained Model

Example: evaluate a trained ResNet50 checkpoint.

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

Grad-CAM helps visualize which regions of the X-ray influenced the model prediction.

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

Grad-CAM outputs are useful for checking whether the model focuses on meaningful bone regions instead of unrelated image artifacts.

## Results

The following table summarizes the current test-set performance.

| Model | Epochs | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 10 | 0.7765 | 0.4027 | 0.5556 | 0.8238 | 0.4669 | 0.7613 | 60 | 89 | 416 | 48 |
| MobileNetV2 | 5 | 0.8532 | 0.5662 | 0.7130 | 0.8832 | 0.6311 | 0.8709 | 77 | 59 | 446 | 31 |
| ResNet50 | 5 | 0.8581 | 0.5734 | 0.7593 | 0.8792 | 0.6534 | 0.8915 | 82 | 61 | 444 | 26 |
| DenseNet121 | 5 | 0.8450 | 0.5520 | 0.6389 | 0.8891 | 0.5923 | 0.8558 | 69 | 56 | 449 | 39 |

## Interpretation of Results

### Best Overall Model

Based on the current experiments, **ResNet50** achieved the strongest overall result:

- Highest accuracy: `0.8581`
- Highest recall: `0.7593`
- Highest F1-score: `0.6534`
- Highest ROC-AUC: `0.8915`
- Lowest number of false negatives: `26`

This suggests that ResNet50 was the best model in the current comparison for identifying fracture cases while maintaining reasonable performance on non-fracture images.

### Why Recall Matters

In fracture detection, recall is especially important because it measures how many actual fractures the model successfully detects.

```text
Recall = TP / (TP + FN)
```

A false negative means the model missed a fracture. In a medical setting, this could be more serious than a false positive because a missed fracture may delay treatment.

In this project, ResNet50 had the fewest false negatives among the tested models:

| Model | False Negatives |
|---|---:|
| Custom CNN | 48 |
| MobileNetV2 | 31 |
| ResNet50 | 26 |
| DenseNet121 | 39 |

### Transfer Learning vs. Custom CNN

The pretrained transfer learning models performed better than the custom CNN baseline. This is expected because medical datasets are often relatively small, and pretrained CNNs already contain useful low-level visual features such as edges, textures, and shapes.

The custom CNN still provides a useful baseline because it shows how much performance improves when using stronger pretrained backbones.

## Grad-CAM Explainability

This project includes Grad-CAM support to improve model interpretability.

Grad-CAM produces a heatmap that highlights image regions that contributed most strongly to the model's prediction. For fracture detection, this is useful because it helps answer questions such as:

- Is the model focusing on the bone region?
- Is the model looking near the fracture line?
- Is the model relying on irrelevant artifacts?
- Why did the model miss a fracture?
- Why did the model incorrectly flag a normal image?

Grad-CAM is not a replacement for clinical validation, but it is a useful debugging and interpretation tool for medical imaging models.

## Reproducibility

The training script includes reproducibility support, including:

- Random seed setting
- PyTorch deterministic settings where available
- Saved training arguments
- Saved Git commit information when available
- Saved dataset CSV hashes
- Saved `requirements.lock`
- Saved run metadata in `run_metadata.json`

Example output files:

```text
run_metadata.json
requirements.lock
metrics.csv
best_model.pth
```

This makes it easier to track exactly how a model was trained and evaluated.

## Limitations

This project is for educational and research purposes only. It is not a clinical diagnostic system.

Current limitations include:

1. **No clinical validation**

   The models were not validated in a real hospital environment and should not be used for patient diagnosis.

2. **Limited dataset size**

   Although FracAtlas is useful for research, real clinical systems require much larger and more diverse datasets.

3. **Binary classification only**

   The current task only predicts fracture or no fracture. It does not classify fracture type, severity, anatomical region, or treatment urgency.

4. **No direct fracture localization**

   The model performs image-level classification. Grad-CAM provides visual explanation, but it is not the same as a supervised localization model.

5. **Potential dataset bias**

   The model may learn patterns specific to the dataset, imaging source, annotation style, or preprocessing pipeline.

6. **Threshold not optimized**

   The default classification threshold is `0.5`. For medical screening, a lower threshold may improve recall and reduce missed fractures.

7. **False negatives still exist**

   Even the best current model still misses some fracture cases, which would be unacceptable in real clinical deployment without further validation.

## Future Work

Potential improvements include:

- Train all models for more epochs under the same experimental setup
- Tune learning rate, batch size, optimizer, and weight decay
- Optimize classification threshold to reduce false negatives
- Use stronger data augmentation while preserving medical realism
- Add EfficientNet, ConvNeXt, or Vision Transformer baselines
- Use ensemble models to improve robustness
- Add external validation using a separate X-ray dataset
- Use bounding box or segmentation annotations for localization-aware evaluation
- Compare Grad-CAM heatmaps with ground-truth fracture annotations
- Build a simple web demo for image upload and prediction visualization
- Add automated experiment tracking with tools such as Weights & Biases or TensorBoard
- Add unit tests for dataset loading, preprocessing, and model output shapes

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
  https://www.nature.com/articles/s41597-023-02432-4

- FracAtlas Dataset on Figshare.  
  https://figshare.com/articles/dataset/The_dataset/22363012

- PyTorch Documentation.  
  https://pytorch.org/docs/stable/index.html

- Torchvision Models Documentation.  
  https://pytorch.org/vision/stable/models.html

- Selvaraju, R. R., et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.”  
  https://arxiv.org/abs/1610.02391

## Disclaimer

This project is not intended for clinical diagnosis or medical decision-making. The predictions, evaluation metrics, and Grad-CAM visualizations are for educational and research exploration only. Any real-world medical application would require expert review, external validation, regulatory approval, and clinical testing.

## Author

**Kenneth / Zijian Wang**

Computer Science student interested in artificial intelligence, machine learning, medical imaging, and applied deep learning systems.
"""
FracAtlas — Grad-CAM Explainability Script
============================================

Generates Gradient-weighted Class Activation Mapping (Grad-CAM) visualizations
for a trained bone fracture classification model.

What is Grad-CAM?
-----------------
Grad-CAM answers the question: "WHERE in the image is the model looking when
it makes its prediction?" It produces a heatmap highlighting the regions that
most influenced the model's decision.

How it works (simplified):
  1. Run a forward pass through the model to get the prediction.
  2. Compute gradients of the output with respect to the feature maps in the
     LAST convolutional layer (e.g., ResNet50's layer4).
  3. Global-average-pool those gradients to get per-channel importance weights.
  4. Multiply each feature map channel by its weight and sum them up.
  5. Apply ReLU (we only care about features that INCREASE the predicted class).
  6. Resize the heatmap back to the original image size and overlay it.

Why this matters for medical imaging:
-------------------------------------
  • TRUST: Radiologists won't use a "black box." Grad-CAM shows them the model
    is looking at the fracture line, not an artifact (e.g., a metal marker).
  • DEBUGGING: If the model is right for the wrong reason (e.g., focusing on
    text annotations rather than bone), Grad-CAM reveals this.
  • FALSE NEGATIVES: For missed fractures, Grad-CAM shows where the model WAS
    looking — helping researchers understand why it failed.
  • VALIDATION: If Grad-CAM highlights the fracture region (matching bounding
    box annotations), we gain confidence the model learned meaningful features.

Usage:
------
    python src/gradcam.py \\
        --checkpoint outputs/resnet50_5ep/best_model.pth \\
        --model resnet50 \\
        --test_csv data/FracAtlas/test.csv \\
        --image_dir data/FracAtlas/images \\
        --predictions_csv outputs/resnet50_5ep/evaluation/predictions.csv \\
        --output_dir outputs/resnet50_5ep/gradcam \\
        --num_examples 5

Author : FracAtlas Project
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Our project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models import get_model


# =============================================================================
# DEVICE SELECTION (supports Apple Silicon MPS, CUDA, and CPU)
# =============================================================================

def get_device() -> torch.device:
    """
    Select the best available device.

    Priority: CUDA GPU → Apple Silicon MPS → CPU

    Apple Silicon (M1/M2/M3) Macs have a GPU accessible via MPS (Metal
    Performance Shaders). This is much faster than CPU for inference.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU")
    return device


# =============================================================================
# GRAD-CAM IMPLEMENTATION
# =============================================================================

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.

    This class hooks into a specific convolutional layer of the model to
    capture its feature maps (activations) and gradients during a forward
    and backward pass. These are then combined to produce a heatmap.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model (must be in eval mode).
    target_layer : torch.nn.Module
        The convolutional layer to visualize (e.g., model.layer4 for ResNet50).

    How the hooks work:
    -------------------
    PyTorch "hooks" are callbacks that fire during forward/backward passes.
    - Forward hook: captures the OUTPUT of target_layer (the feature maps).
    - Backward hook: captures the GRADIENTS flowing back through target_layer.
    We register both hooks so we can access these values after a forward+backward.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: save the feature maps produced by the target layer."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save the gradients flowing through the target layer."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input image.

        Steps:
          1. Forward pass → get model output (logit) and capture activations.
          2. Backward pass → compute gradients of the logit w.r.t. activations.
          3. Global average pool the gradients → per-channel importance weights.
          4. Weighted sum of activation channels → raw heatmap.
          5. ReLU → only keep positive contributions.
          6. Normalize to [0, 1].

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed image tensor, shape (1, 3, 224, 224).

        Returns
        -------
        np.ndarray
            Heatmap of shape (H, W) with values in [0, 1].
            H, W match the spatial dimensions of the target layer's output
            (e.g., 7×7 for ResNet50's layer4).
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)  # shape: (1, 1)

        # For binary classification with BCEWithLogitsLoss, the single output
        # logit represents the "fracture" class. A higher logit = more confident
        # it's a fracture. We backprop from this logit.
        logit = output.squeeze()

        # Backward pass: compute gradients
        self.model.zero_grad()
        logit.backward()

        # Get captured activations and gradients
        # activations shape: (1, C, H, W) — e.g., (1, 2048, 7, 7) for ResNet50
        # gradients shape:   (1, C, H, W)
        activations = self.activations.squeeze(0)  # (C, H, W)
        gradients = self.gradients.squeeze(0)      # (C, H, W)

        # Global average pooling of gradients → channel importance weights
        # For each channel c: weight_c = mean(gradients[c, :, :])
        weights = gradients.mean(dim=(1, 2))  # shape: (C,)

        # Weighted combination of activation maps
        # cam = sum_c(weight_c * activation_c)
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU: only keep features that positively contribute to the prediction
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        """Remove the registered hooks (cleanup)."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# =============================================================================
# TARGET LAYER SELECTION
# =============================================================================

def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """
    Get the appropriate target convolutional layer for Grad-CAM.

    For each architecture, we select the LAST convolutional block because:
      • It has the richest semantic features (high-level concepts like "bone edge")
      • It still has spatial resolution (7×7 for most models at 224 input)
      • Earlier layers capture low-level features (edges, textures) which are
        less interpretable for understanding "why fracture?"

    Parameters
    ----------
    model : nn.Module
        The loaded model.
    model_name : str
        One of: "custom_cnn", "mobilenet_v2", "resnet50", "densenet121"

    Returns
    -------
    nn.Module
        The target layer for Grad-CAM hook registration.
    """
    model_name = model_name.lower().strip()

    if model_name == "resnet50":
        # ResNet50's last conv block: model.layer4
        # Output shape: (batch, 2048, 7, 7) for 224×224 input
        return model.layer4

    elif model_name == "densenet121":
        # DenseNet121's last dense block + final batch norm
        # model.features contains all conv layers
        return model.features[-1]  # Last layer in features (norm5)

    elif model_name == "mobilenet_v2":
        # MobileNetV2's last conv block in features
        # model.features is a Sequential of InvertedResidual blocks
        return model.features[-1]  # Last block

    elif model_name == "custom_cnn":
        # Our CustomCNN's last conv block (before AdaptiveAvgPool)
        # model.features is Sequential, last conv is at index -2 (before pool)
        # We target the last BatchNorm+ReLU before the pool
        return model.features[-2]  # The AdaptiveAvgPool2d is [-1]

    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def load_original_image(image_path: str) -> np.ndarray:
    """Load the original image (before preprocessing) for visualization."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_img : np.ndarray
        Original image, shape (H, W, 3), values in [0, 255].
    heatmap : np.ndarray
        Grad-CAM heatmap, shape (h, w), values in [0, 1].
        Will be resized to match original_img.
    alpha : float
        Blending factor (0 = only image, 1 = only heatmap).

    Returns
    -------
    np.ndarray
        Blended image with heatmap overlay, shape (H, W, 3), values in [0, 255].
    """
    # Resize heatmap to match original image dimensions
    h, w = original_img.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    # Apply colormap (jet: blue=cold/low, red=hot/high)
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend: overlay = alpha * heatmap + (1-alpha) * original
    original_float = original_img.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    blended = (alpha * heatmap_float + (1 - alpha) * original_float)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def save_gradcam_figure(original_img: np.ndarray, heatmap: np.ndarray,
                        overlay: np.ndarray, filename: str, true_label: int,
                        pred_label: int, pred_prob: float, save_path: str):
    """
    Save a 3-panel figure: Original | Heatmap | Overlay.

    Parameters
    ----------
    original_img : Original X-ray image
    heatmap : Raw Grad-CAM heatmap
    overlay : Blended heatmap + original
    filename : Image filename for title
    true_label : Ground truth (0 or 1)
    pred_label : Model prediction (0 or 1)
    pred_prob : Predicted probability of fracture
    save_path : Where to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original X-ray", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Heatmap only (resized to image dimensions)
    h, w = original_img.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    # Determine category and color
    if true_label == 1 and pred_label == 1:
        category = "TRUE POSITIVE (fracture correctly detected)"
        color = "green"
    elif true_label == 0 and pred_label == 0:
        category = "TRUE NEGATIVE (normal correctly identified)"
        color = "green"
    elif true_label == 0 and pred_label == 1:
        category = "FALSE POSITIVE (false alarm)"
        color = "orange"
    else:  # true_label == 1 and pred_label == 0
        category = "FALSE NEGATIVE (MISSED FRACTURE!)"
        color = "red"

    # Super title with prediction info
    true_str = "Fracture" if true_label == 1 else "No Fracture"
    pred_str = "Fracture" if pred_label == 1 else "No Fracture"

    fig.suptitle(
        f"{filename}\n"
        f"True: {true_str} | Predicted: {pred_str} (prob={pred_prob:.3f})\n"
        f"{category}",
        fontsize=12, fontweight="bold", color=color, y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()


# =============================================================================
# EXAMPLE SELECTION FROM PREDICTIONS CSV
# =============================================================================

def select_examples(predictions_csv: str, num_examples: int = 5) -> dict:
    """
    Select example images from each confusion matrix quadrant.

    Reads the predictions CSV (from evaluate.py) and picks examples from:
      • True Positives  (TP): correctly detected fractures
      • True Negatives  (TN): correctly identified normals
      • False Positives (FP): false alarms (model said fracture, actually normal)
      • False Negatives (FN): MISSED fractures (most dangerous!)

    For TP and FP: selects highest-confidence predictions (most "sure" fracture).
    For FN: selects all available (we want to understand every miss).
    For TN: selects random sample.

    Parameters
    ----------
    predictions_csv : str
        Path to predictions.csv from evaluate.py.
    num_examples : int
        Maximum number of examples per category.

    Returns
    -------
    dict
        {"TP": [...], "TN": [...], "FP": [...], "FN": [...]}
        Each value is a list of dicts with keys: filename, true_label,
        predicted_label, predicted_probability.
    """
    df = pd.read_csv(predictions_csv)

    # Categorize each prediction
    df["category"] = "?"
    df.loc[(df["true_label"] == 1) & (df["predicted_label"] == 1), "category"] = "TP"
    df.loc[(df["true_label"] == 0) & (df["predicted_label"] == 0), "category"] = "TN"
    df.loc[(df["true_label"] == 0) & (df["predicted_label"] == 1), "category"] = "FP"
    df.loc[(df["true_label"] == 1) & (df["predicted_label"] == 0), "category"] = "FN"

    selected = {}
    for cat in ["TP", "TN", "FP", "FN"]:
        subset = df[df["category"] == cat]

        if len(subset) == 0:
            selected[cat] = []
            continue

        if cat == "TP":
            # Most confident true positives (highest probability)
            subset = subset.sort_values("predicted_probability", ascending=False)
        elif cat == "FP":
            # Most confident false positives (highest probability — worst mistakes)
            subset = subset.sort_values("predicted_probability", ascending=False)
        elif cat == "FN":
            # All false negatives if possible (missed fractures are critical!)
            # Sort by lowest probability (model was most wrong)
            subset = subset.sort_values("predicted_probability", ascending=True)
        else:  # TN
            # Random sample of true negatives
            subset = subset.sample(frac=1, random_state=42)

        # Take up to num_examples
        subset = subset.head(num_examples)
        selected[cat] = subset.to_dict("records")

    return selected


# =============================================================================
# MAIN GRAD-CAM GENERATION
# =============================================================================

def generate_gradcam(args):
    """
    Main function: generate Grad-CAM visualizations for selected examples.

    Steps:
      1. Load model from checkpoint
      2. Select examples from predictions CSV
      3. For each example: preprocess → forward+backward → generate heatmap → save
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GRAD-CAM VISUALIZATION")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # Device selection
    # -------------------------------------------------------------------------
    device = get_device()

    # -------------------------------------------------------------------------
    # Load model from checkpoint
    # -------------------------------------------------------------------------
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_name = checkpoint.get("model_name", args.model)
    print(f"  Model: {model_name}")

    model = get_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Set up Grad-CAM with the appropriate target layer
    # -------------------------------------------------------------------------
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)
    print(f"  Target layer: {target_layer.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Get preprocessing transform (same as evaluation — deterministic)
    # -------------------------------------------------------------------------
    transform = get_val_transforms()

    # -------------------------------------------------------------------------
    # Select examples from predictions CSV
    # -------------------------------------------------------------------------
    print(f"\n  Selecting examples from: {args.predictions_csv}")
    examples = select_examples(args.predictions_csv, num_examples=args.num_examples)

    for cat, items in examples.items():
        print(f"    {cat}: {len(items)} examples")

    # -------------------------------------------------------------------------
    # Generate Grad-CAM for each selected example
    # -------------------------------------------------------------------------
    total_generated = 0

    for category, items in examples.items():
        if not items:
            print(f"\n  ⚠️  No {category} examples available — skipping.")
            continue

        # Create subdirectory for this category
        cat_dir = output_dir / category.lower()
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Generating Grad-CAM for {category} ({len(items)} images)...")

        for idx, item in enumerate(items):
            filename = item["filename"]
            true_label = int(item["true_label"])
            pred_label = int(item["predicted_label"])
            pred_prob = float(item["predicted_probability"])

            # Full path to the image
            img_path = os.path.join(args.image_dir, filename)

            if not os.path.isfile(img_path):
                print(f"    ⚠️  Image not found: {img_path} — skipping.")
                continue

            # Load original image (for visualization)
            original_img = load_original_image(img_path)

            # Preprocess image (same pipeline as evaluation)
            pil_img = Image.open(img_path).convert("RGB")
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Enable gradients for this input (needed for backward pass)
            input_tensor.requires_grad_(True)

            # Generate Grad-CAM heatmap
            heatmap = grad_cam.generate(input_tensor)

            # Create overlay
            overlay = overlay_heatmap(original_img, heatmap, alpha=0.4)

            # Save figure
            safe_filename = Path(filename).stem  # Remove extension
            save_path = str(cat_dir / f"{category.lower()}_{idx+1:02d}_{safe_filename}.png")
            save_gradcam_figure(
                original_img=original_img,
                heatmap=heatmap,
                overlay=overlay,
                filename=filename,
                true_label=true_label,
                pred_label=pred_label,
                pred_prob=pred_prob,
                save_path=save_path,
            )
            total_generated += 1

    # -------------------------------------------------------------------------
    # Cleanup and summary
    # -------------------------------------------------------------------------
    grad_cam.remove_hooks()

    print(f"\n{'='*60}")
    print(f"  GRAD-CAM COMPLETE")
    print(f"{'='*60}")
    print(f"  Total visualizations: {total_generated}")
    print(f"  Output directory: {output_dir}")
    print(f"\n  Directory structure:")
    print(f"    {output_dir}/")
    for cat in ["tp", "tn", "fp", "fn"]:
        cat_path = output_dir / cat
        if cat_path.exists():
            n_files = len(list(cat_path.glob("*.png")))
            print(f"    ├── {cat}/  ({n_files} images)")
    print()

    # -------------------------------------------------------------------------
    # Clinical interpretation guide
    # -------------------------------------------------------------------------
    print("  📋 HOW TO INTERPRET GRAD-CAM RESULTS:")
    print("  " + "-" * 56)
    print("  • TRUE POSITIVES: The heatmap should highlight the fracture")
    print("    line or region. If it does → model learned correctly.")
    print("  • TRUE NEGATIVES: Heatmap should be diffuse/unfocused or")
    print("    highlight normal bone structure.")
    print("  • FALSE POSITIVES: Check what the model focused on — it may")
    print("    be confused by artifacts, text, or unusual bone shapes.")
    print("  • FALSE NEGATIVES: Most critical! The heatmap shows where the")
    print("    model WAS looking when it missed the fracture. This helps")
    print("    identify if the fracture is subtle or in an unusual location.")
    print()


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for a trained FracAtlas model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Grad-CAM for ResNet50 (best model):
  python src/gradcam.py \\
      --checkpoint outputs/resnet50_5ep/best_model.pth \\
      --model resnet50 \\
      --test_csv data/FracAtlas/test.csv \\
      --image_dir data/FracAtlas/images \\
      --predictions_csv outputs/resnet50_5ep/evaluation/predictions.csv \\
      --output_dir outputs/resnet50_5ep/gradcam \\
      --num_examples 5

  # Generate for DenseNet121:
  python src/gradcam.py \\
      --checkpoint outputs/densenet121/best_model.pth \\
      --model densenet121 \\
      --predictions_csv outputs/densenet121/evaluation/predictions.csv \\
      --test_csv data/FracAtlas/test.csv \\
      --image_dir data/FracAtlas/images \\
      --output_dir outputs/densenet121/gradcam
        """,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--model", type=str, default="resnet50",
        choices=["custom_cnn", "mobilenet_v2", "resnet50", "densenet121"],
        help="Model architecture (default: resnet50). Overridden by checkpoint if available.",
    )
    parser.add_argument(
        "--test_csv", type=str, required=True,
        help="Path to test CSV (used as fallback if predictions_csv not provided)",
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Path to image directory",
    )
    parser.add_argument(
        "--predictions_csv", type=str, required=True,
        help="Path to predictions.csv from evaluate.py",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/gradcam",
        help="Directory to save Grad-CAM visualizations (default: outputs/gradcam)",
    )
    parser.add_argument(
        "--num_examples", type=int, default=5,
        help="Number of examples per category (TP/TN/FP/FN) (default: 5)",
    )

    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    generate_gradcam(args)

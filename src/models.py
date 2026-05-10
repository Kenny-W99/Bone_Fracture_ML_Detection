"""
FracAtlas — Model Definitions
==============================

This module defines all CNN architectures for binary fracture classification:

  1. CustomCNN        — A simple baseline CNN built from scratch (no pretrained weights).
  2. MobileNetV2      — Lightweight pretrained model, fast inference.
  3. ResNet50         — Deep residual network, strong general-purpose backbone.
  4. DenseNet121      — Dense connections for better feature reuse.

Each model outputs a SINGLE LOGIT (not a probability). We use BCEWithLogitsLoss
during training, which internally applies sigmoid. This is numerically more
stable than applying sigmoid yourself + using BCELoss.

Why transfer learning?
----------------------
Medical datasets like FracAtlas are small (thousands, not millions of images).
Training a deep network from scratch on so few images leads to severe overfitting.
Instead, we take a model pretrained on ImageNet (1.2M natural images) and replace
only the final classification layer. The pretrained layers already know how to
extract edges, textures, and shapes — we just teach the new head to distinguish
"fracture" from "no fracture."

Author : FracAtlas Project
"""

import torch
import torch.nn as nn
import torchvision.models as models


# =============================================================================
# 1. CUSTOM CNN BASELINE
# =============================================================================

class CustomCNN(nn.Module):
    """
    A simple CNN baseline for binary classification.

    Architecture:
        Conv(3→32) → ReLU → MaxPool
        Conv(32→64) → ReLU → MaxPool
        Conv(64→128) → ReLU → MaxPool
        Conv(128→256) → ReLU → AdaptiveAvgPool
        Flatten → Dropout → Linear(256→1)

    This is intentionally simple — it serves as a LOWER BOUND on performance.
    If a pretrained model can't beat this, something is wrong with the pipeline.

    Input:  (batch_size, 3, 224, 224)
    Output: (batch_size, 1) — one logit per image
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        # Feature extraction layers (convolutional blocks)
        self.features = nn.Sequential(
            # Block 1: 224×224×3 → 112×112×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112×112×32 → 56×56×64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56×56×64 → 28×28×128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28×28×128 → 14×14×256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Adaptive average pooling: any spatial size → 1×1×256
            # This makes the model work with any input size (not just 224)
            nn.AdaptiveAvgPool2d(1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),           # (batch, 256, 1, 1) → (batch, 256)
            nn.Dropout(dropout),    # Regularization to reduce overfitting
            nn.Linear(256, 1),      # Single output logit for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: images → logit."""
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# 2. PRETRAINED MODEL LOADERS
# =============================================================================

def get_mobilenet_v2(pretrained: bool = True) -> nn.Module:
    """
    Load MobileNetV2 with a modified head for binary classification.

    MobileNetV2 is designed for mobile/edge devices — it's fast and lightweight
    while still achieving good accuracy. Uses depthwise separable convolutions
    to reduce computation.

    Parameters
    ----------
    pretrained : bool
        If True, load ImageNet-pretrained weights (recommended).

    Returns
    -------
    nn.Module
        MobileNetV2 with final layer outputting 1 logit.
    """
    # Load the pretrained model
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # The original classifier outputs 1000 classes (ImageNet).
    # We replace it with a single-output layer for binary classification.
    # MobileNetV2's classifier is: Sequential(Dropout, Linear(1280→1000))
    num_features = model.classifier[1].in_features  # = 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 1),  # 1 logit for binary classification
    )

    return model


def get_resnet50(pretrained: bool = True) -> nn.Module:
    """
    Load ResNet50 with a modified head for binary classification.

    ResNet50 uses residual (skip) connections that allow training very deep
    networks without vanishing gradients. It's one of the most popular
    backbones in medical imaging research.

    Parameters
    ----------
    pretrained : bool
        If True, load ImageNet-pretrained weights (recommended).

    Returns
    -------
    nn.Module
        ResNet50 with final layer outputting 1 logit.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # ResNet's final layer is: model.fc = Linear(2048→1000)
    # Replace with binary output
    num_features = model.fc.in_features  # = 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 1),
    )

    return model


def get_densenet121(pretrained: bool = True) -> nn.Module:
    """
    Load DenseNet121 with a modified head for binary classification.

    DenseNet connects each layer to every other layer in a feed-forward fashion.
    This encourages feature reuse and substantially reduces the number of
    parameters. It's particularly popular in medical imaging (CheXNet used
    DenseNet121 for chest X-ray classification).

    Parameters
    ----------
    pretrained : bool
        If True, load ImageNet-pretrained weights (recommended).

    Returns
    -------
    nn.Module
        DenseNet121 with final layer outputting 1 logit.
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    # DenseNet's classifier is: model.classifier = Linear(1024→1000)
    num_features = model.classifier.in_features  # = 1024
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 1),
    )

    return model


# =============================================================================
# 3. MODEL FACTORY — Get any model by name
# =============================================================================

def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function: get a model by its string name.

    This makes it easy to select models from command-line arguments.

    Parameters
    ----------
    model_name : str
        One of: "custom_cnn", "mobilenet_v2", "resnet50", "densenet121"
    pretrained : bool
        Whether to use ImageNet-pretrained weights (ignored for custom_cnn).

    Returns
    -------
    nn.Module
        The requested model with a single-logit output head.

    Raises
    ------
    ValueError
        If model_name is not recognized.

    Example
    -------
    >>> model = get_model("resnet50", pretrained=True)
    >>> output = model(torch.randn(1, 3, 224, 224))
    >>> output.shape
    torch.Size([1, 1])
    """
    model_name = model_name.lower().strip()

    if model_name == "custom_cnn":
        return CustomCNN()
    elif model_name == "mobilenet_v2":
        return get_mobilenet_v2(pretrained=pretrained)
    elif model_name == "resnet50":
        return get_resnet50(pretrained=pretrained)
    elif model_name == "densenet121":
        return get_densenet121(pretrained=pretrained)
    else:
        available = ["custom_cnn", "mobilenet_v2", "resnet50", "densenet121"]
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {available}"
        )


# =============================================================================
# UTILITY: Count trainable parameters
# =============================================================================

def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.

    Returns
    -------
    dict
        {"total": int, "trainable": int, "frozen": int}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


# =============================================================================
# MAIN — Quick test when run directly
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)

    dummy_input = torch.randn(1, 3, 224, 224)

    for name in ["custom_cnn", "mobilenet_v2", "resnet50", "densenet121"]:
        model = get_model(name, pretrained=False)  # No download for quick test
        output = model(dummy_input)
        params = count_parameters(model)
        print(f"\n  {name:15s} | output shape: {output.shape} | "
              f"params: {params['total']:>10,} | trainable: {params['trainable']:>10,}")

    print("\n" + "=" * 60)
    print("  All models produce shape (1, 1) — single logit ✅")
    print("=" * 60)

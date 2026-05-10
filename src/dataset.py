"""
FracAtlas Bone Fracture Binary Classification — PyTorch Data Pipeline
=====================================================================

This module provides a complete data-loading pipeline for training a CNN to
classify X-ray images as "fracture" or "no fracture" using the FracAtlas dataset.

Key design decisions:
  • Binary classification (fracture=1, no fracture=0)
  • Images loaded as RGB for compatibility with ImageNet-pretrained models
  • Labels returned as float32 (required by BCEWithLogitsLoss)
  • Aspect-ratio preserving resize: pad shorter edge → square → resize to 224×224
  • Train augmentations: rotation, translation, color jitter, optional h-flip
  • NO vertical flip, NO random resized crop (medical images must stay upright)

Author : FracAtlas Data Pipeline Builder Agent
Project: Bone Fracture Detection from X-ray Images
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library
import os
from typing import Optional, Tuple, Dict, List

# Third-party — PyTorch ecosystem
import torch
from torch.utils.data import Dataset, DataLoader

# torchvision provides image transforms and pretrained model utilities
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# pandas for reading CSV metadata files
import pandas as pd

# PIL (Pillow) for loading images from disk
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# numpy for numerical operations (e.g., class counting)
import numpy as np


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — EDIT THESE VARIABLES TO MATCH YOUR CSV FILE           ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  filename_column : the CSV column that contains the image filename     ║
# ║                    (e.g., "img_001.jpg" or a relative path)            ║
# ║  label_column    : the CSV column that contains the binary label       ║
# ║                    (1 = fracture, 0 = no fracture)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

FILENAME_COLUMN: str = "image_id"       # ← EDIT THIS to match your CSV
LABEL_COLUMN: str = "fractured"         # ← EDIT THIS to match your CSV

# =============================================================================
# ImageNet normalization statistics
# These are the mean and std of the ImageNet dataset (per-channel, RGB order).
# We use these because our pretrained backbone was trained on ImageNet.
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target image size for the model (most pretrained CNNs expect 224×224)
TARGET_SIZE: int = 224


# =============================================================================
# CUSTOM TRANSFORM: Aspect-Ratio Preserving Resize (Pad + Resize)
# =============================================================================

class PadToSquareAndResize:
    """
    A custom transform that preserves the aspect ratio of an image by:
      1. Padding the shorter edge with black pixels to make the image square.
      2. Resizing the square image to (target_size × target_size).

    Why do we do this?
    ------------------
    X-ray images come in various aspect ratios (e.g., 2000×1500, 3000×2500).
    If we simply resize them to 224×224, we would DISTORT the anatomy — bones
    would appear stretched or squished. Instead, we pad with black (which looks
    like the background of an X-ray) to make a square, then resize uniformly.

    This is MUCH better than random resized crop for medical images because:
      • We never accidentally crop OUT the fracture region
      • The anatomy stays geometrically correct
      • The model sees the full image every time

    Parameters
    ----------
    target_size : int
        The final square dimension (default: 224 for ImageNet-compatible models).
    fill : int or tuple
        Pixel value used for padding (default: 0 = black, mimics X-ray background).
    """

    def __init__(self, target_size: int = 224, fill: int = 0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the pad-to-square-and-resize transform to a PIL Image.

        Steps:
          1. Get current width (w) and height (h).
          2. Compute how much padding is needed on each side.
          3. Pad the image to make it square (max(w, h) × max(w, h)).
          4. Resize the square image to (target_size × target_size).
        """
        # Get original dimensions
        w, h = img.size  # PIL uses (width, height) convention

        # Find the longer edge — this becomes the side length of our square
        max_side = max(w, h)

        # Calculate padding needed on each side
        # We split the padding evenly between left/right (or top/bottom)
        # so the image stays centered.
        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left  # handles odd-pixel differences
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top

        # Apply padding using torchvision's functional API
        # The padding order is: (left, top, right, bottom)
        img_padded = TF.pad(
            img,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill
        )

        # Resize the now-square image to our target size
        img_resized = TF.resize(
            img_padded,
            size=[self.target_size, self.target_size],
            interpolation=TF.InterpolationMode.BILINEAR
        )

        return img_resized

    def __repr__(self) -> str:
        return f"PadToSquareAndResize(target_size={self.target_size}, fill={self.fill})"


# =============================================================================
# TRANSFORM PIPELINES
# =============================================================================

def get_train_transforms(
    target_size: int = TARGET_SIZE,
    enable_horizontal_flip: bool = True,
) -> T.Compose:
    """
    Build the TRAINING transform pipeline.

    Training transforms include data augmentation to help the model generalize
    better. Each augmentation simulates realistic variations that might occur
    in clinical X-ray acquisition:

      • Rotation (±10°): Patients may be slightly rotated during imaging.
      • Affine translation (±5%): The bone may not be perfectly centered.
      • ColorJitter: Simulates variations in X-ray exposure/brightness.
      • Horizontal flip: A left arm looks like a right arm (optional).

    IMPORTANT — What we do NOT include:
      • NO vertical flip: X-rays have a clear "up" direction (gravity matters
        for fracture patterns). Flipping upside-down is unrealistic.
      • NO random resized crop: We might accidentally crop out the fracture!

    Parameters
    ----------
    target_size : int
        Final image size (default: 224).
    enable_horizontal_flip : bool
        Set to False to disable horizontal flipping (default: True).
        You might disable this if left/right distinction matters for your task.

    Returns
    -------
    torchvision.transforms.Compose
        A composed transform pipeline ready to be passed to the Dataset.
    """
    transform_list = [
        # Step 1: Pad to square and resize (preserves aspect ratio)
        PadToSquareAndResize(target_size=target_size, fill=0),

        # Step 2: Random rotation up to ±10 degrees
        # fill=0 fills any new pixels from rotation with black
        T.RandomRotation(degrees=10, fill=0),

        # Step 3: Random affine translation (shift image up to 5% in x and y)
        # This simulates slight misalignment in patient positioning
        # scale=(1.0, 1.0) means NO random scaling (we don't want to zoom)
        # shear=0 means NO shearing
        T.RandomAffine(
            degrees=0,              # No additional rotation (already done above)
            translate=(0.05, 0.05), # Shift up to 5% of image size
            scale=None,             # No random scaling
            shear=None,             # No shearing
            fill=0                  # Fill exposed pixels with black
        ),

        # Step 4: Brightness and contrast jitter
        # Simulates variation in X-ray exposure settings
        # brightness=0.15 means ±15% brightness change
        # contrast=0.15 means ±15% contrast change
        # saturation and hue are minimal since X-rays are mostly grayscale
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.05,
            hue=0.02
        ),
    ]

    # Step 5: Optional horizontal flip
    # Probability 0.5 means each image has a 50% chance of being flipped
    if enable_horizontal_flip:
        transform_list.append(T.RandomHorizontalFlip(p=0.5))

    # Step 6: Convert PIL Image → PyTorch Tensor
    # This also rescales pixel values from [0, 255] to [0.0, 1.0]
    transform_list.append(T.ToTensor())

    # Step 7: Normalize using ImageNet statistics
    # This is REQUIRED for pretrained models — they expect this normalization
    transform_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transform_list)


def get_val_transforms(target_size: int = TARGET_SIZE) -> T.Compose:
    """
    Build the VALIDATION / TEST transform pipeline.

    This is DETERMINISTIC — no randomness. Every time you pass the same image,
    you get the exact same output. This is essential for:
      • Fair evaluation (no randomness affecting metrics)
      • Reproducible predictions
      • Consistent Grad-CAM visualizations

    The pipeline is simple:
      1. Pad to square and resize (same as training)
      2. Convert to tensor
      3. Normalize with ImageNet stats

    Parameters
    ----------
    target_size : int
        Final image size (default: 224).

    Returns
    -------
    torchvision.transforms.Compose
        A composed transform pipeline for validation/test data.
    """
    return T.Compose([
        # Step 1: Aspect-ratio preserving resize (identical to training)
        PadToSquareAndResize(target_size=target_size, fill=0),

        # Step 2: Convert to tensor (PIL → Tensor, scales to [0, 1])
        T.ToTensor(),

        # Step 3: ImageNet normalization
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# CUSTOM DATASET CLASS
# =============================================================================

class FracAtlasDataset(Dataset):
    """
    PyTorch Dataset for the FracAtlas bone fracture classification task.

    This class:
      1. Reads a CSV file containing image filenames and binary labels.
      2. Loads each image from disk using PIL.
      3. Applies the specified transforms.
      4. Returns (image_tensor, label_tensor) pairs.

    How PyTorch Datasets work (for undergrads):
    -------------------------------------------
    PyTorch's DataLoader needs a Dataset object that implements:
      • __len__()     → returns the total number of samples
      • __getitem__() → returns one (input, target) pair given an index

    The DataLoader then handles batching, shuffling, and parallel loading
    automatically. You never need to write a manual training loop over files!

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns for filenames and labels.
    img_dir : str
        Directory where the image files are stored.
    transform : torchvision.transforms.Compose, optional
        Transform pipeline to apply to each image.
    filename_column : str
        Name of the CSV column containing image filenames.
    label_column : str
        Name of the CSV column containing binary labels (0 or 1).
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[T.Compose] = None,
        filename_column: str = FILENAME_COLUMN,
        label_column: str = LABEL_COLUMN,
    ):
        # Store parameters
        self.img_dir = img_dir
        self.transform = transform
        self.filename_column = filename_column
        self.label_column = label_column

        # Read the CSV file into a pandas DataFrame
        # This gives us a table where each row = one image
        self.df = pd.read_csv(csv_path)

        # Validate that the expected columns exist
        if self.filename_column not in self.df.columns:
            raise ValueError(
                f"Column '{self.filename_column}' not found in CSV. "
                f"Available columns: {list(self.df.columns)}. "
                f"Please edit FILENAME_COLUMN at the top of this file."
            )
        if self.label_column not in self.df.columns:
            raise ValueError(
                f"Column '{self.label_column}' not found in CSV. "
                f"Available columns: {list(self.df.columns)}. "
                f"Please edit LABEL_COLUMN at the top of this file."
            )

        # Extract filenames and labels as lists for fast indexing
        self.filenames: List[str] = self.df[self.filename_column].tolist()
        self.labels: List[int] = self.df[self.label_column].tolist()

        print(f"[FracAtlasDataset] Loaded {len(self)} samples from '{csv_path}'")

    def __len__(self) -> int:
        """Return the total number of images in this dataset split."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return one (image, label) pair.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve (0-based).

        Returns
        -------
        image : torch.Tensor
            Shape (3, 224, 224) — 3 channels (RGB), 224×224 pixels.
        label : torch.Tensor
            Shape () — scalar tensor, dtype=float32.
            Value is 1.0 (fracture) or 0.0 (no fracture).
        """
        # Build the full path to the image file
        img_filename = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)

        # Load image as RGB
        # .convert('RGB') ensures 3 channels even if the source is grayscale.
        # This is REQUIRED for pretrained models that expect 3-channel input.
        image = Image.open(img_path).convert("RGB")

        # Apply transforms (augmentation + normalization)
        if self.transform is not None:
            image = self.transform(image)

        # Convert label to float32 tensor
        # BCEWithLogitsLoss requires the target to be float, not int!
        # This is a common source of bugs for beginners.
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Count how many samples belong to each class.

        Returns
        -------
        dict
            {"no_fracture": count_0, "fracture": count_1}
        """
        labels_array = np.array(self.labels)
        return {
            "no_fracture (0)": int((labels_array == 0).sum()),
            "fracture (1)": int((labels_array == 1).sum()),
        }


# =============================================================================
# DATALOADER CREATION FUNCTIONS
# =============================================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a Dataset.

    What is a DataLoader?
    ---------------------
    A DataLoader wraps a Dataset and provides:
      • Automatic batching (groups samples into batches of `batch_size`)
      • Shuffling (randomizes order each epoch — important for training!)
      • Parallel data loading (uses multiple CPU workers to load images faster)
      • Pin memory (speeds up CPU→GPU transfer)

    Parameters
    ----------
    dataset : Dataset
        The FracAtlasDataset (or any PyTorch Dataset) to load from.
    batch_size : int
        Number of images per batch (default: 32).
        Larger = faster training but more GPU memory.
    shuffle : bool
        Whether to randomize sample order each epoch (default: True).
        ALWAYS True for training, ALWAYS False for validation/test.
    num_workers : int
        Number of parallel processes for data loading (default: 4).
        Set to 0 for debugging (easier to read error messages).
    pin_memory : bool
        If True, tensors are copied to CUDA pinned memory before transfer.
        This speeds up GPU training. Set False if not using GPU.
    drop_last : bool
        If True, drop the last incomplete batch (default: False).
        Useful for training with BatchNorm when last batch might be size 1.

    Returns
    -------
    DataLoader
        Ready-to-use DataLoader for your training loop.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    img_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    enable_horizontal_flip: bool = True,
    target_size: int = TARGET_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders in one call.

    This is a convenience function that sets up the full data pipeline:
      1. Creates appropriate transforms for each split.
      2. Instantiates FracAtlasDataset for each split.
      3. Wraps each in a DataLoader with correct settings.

    Parameters
    ----------
    train_csv : str
        Path to the CSV file for training data.
    val_csv : str
        Path to the CSV file for validation data.
    test_csv : str
        Path to the CSV file for test data.
    img_dir : str
        Directory containing all images (shared across splits).
    batch_size : int
        Batch size for all DataLoaders (default: 32).
    num_workers : int
        Number of parallel data-loading workers (default: 4).
    enable_horizontal_flip : bool
        Whether to include random horizontal flip in training (default: True).
    target_size : int
        Image size after resizing (default: 224).

    Returns
    -------
    tuple of (DataLoader, DataLoader, DataLoader)
        (train_loader, val_loader, test_loader)

    Example
    -------
    >>> train_loader, val_loader, test_loader = create_dataloaders(
    ...     train_csv="data/train.csv",
    ...     val_csv="data/val.csv",
    ...     test_csv="data/test.csv",
    ...     img_dir="data/images/",
    ...     batch_size=32,
    ... )
    >>> for images, labels in train_loader:
    ...     # images.shape = (32, 3, 224, 224)
    ...     # labels.shape = (32,)
    ...     outputs = model(images)
    ...     loss = criterion(outputs.squeeze(), labels)
    """
    # -------------------------------------------------------------------------
    # Step 1: Build transform pipelines
    # -------------------------------------------------------------------------
    train_transforms = get_train_transforms(
        target_size=target_size,
        enable_horizontal_flip=enable_horizontal_flip,
    )
    val_transforms = get_val_transforms(target_size=target_size)

    # -------------------------------------------------------------------------
    # Step 2: Create Dataset objects
    # -------------------------------------------------------------------------
    train_dataset = FracAtlasDataset(
        csv_path=train_csv,
        img_dir=img_dir,
        transform=train_transforms,
    )
    val_dataset = FracAtlasDataset(
        csv_path=val_csv,
        img_dir=img_dir,
        transform=val_transforms,  # Deterministic — no augmentation
    )
    test_dataset = FracAtlasDataset(
        csv_path=test_csv,
        img_dir=img_dir,
        transform=val_transforms,  # Test also uses deterministic transforms
    )

    # -------------------------------------------------------------------------
    # Step 3: Wrap in DataLoaders
    # -------------------------------------------------------------------------
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,           # IMPORTANT: shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,         # Drop last incomplete batch for stable BatchNorm
    )
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,          # NEVER shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,        # Keep all samples for accurate metrics
    )
    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,          # NEVER shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# SANITY CHECK FUNCTION
# =============================================================================

def sanity_check(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    img_dir: str,
    target_size: int = TARGET_SIZE,
) -> None:
    """
    Run a quick sanity check on the data pipeline.

    This function verifies that everything works correctly by:
      1. Printing the number of images in each split (train/val/test).
      2. Printing the class distribution (fracture vs. no fracture).
      3. Loading ONE sample and printing its tensor shape and label.

    Run this BEFORE starting training to catch common issues like:
      • Wrong CSV column names
      • Missing image files
      • Incorrect label encoding
      • Transform errors

    Parameters
    ----------
    train_csv : str
        Path to training CSV.
    val_csv : str
        Path to validation CSV.
    test_csv : str
        Path to test CSV.
    img_dir : str
        Directory containing image files.
    target_size : int
        Expected image size (default: 224).

    Example
    -------
    >>> sanity_check(
    ...     train_csv="data/train.csv",
    ...     val_csv="data/val.csv",
    ...     test_csv="data/test.csv",
    ...     img_dir="data/images/"
    ... )
    """
    print("=" * 70)
    print("  FRACATLAS DATA PIPELINE — SANITY CHECK")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Create datasets (with val transforms for quick deterministic loading)
    # -------------------------------------------------------------------------
    val_transforms = get_val_transforms(target_size=target_size)

    datasets_info = {}
    for split_name, csv_path in [("TRAIN", train_csv), ("VAL", val_csv), ("TEST", test_csv)]:
        try:
            ds = FracAtlasDataset(
                csv_path=csv_path,
                img_dir=img_dir,
                transform=val_transforms,
            )
            datasets_info[split_name] = ds
        except Exception as e:
            print(f"\n  ❌ ERROR loading {split_name} split: {e}")
            datasets_info[split_name] = None

    # -------------------------------------------------------------------------
    # 2. Print dataset sizes
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  📊 DATASET SIZES")
    print("-" * 70)
    total = 0
    for split_name, ds in datasets_info.items():
        if ds is not None:
            count = len(ds)
            total += count
            print(f"    {split_name:6s}: {count:,} images")
        else:
            print(f"    {split_name:6s}: ❌ FAILED TO LOAD")
    print(f"    {'TOTAL':6s}: {total:,} images")

    # -------------------------------------------------------------------------
    # 3. Print class distribution for each split
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  ⚖️  CLASS DISTRIBUTION")
    print("-" * 70)
    for split_name, ds in datasets_info.items():
        if ds is not None:
            dist = ds.get_class_distribution()
            n_total = len(ds)
            n_frac = dist["fracture (1)"]
            n_no_frac = dist["no_fracture (0)"]
            ratio = n_frac / n_no_frac if n_no_frac > 0 else float("inf")
            print(f"    {split_name:6s}: "
                  f"No Fracture = {n_no_frac:,} ({100*n_no_frac/n_total:.1f}%) | "
                  f"Fracture = {n_frac:,} ({100*n_frac/n_total:.1f}%) | "
                  f"Ratio = 1:{ratio:.2f}")

    # -------------------------------------------------------------------------
    # 4. Load one sample and print tensor info
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  🔍 SAMPLE INSPECTION (first image from TRAIN split)")
    print("-" * 70)

    train_ds = datasets_info.get("TRAIN")
    if train_ds is not None and len(train_ds) > 0:
        try:
            sample_image, sample_label = train_ds[0]
            print(f"    Image tensor shape : {sample_image.shape}")
            print(f"    Image tensor dtype : {sample_image.dtype}")
            print(f"    Image value range  : [{sample_image.min():.3f}, {sample_image.max():.3f}]")
            print(f"    Label value        : {sample_label.item()}")
            print(f"    Label dtype        : {sample_label.dtype}")
            print(f"    Expected shape     : torch.Size([3, {target_size}, {target_size}])")

            # Verify correctness
            assert sample_image.shape == torch.Size([3, target_size, target_size]), \
                f"Shape mismatch! Got {sample_image.shape}"
            assert sample_label.dtype == torch.float32, \
                f"Label dtype should be float32, got {sample_label.dtype}"
            print("\n    ✅ All checks passed! Pipeline is working correctly.")
        except FileNotFoundError as e:
            print(f"\n    ❌ Image file not found: {e}")
            print("       Check that 'img_dir' points to the correct folder.")
        except Exception as e:
            print(f"\n    ❌ Error loading sample: {e}")
    else:
        print("    ❌ Cannot inspect sample — TRAIN dataset not loaded.")

    # -------------------------------------------------------------------------
    # 5. Quick DataLoader test
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  🚀 DATALOADER TEST (batch_size=4)")
    print("-" * 70)
    if train_ds is not None and len(train_ds) >= 4:
        try:
            test_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)
            batch_images, batch_labels = next(iter(test_loader))
            print(f"    Batch images shape : {batch_images.shape}")
            print(f"    Batch labels shape : {batch_labels.shape}")
            print(f"    Batch labels       : {batch_labels.tolist()}")
            print("\n    ✅ DataLoader is working correctly!")
        except Exception as e:
            print(f"\n    ❌ DataLoader error: {e}")

    print("\n" + "=" * 70)
    print("  SANITY CHECK COMPLETE")
    print("=" * 70)


# =============================================================================
# MAIN — Run sanity check when this file is executed directly
# =============================================================================

if __name__ == "__main__":
    """
    Usage example:
        python src/dataset.py

    Before running, make sure you have:
      1. A CSV file with columns matching FILENAME_COLUMN and LABEL_COLUMN
      2. An image directory containing the referenced image files
      3. Edited the paths below to match your local setup
    """
    # =========================================================================
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  EDIT THESE PATHS to match your local directory structure           ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    # =========================================================================
    TRAIN_CSV = "data/train.csv"
    VAL_CSV = "data/val.csv"
    TEST_CSV = "data/test.csv"
    IMG_DIR = "data/images/"

    # Run the sanity check
    sanity_check(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        img_dir=IMG_DIR,
    )

    # =========================================================================
    # Example: Create full DataLoaders for training
    # =========================================================================
    # Uncomment the following to create DataLoaders:
    #
    # train_loader, val_loader, test_loader = create_dataloaders(
    #     train_csv=TRAIN_CSV,
    #     val_csv=VAL_CSV,
    #     test_csv=TEST_CSV,
    #     img_dir=IMG_DIR,
    #     batch_size=32,
    #     num_workers=4,
    #     enable_horizontal_flip=True,  # Set to False to disable h-flip
    # )
    #
    # # Training loop skeleton:
    # for epoch in range(num_epochs):
    #     for images, labels in train_loader:
    #         images = images.to(device)   # Move to GPU
    #         labels = labels.to(device)
    #         outputs = model(images)       # Forward pass
    #         loss = criterion(outputs.squeeze(), labels)  # BCEWithLogitsLoss
    #         ...

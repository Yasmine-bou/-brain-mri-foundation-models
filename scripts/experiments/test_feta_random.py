"""
Smoke test for a randomly initialized ResEncL encoder on FeTA MRI data.

This script verifies that a ResEncL encoder with randomly initialized
weights can process a preprocessed fetal brain MRI from the FeTA dataset.

The following steps are performed:
    1. Load the selected preprocessed FeTA MRI volume.
    2. Extract a 96 x 96 x 96 center crop, with zero padding if needed.
    3. Build a randomly initialized ResEncL architecture.
    4. Extract and freeze the encoder.
    5. Perform a forward pass without gradient computation.
    6. Display feature-map shapes and check numerical validity.

Expected input:
    A single-channel FeTA MRI volume with shape (1, D, H, W).

Network input:
    A PyTorch tensor with shape (1, 1, 96, 96, 96).

Expected output:
    Intermediate feature maps produced by the random encoder.

Requirements:
    - PyTorch with CUDA support and an available CUDA GPU.
    - nnU-Net v2 with the ResEncL architecture.
    - Blosc2 and NumPy.

Notes:
    - The selected FeTA subject is sub-006.
    - No pretrained checkpoint is loaded.
    - The random initialization seed is fixed to 42.
    - The encoder is frozen and evaluated without training.
    - This script tests forward-pass functionality, not
      segmentation quality or model generalization.
"""

from pathlib import Path

import blosc2
import numpy as np
import torch

from nnunetv2.utilities.get_network_via_name import get_network_from_name


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Directory containing preprocessed FeTA validation MRI volumes.
DATA_DIR = Path(
    "/home/bouattou/brain-mri-foundation-models/"
    "data/nnUNet_preprocessed/Dataset101_FeTA/SSL3D_val"
)

# Spatial size of the cubic MRI input patch.
PATCH_SIZE = 96


# ---------------------------------------------------------------------
# 1. Load the selected preprocessed FeTA MRI
# ---------------------------------------------------------------------

print("=== 1. LOADING FETA MRI ===", flush=True)

# Load the MRI volume for subject sub-006.
# The file contains a preprocessed single-channel fetal MRI.
mri = np.asarray(blosc2.open(str(DATA_DIR / "sub-006.b2nd"), mode="r")[:])

# Verify the expected four-dimensional shape:
# (channels=1, depth, height, width).
assert mri.ndim == 4 and mri.shape[0] == 1


# ---------------------------------------------------------------------
# 2. Extract a fixed-size 3D patch
# ---------------------------------------------------------------------

# Initialize the output patch with zeros.
# This also provides zero padding for dimensions smaller
# than the requested patch size.
patch = np.zeros(
    (1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
    dtype=np.float32,
)

# Compute the starting coordinates for a center crop.
#
# For each spatial dimension:
#     - Use the centered starting position when large enough.
#     - Use coordinate zero when smaller than PATCH_SIZE.
starts = [max(0, (s - PATCH_SIZE) // 2) for s in mri.shape[1:]]

# Extract the requested region from the original MRI.
# NumPy slicing automatically handles image boundaries.
cropped = mri[
    :,
    starts[0] : starts[0] + PATCH_SIZE,
    starts[1] : starts[1] + PATCH_SIZE,
    starts[2] : starts[2] + PATCH_SIZE,
]

# Get the extracted depth, height, and width.
d, h, w = cropped.shape[1:]

# Copy the cropped MRI into the fixed-size output patch.
# Remaining voxels keep their zero-padding value.
patch[:, :d, :h, :w] = cropped

# Convert the patch to a PyTorch tensor and add the batch dimension.
#
# Resulting shape: (batch=1, channels=1, D=96, H=96, W=96).
#
# Move the input to the CUDA GPU.
x = torch.from_numpy(patch).unsqueeze(0).to("cuda")

print("Input tensor:", tuple(x.shape), flush=True)


# ---------------------------------------------------------------------
# 3. Build a randomly initialized ResEncL encoder
# ---------------------------------------------------------------------

print("\n=== 2. BUILDING RANDOM RESENCL ===", flush=True)

# Fix the PyTorch random seed before model initialization.
# This supports reproducibility of the initial random weights.
torch.manual_seed(42)

# Build the official ResEncL segmentation architecture.
#
# input_channels=1:
#     Single-channel fetal MRI input.
#
# output_channels=8:
#     Background and seven foreground tissue classes.
#
# allow_init=True:
#     Initialize model parameters randomly.
#
# deep_supervision=False:
#     Disable additional deep-supervision outputs.
network = get_network_from_name(
    arch_class_name="ResEncL",
    input_channels=1,
    output_channels=8,
    allow_init=True,
    deep_supervision=False,
)

# No pretrained checkpoint is used.
# The encoder therefore retains its randomly initialized weights.
encoder = network.encoder

# Disable gradient updates for all encoder parameters.
encoder.requires_grad_(False)

# Move the encoder to the GPU and activate evaluation mode.
encoder = encoder.cuda()
encoder.eval()

print("Random encoder initialized and frozen!", flush=True)


# ---------------------------------------------------------------------
# 4. Run the forward pass
# ---------------------------------------------------------------------

print("\n=== 3. FORWARD PASS ===", flush=True)

# Perform inference without building an autograd graph.
# No backward pass or optimization is performed.
with torch.inference_mode():
    features = encoder(x)

# Convert a single tensor output into a list for uniform processing.
if isinstance(features, torch.Tensor):
    features = [features]

# Inspect each feature map returned by the random encoder.
for i, feature in enumerate(features):
    # Display feature-map dimensions and check numerical validity.
    # Finite values indicate the absence of NaN and infinity.
    print(
        f"Feature map {i}: {tuple(feature.shape)} "
        f"| finite={bool(torch.isfinite(feature).all())}",
        flush=True,
    )

print("\nRANDOM FORWARD PASS SUCCESSFUL!", flush=True)

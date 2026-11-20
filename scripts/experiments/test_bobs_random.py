"""
Smoke test for a randomly initialized ResEncL encoder on BOBs MRI data.

This script verifies that a ResEncL encoder with randomly initialized
weights can successfully process a preprocessed Baby Open Brains (BOBs)
MRI volume.

The following steps are performed:
    1. Load one preprocessed BOBs MRI volume from a .b2nd file.
    2. Extract a 96 x 96 x 96 center crop, with zero padding if needed.
    3. Build a randomly initialized ResEncL network.
    4. Extract and freeze the encoder.
    5. Perform a forward pass without gradient computation.
    6. Display the shapes and numerical validity of the feature maps.

Expected input:
    A single-channel MRI volume with shape (1, D, H, W).

Network input:
    A PyTorch tensor with shape (1, 1, 96, 96, 96).

Expected output:
    Intermediate feature maps produced by the random encoder.

Requirements:
    - PyTorch with CUDA support and an available CUDA GPU.
    - nnU-Net v2 with the ResEncL architecture.
    - Blosc2 and NumPy.

Notes:
    - No pretrained checkpoint is loaded.
    - The random initialization seed is fixed to 42.
    - The encoder is frozen and used only for inference.
    - This script verifies forward-pass execution and does not
      train a segmentation model or evaluate segmentation accuracy.
"""

from pathlib import Path

import blosc2
import numpy as np
import torch

from nnunetv2.utilities.get_network_via_name import get_network_from_name


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Directory containing preprocessed BOBs validation MRI volumes.
DATA_DIR = Path(
    "/home/bouattou/brain-mri-foundation-models/"
    "data/nnUNet_preprocessed/Dataset102_BOBs/SSL3D_val"
)

# Spatial dimensions of the input patch.
PATCH_SIZE = 96


# ---------------------------------------------------------------------
# 1. Load one preprocessed BOBs MRI volume
# ---------------------------------------------------------------------

print("=== 1. LOADING BOBS MRI ===", flush=True)

# Find available MRI volumes, excluding segmentation masks.
#
# Sorting provides a consistent ordering of the available cases.
# The first volume is selected for this forward-pass test.
mri_path = sorted(
    p for p in DATA_DIR.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")
)[0]

print("Selected BOBs MRI:", mri_path.name, flush=True)

# Load the complete MRI volume from its Blosc2 file.
mri = np.asarray(blosc2.open(str(mri_path), mode="r")[:])

# Verify the expected MRI dimensions: (1, D, H, W).
assert mri.ndim == 4 and mri.shape[0] == 1


# ---------------------------------------------------------------------
# 2. Extract a fixed-size 3D patch
# ---------------------------------------------------------------------

# Allocate a zero-filled patch with one image channel.
# Zero padding remains in regions not covered by the original MRI.
patch = np.zeros(
    (1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
    dtype=np.float32,
)

# Compute center-crop starting coordinates along the three axes.
# Smaller dimensions start at coordinate zero.
starts = [max(0, (s - PATCH_SIZE) // 2) for s in mri.shape[1:]]

# Extract the requested region from the MRI.
# If the MRI is smaller than the patch, the extracted region
# is automatically limited by the image boundaries.
cropped = mri[
    :,
    starts[0] : starts[0] + PATCH_SIZE,
    starts[1] : starts[1] + PATCH_SIZE,
    starts[2] : starts[2] + PATCH_SIZE,
]

# Get the dimensions of the extracted MRI region.
d, h, w = cropped.shape[1:]

# Copy the extracted region into the output patch.
# Remaining voxels retain the zero-padding value.
patch[:, :d, :h, :w] = cropped

# Convert the patch into a PyTorch tensor and add a batch dimension.
#
# Resulting shape:
# (batch=1, channels=1, depth=96, height=96, width=96).
#
# Transfer the tensor to the CUDA GPU.
x = torch.from_numpy(patch).unsqueeze(0).to("cuda")

print("Input tensor:", tuple(x.shape), flush=True)


# ---------------------------------------------------------------------
# 3. Build a randomly initialized ResEncL encoder
# ---------------------------------------------------------------------

print("\n=== 2. BUILDING RANDOM RESENCL ===", flush=True)

# Set the random seed before constructing the network.
# This supports reproducible random parameter initialization.
torch.manual_seed(42)

# Instantiate the official ResEncL architecture.
#
# input_channels=1:
#     Single-channel MRI input.
#
# output_channels=8:
#     Background and seven foreground segmentation classes.
#
# allow_init=True:
#     Enable random initialization of model parameters.
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

# No pretrained checkpoint is loaded.
# The encoder therefore retains its randomly initialized weights.
encoder = network.encoder

# Freeze the encoder parameters.
# The random encoder is evaluated without parameter updates.
encoder.requires_grad_(False)

# Move the encoder to GPU and enable evaluation mode.
encoder = encoder.cuda()
encoder.eval()

print("Random encoder initialized and frozen!", flush=True)


# ---------------------------------------------------------------------
# 4. Run the forward pass
# ---------------------------------------------------------------------

print("\n=== 3. FORWARD PASS ===", flush=True)

# Disable gradient tracking for memory-efficient inference.
with torch.inference_mode():
    features = encoder(x)

# Convert a single tensor output into a list for consistent iteration.
if isinstance(features, torch.Tensor):
    features = [features]

# Inspect each intermediate feature map.
for i, feature in enumerate(features):
    # Display the feature-map shape and check whether all values
    # are finite (neither NaN nor infinity).
    print(
        f"Feature map {i}: {tuple(feature.shape)} "
        f"| finite={bool(torch.isfinite(feature).all())}",
        flush=True,
    )

print("\nRANDOM FORWARD PASS SUCCESSFUL!", flush=True)

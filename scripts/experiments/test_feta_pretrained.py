"""
Smoke test for the pretrained SSL3D ResEncL encoder on FeTA MRI data.

This script verifies that a pretrained ResEncL encoder can process
a preprocessed fetal brain MRI volume from the FeTA dataset.

The following steps are performed:
    1. Load a selected preprocessed FeTA MRI volume.
    2. Extract a 96 x 96 x 96 center crop, with zero padding if needed.
    3. Build the official ResEncL network using nnU-Net v2.
    4. Load pretrained SSL3D encoder weights from a checkpoint.
    5. Freeze the encoder and switch it to evaluation mode.
    6. Perform a forward pass without gradient computation.
    7. Display the resulting feature-map shapes and numerical validity.

Expected input:
    A single-channel fetal MRI volume with shape (1, D, H, W).

Network input:
    A PyTorch tensor with shape (1, 1, 96, 96, 96).

Expected output:
    Intermediate feature maps produced by the pretrained encoder.

Requirements:
    - PyTorch with CUDA support and an available CUDA GPU.
    - nnU-Net v2 with the ResEncL architecture.
    - Blosc2 and NumPy.
    - A compatible pretrained SSL3D checkpoint.

Notes:
    - The selected FeTA subject is sub-006.
    - Only the encoder is evaluated; no decoder is used.
    - The pretrained encoder parameters are frozen.
    - No segmentation loss, optimization, or evaluation metrics
      are computed in this test.
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

# Path to the pretrained SSL3D checkpoint.
CKPT_PATH = Path("/home/bouattou/ssl3d_checkpoints/checkpoint_best.pth")

# Spatial dimensions of the 3D input patch.
PATCH_SIZE = 96


# ---------------------------------------------------------------------
# 1. Load a preprocessed FeTA MRI volume
# ---------------------------------------------------------------------

print("=== 1. LOADING FETA MRI ===", flush=True)

# Select the preprocessed MRI volume for subject sub-006.
mri_path = DATA_DIR / "sub-006.b2nd"

# Load the complete MRI volume into a NumPy array.
mri = np.asarray(blosc2.open(str(mri_path), mode="r")[:])

print("MRI shape:", mri.shape, flush=True)

# Expected shape: (1, D, H, W).
# The first dimension represents the MRI channel.
assert mri.ndim == 4 and mri.shape[0] == 1


# ---------------------------------------------------------------------
# 2. Extract a fixed-size 3D patch
# ---------------------------------------------------------------------

# Allocate a zero-filled output patch with the requested dimensions.
# Regions outside the available MRI data remain zero-padded.
patch = np.zeros((1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

# Compute the starting coordinate of the center crop
# independently for depth, height, and width.
#
# If a spatial dimension is smaller than PATCH_SIZE,
# its starting coordinate is set to zero.
for_axis = []
for size in mri.shape[1:]:
    start = max(0, (size - PATCH_SIZE) // 2)
    for_axis.append(start)

# Extract the corresponding region from the MRI volume.
# The crop is limited automatically by the image boundaries.
cropped = mri[
    :,
    for_axis[0] : for_axis[0] + PATCH_SIZE,
    for_axis[1] : for_axis[1] + PATCH_SIZE,
    for_axis[2] : for_axis[2] + PATCH_SIZE,
]

# Retrieve the actual dimensions of the extracted region.
d, h, w = cropped.shape[1:]

# Copy the cropped MRI into the zero-initialized patch.
# Any unused region remains zero-padded.
patch[:, :d, :h, :w] = cropped

# Convert the NumPy patch to a PyTorch tensor.
#
# The additional batch dimension changes the shape from
# (1, 96, 96, 96) to (1, 1, 96, 96, 96).
#
# Transfer the tensor to the CUDA GPU.
x = torch.from_numpy(patch).unsqueeze(0).to("cuda")

print("Input tensor:", tuple(x.shape), flush=True)


# ---------------------------------------------------------------------
# 3. Build the official ResEncL architecture
# ---------------------------------------------------------------------

print("\n=== 2. BUILDING RESENCL ===", flush=True)

# Instantiate the official ResEncL segmentation architecture.
#
# input_channels=1:
#     Single-channel fetal MRI input.
#
# output_channels=8:
#     Eight segmentation classes (background + seven tissues).
#
# allow_init=False:
#     Disable random initialization because pretrained weights
#     will be loaded into the encoder.
#
# deep_supervision=False:
#     Disable additional deep-supervision outputs.
network = get_network_from_name(
    arch_class_name="ResEncL",
    input_channels=1,
    output_channels=8,
    allow_init=False,
    deep_supervision=False,
)

# Extract the encoder from the complete segmentation network.
# The decoder is not required for this forward-pass test.
encoder = network.encoder


# ---------------------------------------------------------------------
# 4. Load pretrained SSL3D encoder weights
# ---------------------------------------------------------------------

print("\n=== 3. LOADING PRETRAINED WEIGHTS ===", flush=True)

# Load the pretrained checkpoint onto CPU.
#
# weights_only=True uses PyTorch's restricted loading mode
# for supported weight-oriented checkpoint contents.
checkpoint = torch.load(
    CKPT_PATH,
    map_location="cpu",
    weights_only=True,
)

# Retrieve the state dictionary containing network parameters.
weights = checkpoint["network_weights"]

# Extract encoder parameters from the saved network state.
#
# Only parameters belonging to "encoder.stem." and
# "encoder.stages." are selected.
#
# Removing the "encoder." prefix aligns their names with
# the standalone encoder's state dictionary.
encoder_weights = {
    key.removeprefix("encoder."): value
    for key, value in weights.items()
    if key.startswith(("encoder.stem.", "encoder.stages."))
}

# Strict loading requires the checkpoint parameters to match
# the expected encoder parameter names and tensor shapes.
encoder.load_state_dict(encoder_weights, strict=True)

print("Pretrained encoder loaded successfully!", flush=True)
print("Loaded encoder entries:", len(encoder_weights), flush=True)

# Release references to objects no longer required for inference.
del checkpoint, weights, encoder_weights, network

# Freeze all pretrained encoder parameters.
# This prevents gradient-based updates to encoder weights.
encoder.requires_grad_(False)

# Move the encoder to the CUDA GPU.
encoder = encoder.cuda()

# Switch the encoder to evaluation mode.
encoder.eval()


# ---------------------------------------------------------------------
# 5. Run the forward pass
# ---------------------------------------------------------------------

print("\n=== 4. FORWARD PASS ===", flush=True)

# Run the pretrained encoder without gradient tracking.
# No backward pass or optimizer step is performed.
with torch.inference_mode():
    features = encoder(x)

# Standardize a single tensor output into a list.
if isinstance(features, torch.Tensor):
    features = [features]

# Inspect the intermediate feature maps.
for i, feature in enumerate(features):
    # Display the feature-map dimensions and numerical validity.
    # torch.isfinite checks for NaN and infinity values.
    print(
        f"Feature map {i}: {tuple(feature.shape)} "
        f"| finite={bool(torch.isfinite(feature).all())}",
        flush=True,
    )

print("\nFORWARD PASS SUCCESSFUL!", flush=True)

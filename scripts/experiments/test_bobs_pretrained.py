"""
Smoke test for the pretrained SSL3D ResEncL encoder on BOBs MRI data.

This script verifies that a pretrained ResEncL encoder can successfully
process a preprocessed Baby Open Brains (BOBs) MRI volume.

The following steps are performed:
    1. Load one preprocessed BOBs MRI volume from a .b2nd file.
    2. Extract a 96 x 96 x 96 center crop, with zero padding if needed.
    3. Build the official ResEncL architecture using nnU-Net v2.
    4. Load pretrained SSL3D encoder weights from a checkpoint.
    5. Freeze the encoder and switch it to evaluation mode.
    6. Perform a forward pass without gradient computation.
    7. Verify the number, shapes, and numerical validity of feature maps.

Expected input:
    A single-channel MRI volume with shape (1, D, H, W).

Network input:
    A PyTorch tensor with shape (1, 1, 96, 96, 96), where the
    dimensions represent (batch, channels, depth, height, width).

Expected output:
    Six feature maps produced by the pretrained encoder.

Requirements:
    - PyTorch with CUDA support and an available CUDA GPU.
    - nnU-Net v2 with the ResEncL architecture.
    - Blosc2 and NumPy.
    - A compatible pretrained SSL3D checkpoint.

Notes:
    - The encoder is frozen; no model training is performed.
    - No segmentation decoder or segmentation metrics are evaluated.
    - This script validates encoder compatibility and forward-pass
      execution, not segmentation performance.
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

# Path to the pretrained SSL3D model checkpoint.
CKPT_PATH = Path("/home/bouattou/ssl3d_checkpoints/checkpoint_best.pth")

# Spatial dimensions of the 3D input patch.
PATCH_SIZE = 96


# ---------------------------------------------------------------------
# 1. Load one preprocessed BOBs MRI volume
# ---------------------------------------------------------------------

print("=== 1. LOADING BOBS MRI ===", flush=True)

# Find all MRI volumes stored as .b2nd files.
# Exclude segmentation masks, identified by the "_seg.b2nd" suffix.
#
# Sorting ensures that the same first volume is selected when
# the directory contents remain unchanged.
mri_paths = sorted(
    p for p in DATA_DIR.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")
)

# Stop execution if no MRI volumes are available.
assert mri_paths, f"No MRI found in {DATA_DIR}"

# Select the first available BOBs MRI volume.
mri_path = mri_paths[0]
print("Selected BOBs MRI:", mri_path.name, flush=True)

# Load the complete preprocessed MRI into a NumPy array.
mri = np.asarray(blosc2.open(str(mri_path), mode="r")[:])

print("MRI shape:", mri.shape, flush=True)

# Expected shape: (1, D, H, W).
# The first dimension represents the single MRI channel.
assert mri.ndim == 4 and mri.shape[0] == 1


# ---------------------------------------------------------------------
# 2. Extract a fixed-size 3D patch
# ---------------------------------------------------------------------

# Allocate a zero-filled output patch.
# Zero padding is retained wherever the MRI is smaller than the
# requested patch dimensions.
patch = np.zeros(
    (1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
    dtype=np.float32,
)

# Determine the center-crop starting coordinate for each spatial axis.
#
# For dimensions larger than PATCH_SIZE, the crop is centered.
# For smaller dimensions, the starting coordinate is set to zero.
starts = [max(0, (s - PATCH_SIZE) // 2) for s in mri.shape[1:]]

# Extract the selected spatial region from the original MRI.
# NumPy slicing automatically stops at the volume boundary when
# a spatial dimension is smaller than PATCH_SIZE.
cropped = mri[
    :,
    starts[0] : starts[0] + PATCH_SIZE,
    starts[1] : starts[1] + PATCH_SIZE,
    starts[2] : starts[2] + PATCH_SIZE,
]

# Retrieve the actual spatial dimensions of the extracted region.
d, h, w = cropped.shape[1:]

# Copy the extracted region into the fixed-size output patch.
# Any remaining area stays zero-padded.
patch[:, :d, :h, :w] = cropped

# Convert the NumPy patch to a PyTorch tensor.
#
# unsqueeze(0) introduces the batch dimension:
# (1, 96, 96, 96) -> (1, 1, 96, 96, 96).
#
# Move the input tensor to the CUDA GPU.
x = torch.from_numpy(patch).unsqueeze(0).to("cuda")

print("Input tensor:", tuple(x.shape), flush=True)


# ---------------------------------------------------------------------
# 3. Build the official ResEncL architecture
# ---------------------------------------------------------------------

print("\n=== 2. BUILDING RESENCL ===", flush=True)

# Instantiate the ResEncL segmentation architecture.
#
# input_channels=1:
#     The input MRI contains one imaging channel.
#
# output_channels=8:
#     The complete network is configured for eight segmentation
#     classes (background + seven foreground classes).
#
# allow_init=False:
#     Disable random weight initialization because pretrained
#     encoder parameters will be loaded from a checkpoint.
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

# Extract the encoder component.
# Only the encoder is evaluated in this script.
encoder = network.encoder


# ---------------------------------------------------------------------
# 4. Load pretrained SSL3D encoder weights
# ---------------------------------------------------------------------

print("\n=== 3. LOADING PRETRAINED WEIGHTS ===", flush=True)

# Load the checkpoint onto CPU before transferring the encoder to GPU.
#
# weights_only=True restricts checkpoint loading to supported
# weight-oriented data structures.
checkpoint = torch.load(
    CKPT_PATH,
    map_location="cpu",
    weights_only=True,
)

# Retrieve the saved network parameter dictionary.
weights = checkpoint["network_weights"]

# Select encoder parameters only.
#
# Keys beginning with "encoder.stem." or "encoder.stages."
# correspond to encoder components in the saved network.
#
# Remove the "encoder." prefix because the extracted encoder
# expects parameter names relative to its own architecture.
encoder_weights = {
    key.removeprefix("encoder."): value
    for key, value in weights.items()
    if key.startswith(("encoder.stem.", "encoder.stages."))
}

# Strict loading verifies that the parameter names and shapes
# match the expected encoder architecture.
encoder.load_state_dict(encoder_weights, strict=True)

print("Pretrained encoder loaded successfully!", flush=True)
print("Loaded encoder entries:", len(encoder_weights), flush=True)

# Release references to objects no longer needed for inference.
del checkpoint, weights, encoder_weights, network


# ---------------------------------------------------------------------
# 5. Freeze the pretrained encoder
# ---------------------------------------------------------------------

# Disable gradient computation for every encoder parameter.
# This prevents encoder weights from being updated during training.
encoder.requires_grad_(False)

# Move the encoder to the CUDA GPU.
encoder = encoder.cuda()

# Switch to evaluation mode.
# This controls the behavior of layers such as BatchNorm and Dropout,
# when present in the architecture.
encoder.eval()

# Verify that all encoder parameters are frozen.
assert not any(p.requires_grad for p in encoder.parameters())

print("Encoder frozen successfully!", flush=True)


# ---------------------------------------------------------------------
# 6. Run the forward pass and validate feature maps
# ---------------------------------------------------------------------

print("\n=== 4. FORWARD PASS ===", flush=True)

# Perform inference without constructing an autograd graph.
# This reduces memory usage during the forward-pass test.
with torch.inference_mode():
    features = encoder(x)

# Standardize the output representation to a list.
# This allows the feature maps to be processed consistently.
if isinstance(features, torch.Tensor):
    features = [features]

# The ResEncL encoder configuration used here is expected
# to produce six intermediate feature maps.
assert len(features) == 6, f"Expected 6 feature maps, got {len(features)}"

# Inspect each feature map produced by the encoder.
for i, feature in enumerate(features):
    # Check that all activation values are finite.
    # Non-finite values (NaN or infinity) may indicate numerical issues.
    is_finite = bool(torch.isfinite(feature).all())

    # Display spatial dimensions and numerical validity.
    print(
        f"Feature map {i}: {tuple(feature.shape)} | finite={is_finite}",
        flush=True,
    )

    # Stop execution if invalid activations are detected.
    assert is_finite, f"Non-finite values in feature map {i}"

print("\nBOBS PRETRAINED FORWARD PASS SUCCESSFUL!", flush=True)

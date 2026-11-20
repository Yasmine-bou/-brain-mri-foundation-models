"""
Smoke test for pretrained and randomly initialized frozen ResEncL encoders.

This script verifies that the frozen encoder implementation can be
successfully initialized and executed in both supported modes:

    - "pretrained": Load encoder weights from an SSL3D checkpoint.
    - "random": Initialize encoder weights randomly.

The encoders are created using the make_frozen_encoder() utility
implemented in src.models.frozen_encoder.

Main validation steps:
    1. Define the pretrained SSL3D checkpoint location.
    2. Initialize the pretrained and random encoders independently.
    3. Move each encoder to the CUDA GPU.
    4. Verify that all encoder parameters are frozen.
    5. Verify that each encoder is in evaluation mode.
    6. Generate a synthetic 3D MRI input tensor.
    7. Perform a forward pass without gradient computation.
    8. Verify that six feature maps are produced.
    9. Check that all feature-map values are finite.
   10. Release GPU resources between encoder tests.

Expected input:
    A synthetic single-channel 3D MRI tensor with shape
    (1, 1, 96, 96, 96).

    Tensor dimensions:
        - Batch size: 1
        - Input channels: 1
        - Depth: 96
        - Height: 96
        - Width: 96

Expected output:
    Six multiscale feature maps produced by each ResEncL encoder.

Requirements:
    - PyTorch with CUDA support.
    - An available CUDA-compatible GPU.
    - The ResEncL architecture and its dependencies.
    - src.models.frozen_encoder.make_frozen_encoder.
    - A compatible SSL3D pretrained checkpoint.

Usage (from the repository root):
    python -m scripts.training.frozen_encoder

Notes:
    - This is a smoke test, not a model training script.
    - Synthetic random input is used instead of real MRI data.
    - No segmentation decoder or segmentation masks are involved.
    - Both encoders remain frozen throughout execution.
    - No backward pass, optimization, or parameter updates occur.
    - No results or model checkpoints are saved.
"""

import torch
from pathlib import Path
from src.models.frozen_encoder import make_frozen_encoder


# ---------------------------------------------------------------------
# 1. Pretrained checkpoint configuration
# ---------------------------------------------------------------------

# Path to the pretrained SSL3D checkpoint.
#
# The checkpoint is used only in "pretrained" mode.
# In "random" mode, the encoder is initialized without loading
# pretrained weights.
checkpoint = Path.home() / "ssl3d_checkpoints/checkpoint_best.pth"


# ---------------------------------------------------------------------
# 2. Test pretrained and random encoder initialization
# ---------------------------------------------------------------------

# Evaluate both initialization strategies independently.
#
# "pretrained":
#     Initialize the encoder using weights from the SSL3D checkpoint.
#
# "random":
#     Initialize the encoder with randomly initialized parameters.
for mode in ("pretrained", "random"):
    print(f"\n=== TESTING {mode.upper()} ===", flush=True)

    # Create the frozen ResEncL encoder using the project utility.
    #
    # The checkpoint path is provided only for pretrained mode.
    # The utility is responsible for loading or initializing the
    # encoder and configuring its frozen state.
    #
    # Move the resulting encoder to the CUDA GPU.
    encoder = make_frozen_encoder(
        mode=mode,
        checkpoint_path=checkpoint if mode == "pretrained" else None,
    ).cuda()

    # -------------------------------------------------------------
    # 3. Validate encoder configuration
    # -------------------------------------------------------------

    # Verify that all encoder parameters have requires_grad=False.
    #
    # This ensures that the encoder is frozen and its weights
    # are not intended to be updated during training.
    assert not any(p.requires_grad for p in encoder.parameters())

    # Verify that the encoder is in evaluation mode.
    #
    # Evaluation mode controls the behavior of layers such as
    # BatchNorm and Dropout, when present in the architecture.
    assert not encoder.training

    # -------------------------------------------------------------
    # 4. Generate a synthetic 3D MRI input
    # -------------------------------------------------------------

    # Generate a random input tensor directly on the CUDA GPU.
    #
    # Shape: (batch, channels, depth, height, width)
    #      = (1, 1, 96, 96, 96).
    #
    # The tensor contains random values and does not represent
    # an actual FeTA or BOBs MRI volume.
    x = torch.randn(1, 1, 96, 96, 96, device="cuda")

    # -------------------------------------------------------------
    # 5. Run the encoder forward pass
    # -------------------------------------------------------------

    # Perform inference without constructing an autograd graph.
    #
    # This avoids storing intermediate values required for
    # backpropagation and reduces memory usage.
    with torch.inference_mode():
        features = encoder(x)

    # -------------------------------------------------------------
    # 6. Validate encoder outputs
    # -------------------------------------------------------------

    # The ResEncL encoder is expected to return six feature maps
    # corresponding to different spatial resolutions.
    assert len(features) == 6

    # Inspect each feature map independently.
    for i, feature in enumerate(features):
        # Verify that all activation values are finite.
        #
        # torch.isfinite() checks that the tensor contains
        # neither NaN nor infinity values.
        assert torch.isfinite(feature).all()

        # Display the feature-map index, shape, and validity.
        print(f"Feature {i}: {tuple(feature.shape)} | finite=True")

    # Report successful execution for the current encoder mode.
    print(f"{mode.upper()} PASSED!", flush=True)

    # -------------------------------------------------------------
    # 7. Release GPU resources
    # -------------------------------------------------------------

    # Delete references to the encoder, input tensor,
    # and generated feature maps.
    del encoder, x, features

    # Release unused cached CUDA memory managed by PyTorch.
    #
    # This helps reduce unnecessary memory retention before
    # initializing the next encoder.
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# 8. Final validation summary
# ---------------------------------------------------------------------

# Reaching this statement means both encoder configurations passed
# all assertions and completed their forward passes successfully.
print("\nBOTH FROZEN ENCODERS PASSED!")

"""
Integration smoke test for SSL3DDataset and frozen ResEncL encoders.

This script verifies that preprocessed FeTA and BOBs MRI patches
can be loaded through the custom SSL3DDataset and passed through
the frozen ResEncL encoder implementation.

Both encoder initialization strategies are evaluated:
    - "pretrained": Encoder initialized with SSL3D pretrained weights.
    - "random": Encoder initialized with random weights.

Main validation steps:
    1. Check CUDA availability and checkpoint existence.
    2. Load one validation MRI/segmentation patch from FeTA.
    3. Load one validation MRI/segmentation patch from BOBs.
    4. Validate image and segmentation tensor properties.
    5. Initialize pretrained and random frozen encoders.
    6. Verify that encoders are in evaluation mode.
    7. Verify that encoder parameters are frozen.
    8. Run forward passes without gradient computation.
    9. Validate multiscale feature-map shapes and channel counts.
   10. Verify that all feature-map values are finite.
   11. Report peak allocated GPU tensor memory.

Expected input tensor:
    Image: (1, 1, P, P, P)
    Mask:  (1, P, P, P)

Where P is a positive multiple of 32.

Expected encoder feature channels:
    (32, 64, 128, 256, 320, 320)

The encoder is expected to produce six multiscale feature maps
with progressively reduced spatial dimensions.

Usage (from the repository root on an Izar GPU compute node):

    apptainer exec --nv --bind "$HOME:$HOME" ~/containers/ssl3d_nnunet.sif \
        python -m scripts.tests.test_ssl3d_integration --patch-size 160

Requirements:
    - A CUDA-compatible GPU.
    - PyTorch with CUDA support.
    - The nnSSL/nnU-Net Apptainer environment.
    - Preprocessed FeTA and BOBs datasets.
    - A compatible SSL3D pretrained checkpoint.
    - src.models.frozen_encoder.make_frozen_encoder.
    - src.datasets.ssl3d_dataset.SSL3DDataset.

Important:
    - Only one validation case is tested per dataset.
    - Both encoder initialization modes are evaluated.
    - No segmentation decoder or segmentation loss is evaluated.
    - No optimizer step or backward pass is performed.
    - No model parameters are updated.
    - No features or checkpoints are saved.
    - Original MRI and segmentation files are never modified.

This is a forward-only integration test, not a segmentation
training or performance evaluation experiment.
"""

import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.frozen_encoder import make_frozen_encoder
from src.datasets.ssl3d_dataset import SSL3DDataset


# ---------------------------------------------------------------------
# Expected ResEncL encoder configuration
# ---------------------------------------------------------------------

# Number of output channels at each of the six encoder feature levels.
#
# The spatial resolution is progressively reduced while the number
# of feature channels generally increases.
CHANNELS = (32, 64, 128, 256, 320, 320)


def test_encoder(mode: str, image: torch.Tensor, checkpoint: Path | None) -> None:
    """
    Run a frozen ResEncL encoder and validate its multiscale outputs.

    Parameters
    ----------
    mode : str
        Encoder initialization strategy.

        Expected values:
            - "pretrained": Load SSL3D pretrained encoder weights.
            - "random": Use randomly initialized encoder weights.

    image : torch.Tensor
        Input MRI batch with shape (B, 1, D, H, W).

        The spatial dimensions must be divisible by 32.

    checkpoint : Path or None
        Path to the pretrained SSL3D checkpoint.

        The checkpoint is passed to make_frozen_encoder only
        when mode="pretrained".

    Returns
    -------
    None
        The function validates encoder outputs and prints results.

    Validation
    ----------
    Encoder:
        - Must be in evaluation mode.
        - All parameters must have requires_grad=False.

    Feature maps:
        - Exactly six feature maps must be produced.
        - Channel counts must match CHANNELS.
        - Spatial dimensions must decrease by powers of two.
        - All activation values must be finite.

    GPU memory:
        - Peak allocated CUDA tensor memory is reported in GiB.

    Notes
    -----
    The forward pass runs under torch.inference_mode().

    GPU memory statistics are reset before inference.

    The function releases encoder and feature references
    and requests garbage collection after validation.
    """

    # -------------------------------------------------------------
    # 1. Initialize the selected frozen encoder
    # -------------------------------------------------------------

    print(f"  --- {mode.upper()} ---", flush=True)

    # Construct either the pretrained or randomly initialized encoder.
    #
    # The checkpoint is supplied only in pretrained mode.
    # Random mode explicitly receives checkpoint_path=None.
    #
    # Transfer the encoder to the CUDA GPU.
    encoder = make_frozen_encoder(
        mode=mode,
        checkpoint_path=checkpoint if mode == "pretrained" else None,
    ).to("cuda")

    # -------------------------------------------------------------
    # 2. Verify encoder evaluation mode and frozen parameters
    # -------------------------------------------------------------

    # Evaluation mode is required for this inference-only test.
    assert not encoder.training, "Encoder must be in eval mode"

    # Verify that no encoder parameter requires gradient computation.
    assert not any(p.requires_grad for p in encoder.parameters()), (
        "Encoder is not frozen"
    )

    # -------------------------------------------------------------
    # 3. Run the forward pass
    # -------------------------------------------------------------

    # Reset peak allocated CUDA memory statistics before inference.
    torch.cuda.reset_peak_memory_stats()

    # Disable autograd and perform the encoder forward pass.
    with torch.inference_mode():
        features = encoder(image)

    # Convert a single tensor output to a list for uniform handling.
    if isinstance(features, torch.Tensor):
        features = [features]

    # -------------------------------------------------------------
    # 4. Validate the number of encoder feature maps
    # -------------------------------------------------------------

    # The ResEncL encoder configuration used in this project
    # is expected to return six feature levels.
    assert len(features) == 6, f"Expected 6 feature maps, got {len(features)}"

    # -------------------------------------------------------------
    # 5. Validate expected spatial dimensions
    # -------------------------------------------------------------

    # Extract the batch size and input spatial dimensions.
    #
    # Input tensor format:
    # (batch, channels, depth, height, width).
    batch_size, _, depth, height, width = image.shape

    # The expected feature-map hierarchy requires spatial
    # dimensions divisible by 32.
    assert all(s % 32 == 0 for s in (depth, height, width)), (
        "This test expects a patch size divisible by 32"
    )

    # -------------------------------------------------------------
    # 6. Validate every encoder feature map
    # -------------------------------------------------------------

    # Check the output of each encoder level.
    for index, (feature, channels) in enumerate(zip(features, CHANNELS)):
        # Spatial downsampling factor at the current feature level.
        #
        # Level 0: factor = 1
        # Level 1: factor = 2
        # Level 2: factor = 4
        # Level 3: factor = 8
        # Level 4: factor = 16
        # Level 5: factor = 32
        factor = 2**index

        # Construct the expected feature-map shape.
        #
        # Batch size remains unchanged.
        # Channels depend on the encoder level.
        # Spatial dimensions are divided by the downsampling factor.
        expected = (
            batch_size,
            channels,
            depth // factor,
            height // factor,
            width // factor,
        )

        # Verify that the actual feature-map shape is correct.
        assert tuple(feature.shape) == expected, (
            f"Feature {index}: expected {expected}, got {tuple(feature.shape)}"
        )

        # Ensure the feature map contains no NaN or infinity values.
        assert bool(torch.isfinite(feature).all()), f"Feature {index} contains NaN/Inf"

        # Display the feature-map dimensions and validation result.
        print(f"    Feature {index}: {tuple(feature.shape)} | finite=True", flush=True)

    # -------------------------------------------------------------
    # 7. Report peak CUDA tensor memory
    # -------------------------------------------------------------

    # Convert peak allocated GPU tensor memory from bytes to GiB.
    #
    # 1 GiB = 1024^3 bytes.
    # This measures PyTorch allocated tensor memory, not total
    # system-wide GPU memory consumption.
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)

    print(
        f"  {mode.upper()} PASSED! | peak GPU tensor memory: {peak_gib:.2f} GiB",
        flush=True,
    )

    # -------------------------------------------------------------
    # 8. Release resources
    # -------------------------------------------------------------

    # Remove references to the encoder and generated feature maps.
    del features, encoder

    # Request Python garbage collection.
    gc.collect()

    # Release unused cached CUDA memory managed by PyTorch.
    torch.cuda.empty_cache()


def main() -> None:
    """
    Run the SSL3D dataset-to-encoder integration tests.

    The procedure:
        1. Parse test configuration.
        2. Validate patch size, CUDA availability, and checkpoint.
        3. Load one validation batch from FeTA.
        4. Run pretrained and random encoders.
        5. Repeat the same process for BOBs.
        6. Release resources between datasets.

    No segmentation training is performed.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable.

    FileNotFoundError
        If the pretrained checkpoint cannot be found.

    AssertionError
        If tensor shapes, data types, labels, encoder configuration,
        or feature-map properties do not match expectations.
    """

    # -------------------------------------------------------------
    # 1. Parse command-line arguments
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)

    # Root directory containing the preprocessed FeTA and BOBs datasets.
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path.home() / "brain-mri-foundation-models/data/nnUNet_preprocessed",
    )

    # Location of the pretrained SSL3D encoder checkpoint.
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "ssl3d_checkpoints/checkpoint_best.pth",
    )

    # Cubic input patch size.
    parser.add_argument("--patch-size", type=int, default=160)

    args = parser.parse_args()

    # -------------------------------------------------------------
    # 2. Validate the test environment
    # -------------------------------------------------------------

    # The input patch size must be a positive multiple of 32
    # to match the encoder's expected downsampling hierarchy.
    if args.patch_size <= 0 or args.patch_size % 32 != 0:
        parser.error("--patch-size must be a positive multiple of 32 (e.g. 160)")

    # Ensure that a CUDA GPU is available for model inference.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable: run inside Apptainer --nv on an allocated GPU node"
        )

    # Verify that the pretrained checkpoint exists.
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    # Display the GPU and input configuration.
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Patch size: {args.patch_size}^3 | batch size: 1", flush=True)

    # -------------------------------------------------------------
    # 3. Test FeTA and BOBs independently
    # -------------------------------------------------------------

    for name in ("feta", "bobs"):
        print(f"\n=== {name.upper()} VALIDATION ===", flush=True)

        # Initialize the validation dataset.
        #
        # Validation uses deterministic center-patch sampling
        # by default in SSL3DDataset.
        dataset = SSL3DDataset(
            preprocessed_root=args.preprocessed_root,
            dataset=name,
            split="val",
            patch_size=args.patch_size,
        )

        # Create a DataLoader and retrieve the first validation case.
        #
        # batch_size=1:
        #     One 3D MRI volume patch is processed at a time.
        #
        # shuffle=False:
        #     Preserve the deterministic case ordering.
        #
        # num_workers=0:
        #     Load data in the main Python process.
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        batch = next(iter(loader))

        # Extract the MRI and segmentation tensors.
        image, mask = batch["image"], batch["mask"]

        # ---------------------------------------------------------
        # 4. Validate MRI and segmentation tensors
        # ---------------------------------------------------------

        # Expected MRI batch shape:
        # (batch=1, channels=1, depth=P, height=P, width=P).
        assert image.shape == (1, 1, args.patch_size, args.patch_size, args.patch_size)

        # Expected segmentation batch shape:
        # (batch=1, depth=P, height=P, width=P).
        assert mask.shape == (1, args.patch_size, args.patch_size, args.patch_size)

        # MRI intensities must be float32.
        # Segmentation labels must be int64.
        assert image.dtype == torch.float32 and mask.dtype == torch.int64

        # Verify that both tensors contain only finite values.
        assert bool(torch.isfinite(image).all()) and bool(torch.isfinite(mask).all())

        # Check the supported segmentation label convention.
        #
        # -1: Ignore label
        #  0: Background
        # 1..7: Foreground classes
        assert set(torch.unique(mask).tolist()).issubset(set(range(-1, 8)))

        # Display the selected case and tensor dimensions.
        print(f"  Case: {batch['case_id'][0]}", flush=True)
        print(f"  Image: {tuple(image.shape)} | mask: {tuple(mask.shape)}", flush=True)

        # ---------------------------------------------------------
        # 5. Transfer MRI data to GPU
        # ---------------------------------------------------------

        # Only the image tensor is required for encoder inference.
        # The mask remains on CPU because segmentation prediction
        # and evaluation are not performed in this test.
        image = image.to("cuda")

        # ---------------------------------------------------------
        # 6. Test pretrained and random encoder modes
        # ---------------------------------------------------------

        # Evaluate both encoder initialization strategies
        # using the same MRI patch.
        for mode in ("pretrained", "random"):
            test_encoder(mode, image, args.checkpoint)

        # ---------------------------------------------------------
        # 7. Release resources before the next dataset
        # ---------------------------------------------------------

        # Remove references to the current dataset, batch,
        # image, mask, and DataLoader.
        del image, mask, batch, loader, dataset

        # Request Python garbage collection.
        gc.collect()

        # Release unused PyTorch CUDA cached memory.
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 8. Report final integration-test results
    # -------------------------------------------------------------

    print("\nALL SSL3D INTEGRATION TESTS PASSED!", flush=True)

    # Clarify the scope of the validation.
    # This test does not verify segmentation heads or training behavior.
    print(
        "Note: forward-only validation; segmentation head/training not tested.",
        flush=True,
    )


if __name__ == "__main__":
    main()

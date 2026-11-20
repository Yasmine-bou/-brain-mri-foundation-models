"""Build a frozen SSL3D ResEncL encoder (pretrained or random).

Run in the project's nnSSL environment, where ``nnunetv2`` is installed.
The returned encoder stays on CPU until moved to a device by the caller.
Its forward pass returns six multiscale feature maps for a 3D input.
"""

from pathlib import Path
from typing import Literal

import torch
from torch import nn

from nnunetv2.utilities.get_network_via_name import get_network_from_name


EncoderMode = Literal["pretrained", "random"]


def make_frozen_encoder(
    mode: EncoderMode,
    checkpoint_path: str | Path | None = None,
    seed: int = 42,
) -> nn.Module:
    """Create the same ResEncL architecture with either initialization.

    Args:
        mode: ``"pretrained"`` loads SSL3D encoder weights;
            ``"random"`` uses seeded random initialization.
        checkpoint_path: Required only for pretrained mode; ignored for random.
        seed: Random initialization seed for random mode.

    Returns:
        Encoder with ``requires_grad=False`` for every parameter and in eval mode.
        Move it to a GPU with ``encoder.to('cuda')`` when needed.
    """
    if mode not in ("pretrained", "random"):
        raise ValueError("mode must be 'pretrained' or 'random'")

    if mode == "pretrained":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for pretrained mode")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
    elif checkpoint_path is not None:
        raise ValueError("Do not supply checkpoint_path for random mode")

    # Use the exact architecture and flags validated in the Izar smoke tests.
    # Isolate seeded random initialization from the caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        if mode == "random":
            torch.manual_seed(seed)
        network = get_network_from_name(
            arch_class_name="ResEncL",
            input_channels=1,
            output_channels=8,
            allow_init=(mode == "random"),
            deep_supervision=False,
        )
    encoder = network.encoder

    if mode == "pretrained":
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        weights = checkpoint["network_weights"]
        encoder_weights = {
            key.removeprefix("encoder."): value
            for key, value in weights.items()
            if key.startswith(("encoder.stem.", "encoder.stages."))
        }
        # strict=True fails if a required encoder key is absent or mismatched.
        encoder.load_state_dict(encoder_weights, strict=True)
        print(f"Loaded {len(encoder_weights)} SSL3D encoder state entries")
        del checkpoint, weights, encoder_weights

    encoder.requires_grad_(False)
    encoder.eval()
    return encoder

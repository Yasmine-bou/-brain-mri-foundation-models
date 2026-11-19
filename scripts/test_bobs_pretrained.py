
from pathlib import Path

import blosc2
import numpy as np
import torch

from nnunetv2.utilities.get_network_via_name import get_network_from_name


DATA_DIR = Path(
    "/home/bouattou/brain-mri-foundation-models/"
    "data/nnUNet_preprocessed/Dataset102_BOBs/SSL3D_val"
)

CKPT_PATH = Path("/home/bouattou/ssl3d_checkpoints/checkpoint_best.pth")
PATCH_SIZE = 96


# 1. Load one preprocessed BOBs MRI
print("=== 1. LOADING BOBS MRI ===", flush=True)

mri_paths = sorted(
    p for p in DATA_DIR.glob("*.b2nd")
    if not p.name.endswith("_seg.b2nd")
)

assert mri_paths, f"No MRI found in {DATA_DIR}"

mri_path = mri_paths[0]
print("Selected BOBs MRI:", mri_path.name, flush=True)

mri = np.asarray(blosc2.open(str(mri_path), mode="r")[:])

print("MRI shape:", mri.shape, flush=True)

assert mri.ndim == 4 and mri.shape[0] == 1


# 2. Center crop or pad to 96 x 96 x 96
patch = np.zeros(
    (1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
    dtype=np.float32,
)

starts = [max(0, (s - PATCH_SIZE) // 2) for s in mri.shape[1:]]

cropped = mri[
    :,
    starts[0]:starts[0] + PATCH_SIZE,
    starts[1]:starts[1] + PATCH_SIZE,
    starts[2]:starts[2] + PATCH_SIZE,
]

d, h, w = cropped.shape[1:]
patch[:, :d, :h, :w] = cropped

x = torch.from_numpy(patch).unsqueeze(0).to("cuda")

print("Input tensor:", tuple(x.shape), flush=True)


# 3. Build official ResEncL
print("\n=== 2. BUILDING RESENCL ===", flush=True)

network = get_network_from_name(
    arch_class_name="ResEncL",
    input_channels=1,
    output_channels=8,
    allow_init=False,
    deep_supervision=False,
)

encoder = network.encoder


# 4. Load pretrained SSL3D encoder weights
print("\n=== 3. LOADING PRETRAINED WEIGHTS ===", flush=True)

checkpoint = torch.load(
    CKPT_PATH,
    map_location="cpu",
    weights_only=True,
)

weights = checkpoint["network_weights"]

encoder_weights = {
    key.removeprefix("encoder."): value
    for key, value in weights.items()
    if key.startswith(("encoder.stem.", "encoder.stages."))
}

encoder.load_state_dict(encoder_weights, strict=True)

print("Pretrained encoder loaded successfully!", flush=True)
print("Loaded encoder entries:", len(encoder_weights), flush=True)

del checkpoint, weights, encoder_weights, network


# 5. Freeze encoder
encoder.requires_grad_(False)
encoder = encoder.cuda()
encoder.eval()

assert not any(p.requires_grad for p in encoder.parameters())

print("Encoder frozen successfully!", flush=True)


# 6. Forward pass
print("\n=== 4. FORWARD PASS ===", flush=True)

with torch.inference_mode():
    features = encoder(x)

if isinstance(features, torch.Tensor):
    features = [features]

assert len(features) == 6, f"Expected 6 feature maps, got {len(features)}"

for i, feature in enumerate(features):
    is_finite = bool(torch.isfinite(feature).all())

    print(
        f"Feature map {i}: {tuple(feature.shape)} "
        f"| finite={is_finite}",
        flush=True,
    )

    assert is_finite, f"Non-finite values in feature map {i}"

print("\nBOBS PRETRAINED FORWARD PASS SUCCESSFUL!", flush=True)

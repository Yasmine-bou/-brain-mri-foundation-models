from pathlib import Path

import blosc2
import numpy as np
import torch

from nnunetv2.utilities.get_network_via_name import get_network_from_name


DATA_DIR = Path(
    "/home/bouattou/brain-mri-foundation-models/"
    "data/nnUNet_preprocessed/Dataset101_FeTA/SSL3D_val"
)
CKPT_PATH = Path("/home/bouattou/ssl3d_checkpoints/checkpoint_best.pth")
PATCH_SIZE = 96


# 1. Load one preprocessed FeTA MRI
print("=== 1. LOADING FETA MRI ===", flush=True)

mri_path = DATA_DIR / "sub-006.b2nd"
mri = np.asarray(blosc2.open(str(mri_path), mode="r")[:])

print("MRI shape:", mri.shape, flush=True)

# Expected: (1, D, H, W)
assert mri.ndim == 4 and mri.shape[0] == 1

# Center crop or pad to a 96 x 96 x 96 patch
patch = np.zeros((1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

for_axis = []
for size in mri.shape[1:]:
    start = max(0, (size - PATCH_SIZE) // 2)
    for_axis.append(start)

cropped = mri[
    :,
    for_axis[0]:for_axis[0] + PATCH_SIZE,
    for_axis[1]:for_axis[1] + PATCH_SIZE,
    for_axis[2]:for_axis[2] + PATCH_SIZE,
]

d, h, w = cropped.shape[1:]
patch[:, :d, :h, :w] = cropped

x = torch.from_numpy(patch).unsqueeze(0).to("cuda")

print("Input tensor:", tuple(x.shape), flush=True)


# 2. Build official ResEncL
print("\n=== 2. BUILDING RESENCL ===", flush=True)

network = get_network_from_name(
    arch_class_name="ResEncL",
    input_channels=1,
    output_channels=8,
    allow_init=False,
    deep_supervision=False,
)

encoder = network.encoder


# 3. Load pretrained encoder weights
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

# Strict loading: every expected encoder weight must match.
encoder.load_state_dict(encoder_weights, strict=True)

print("Pretrained encoder loaded successfully!", flush=True)
print("Loaded encoder entries:", len(encoder_weights), flush=True)

del checkpoint, weights, encoder_weights, network

# Freeze encoder
encoder.requires_grad_(False)
encoder = encoder.cuda()
encoder.eval()


# 4. Forward pass
print("\n=== 4. FORWARD PASS ===", flush=True)

with torch.inference_mode():
    features = encoder(x)

if isinstance(features, torch.Tensor):
    features = [features]

for i, feature in enumerate(features):
    print(
        f"Feature map {i}: {tuple(feature.shape)} "
        f"| finite={bool(torch.isfinite(feature).all())}",
        flush=True,
    )

print("\nFORWARD PASS SUCCESSFUL!", flush=True)

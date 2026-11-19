
from pathlib import Path

import blosc2
import numpy as np
import torch

from nnunetv2.utilities.get_network_via_name import get_network_from_name

DATA_DIR = Path(
    "/home/bouattou/brain-mri-foundation-models/"
    "data/nnUNet_preprocessed/Dataset101_FeTA/SSL3D_val"
)

PATCH_SIZE = 96

# 1. Load the same FeTA MRI
print("=== 1. LOADING FETA MRI ===", flush=True)

mri = np.asarray(
    blosc2.open(str(DATA_DIR / "sub-006.b2nd"), mode="r")[:]
)

assert mri.ndim == 4 and mri.shape[0] == 1

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


# 2. Build randomly initialized ResEncL
print("\n=== 2. BUILDING RANDOM RESENCL ===", flush=True)

torch.manual_seed(42)

network = get_network_from_name(
    arch_class_name="ResEncL",
    input_channels=1,
    output_channels=8,
    allow_init=True,
    deep_supervision=False,
)

# IMPORTANT: no pretrained checkpoint is loaded.
encoder = network.encoder

encoder.requires_grad_(False)
encoder = encoder.cuda()
encoder.eval()

print("Random encoder initialized and frozen!", flush=True)


# 3. Forward pass
print("\n=== 3. FORWARD PASS ===", flush=True)

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

print("\nRANDOM FORWARD PASS SUCCESSFUL!", flush=True)


import torch
from pathlib import Path
from src.frozen_encoder import make_frozen_encoder

checkpoint = Path.home() / "ssl3d_checkpoints/checkpoint_best.pth"

for mode in ("pretrained", "random"):
    print(f"\n=== TESTING {mode.upper()} ===", flush=True)

    encoder = make_frozen_encoder(
        mode=mode,
        checkpoint_path=checkpoint if mode == "pretrained" else None,
    ).cuda()

    assert not any(p.requires_grad for p in encoder.parameters())
    assert not encoder.training

    x = torch.randn(1, 1, 96, 96, 96, device="cuda")

    with torch.inference_mode():
        features = encoder(x)

    assert len(features) == 6

    for i, feature in enumerate(features):
        assert torch.isfinite(feature).all()
        print(f"Feature {i}: {tuple(feature.shape)} | finite=True")

    print(f"{mode.upper()} PASSED!", flush=True)

    del encoder, x, features
    torch.cuda.empty_cache()

print("\nBOTH FROZEN ENCODERS PASSED!")

"""Tranches (image + segmentation en couleur) d'un cas et de ses copies augmentées.

Usage (cluster, conteneur) :
    python -I check_augmentation.py <dossier Spacing__...> sub-002 sub-002_aug3 sortie.png
"""
import sys

import blosc2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d, names, out = sys.argv[1], sys.argv[2:-1], sys.argv[-1]
fig, axes = plt.subplots(len(names), 3, figsize=(12, 4 * len(names)))
axes = np.atleast_2d(axes)
for row, n in zip(axes, names):
    img = blosc2.open(f"{d}/{n}.b2nd")[:][0]
    seg = blosc2.open(f"{d}/{n}_seg.b2nd")[:][0]
    for ax, axis in zip(row, range(3)):
        i = img.shape[axis] // 2
        im = np.take(img, i, axis=axis).T
        sg = np.take(seg, i, axis=axis).T
        ax.imshow(im, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(sg <= 0, sg), cmap="tab10", vmin=0, vmax=9,
                  alpha=0.45, origin="lower", interpolation="nearest")
        ax.set_title(f"{n} - coupe {axis} ({i})")
        ax.axis("off")
plt.tight_layout()
plt.savefig(out, dpi=100)
print("OK", out)
"""PCA des latents (vecteur 'pooled') de l'encodeur fine-tuné, un graphique par dataset.

Lit data/latents/<dataset>/{train,val,test}/*.npz (dossier ignoré par Git).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2] / "data" / "latents"
MARKERS = {"train": "o", "val": "s", "test": "^"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, ds, title in zip(axes, ["Dataset101_FeTA", "Dataset102_BOBs"], ["FeTA", "BOBs"]):
    X, split, age = [], [], []
    for s in MARKERS:
        for f in sorted((ROOT / ds / s).glob("*.npz")):
            X.append(np.load(f)["pooled"])
            split.append(s)
            m = re.search(r"ses-(\d+)mo", f.name)
            age.append(int(m.group(1)) if m else np.nan)
    X, split, age = np.array(X), np.array(split), np.array(age)
    pca = PCA(n_components=2)
    P = pca.fit_transform(StandardScaler().fit_transform(X))
    sc = None
    for s in MARKERS:
        i = split == s
        if title == "BOBs":
            sc = ax.scatter(P[i, 0], P[i, 1], c=age[i], cmap="viridis",
                            vmin=np.nanmin(age), vmax=np.nanmax(age),
                            marker=MARKERS[s], label=s, edgecolor="k", linewidth=0.4)
        else:
            ax.scatter(P[i, 0], P[i, 1], marker=MARKERS[s], label=s,
                       edgecolor="k", linewidth=0.4)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="âge (mois)")
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.0f} %)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0f} %)")
    ax.set_title(f"{title} : {len(X)} cas")
    ax.legend(title="groupe")

out = ROOT / "pca_latents.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("SAUVEGARDE:", out)
plt.show()
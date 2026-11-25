"""Augmentation hors-ligne d'un dataset nnUNet pré-traité (images + segmentations).

Pour chaque cas d'ENTRAÎNEMENT (jamais val/test), crée N copies avec :
  - rotation aléatoire (+-max_rot degrés autour de chaque axe)
  - zoom aléatoire (isotrope)
  - luminosité / contraste / bruit gaussien léger (image seulement)
La même transformation géométrique est appliquée à l'image (interpolation
linéaire) et à la segmentation (plus proche voisin). Les copies sont ajoutées
à la liste d'entraînement de splits_final.json (ancien fichier sauvegardé en .bak).

Utilisation (sur le cluster, dans le conteneur) :
    python -I augment_nnunet_dataset.py \
        --dataset-dir ~/nnunet/nnUNet_preprocessed/Dataset103_FeTA_aug --n-copies 5
"""
import argparse
import json
import pickle
import shutil
import zlib
from pathlib import Path

import blosc2
import numpy as np
from scipy.ndimage import affine_transform


def rotation_matrix(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def corner_value(vol):
    """Valeur de fond : médiane des 8 coins du volume."""
    corners = [vol[i, j, k] for i in (0, -1) for j in (0, -1) for k in (0, -1)]
    return float(np.median(corners))


def sample_locations(seg4d, keys, seed=1234):
    """Même logique que nnUNet : positions échantillonnées de chaque label."""
    rng = np.random.RandomState(seed)
    out = {}
    for c in keys:
        mask = np.isin(seg4d, list(c)) if isinstance(c, (tuple, list)) else seg4d == c
        locs = np.argwhere(mask)
        if len(locs) == 0:
            out[c] = []
            continue
        n = min(10000, len(locs))
        n = max(n, int(np.ceil(len(locs) * 0.01)))
        out[c] = locs[rng.choice(len(locs), n, replace=False)]
    return out


def save_b2nd(arr, path, like):
    arr = np.ascontiguousarray(arr)
    kw = dict(urlpath=str(path), chunks=like.chunks, blocks=like.blocks,
              cparams={"codec": blosc2.Codec.ZSTD, "clevel": 8})
    try:
        blosc2.asarray(arr, mmap_mode="w+", **kw)
    except TypeError:
        blosc2.asarray(arr, **kw)
    check = blosc2.open(str(path))
    assert tuple(check.shape) == arr.shape, f"relecture incorrecte : {path}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--n-copies", type=int, default=5)
    p.add_argument("--max-rot", type=float, default=15.0, help="degrés")
    p.add_argument("--scale-min", type=float, default=0.85)
    p.add_argument("--scale-max", type=float, default=1.15)
    p.add_argument("--pad", type=int, default=20, help="marge (voxels) avant rotation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="seulement N cas (test)")
    a = p.parse_args()

    root = Path(a.dataset_dir).expanduser()
    if "aug" not in root.name:
        raise SystemExit("Refus : le nom du dossier doit contenir 'aug' "
                         "(on ne modifie jamais le dataset d'origine).")
    plans = [d for d in root.glob("Spacing__*") if d.is_dir()]
    if len(plans) != 1:
        raise SystemExit(f"Dossier Spacing introuvable ou ambigu : {plans}")
    plan = plans[0]

    splits_path = root / "splits_final.json"
    splits = json.loads(splits_path.read_text())
    train_ids = [c for c in splits[0]["train"] if "_aug" not in c]
    if a.limit:
        train_ids = train_ids[: a.limit]

    log, new_ids = {}, []
    for cid in train_ids:
        img_nd = blosc2.open(str(plan / f"{cid}.b2nd"))
        seg_nd = blosc2.open(str(plan / f"{cid}_seg.b2nd"))
        img, seg = img_nd[:][0], seg_nd[:][0]  # (X, Y, Z)
        with open(plan / f"{cid}.pkl", "rb") as f:
            props = pickle.load(f)
        bg_img, bg_seg = corner_value(img), int(round(corner_value(seg)))
        w = a.pad
        img_p = np.pad(img, w, mode="constant", constant_values=bg_img)
        seg_p = np.pad(seg, w, mode="constant", constant_values=bg_seg)
        centre = (np.array(img_p.shape) - 1) / 2

        for k in range(1, a.n_copies + 1):
            new = f"{cid}_aug{k}"
            new_ids.append(new)
            if (plan / f"{new}.b2nd").exists():
                print("déjà là, on saute :", new)
                continue
            rng = np.random.default_rng([a.seed, k, zlib.crc32(cid.encode())])
            ang = np.deg2rad(rng.uniform(-a.max_rot, a.max_rot, 3))
            scale = rng.uniform(a.scale_min, a.scale_max)
            m = rotation_matrix(*ang).T / scale
            off = centre - m @ centre
            im_o = affine_transform(img_p, m, offset=off, order=1,
                                    mode="constant", cval=bg_img)
            sg_o = affine_transform(seg_p, m, offset=off, order=0,
                                    mode="constant", cval=bg_seg).astype(seg.dtype)
            gain, shift = rng.uniform(0.9, 1.1), rng.uniform(-0.1, 0.1)
            sigma = rng.uniform(0.0, 0.05)
            brain = np.abs(im_o - bg_img) > 1e-6
            im_o = np.where(brain, im_o * gain + shift
                            + rng.normal(0, sigma, im_o.shape), im_o).astype(np.float32)

            nprops = dict(props)
            nprops["class_locations"] = sample_locations(
                sg_o[None], list(props["class_locations"].keys()))
            save_b2nd(im_o[None], plan / f"{new}.b2nd", img_nd)
            save_b2nd(sg_o[None], plan / f"{new}_seg.b2nd", seg_nd)
            with open(plan / f"{new}.pkl", "wb") as f:
                pickle.dump(nprops, f)

            fg_in, fg_out = int((seg_p > 0).sum()), int((sg_o > 0).sum())
            log[new] = dict(angles_deg=[round(float(x), 2) for x in np.rad2deg(ang)],
                            scale=round(float(scale), 3), gain=round(float(gain), 3),
                            shift=round(float(shift), 3), sigma=round(float(sigma), 3),
                            shape=list(im_o.shape), fg_voxels_ratio=round(fg_out / max(fg_in, 1), 3))
            print(new, log[new])

    backup = splits_path.with_name("splits_final.json.bak")
    if not backup.exists():
        shutil.copy2(splits_path, backup)
    splits[0]["train"] = list(dict.fromkeys(splits[0]["train"] + new_ids))
    splits_path.write_text(json.dumps(splits, indent=2))
    (root / "augmentation_log.json").write_text(json.dumps(log, indent=2))
    print(f"Terminé : {len(new_ids)} cas augmentés ; train = {len(splits[0]['train'])} cas.")


if __name__ == "__main__":
    main()
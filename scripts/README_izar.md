# FeTA frozen encoder smoke test on EPFL Izar

This smoke test uses the **same official ResEncL architecture** for two frozen encoders: (A) SSL3D pretrained weights and (B) reproducibly random weights. It tests **3 validation MRIs**, returns six multiscale feature maps per MRI, and **does not train or save features**.

## Already prepared on Izar

- Preprocessed FeTA: `~/brain-mri-foundation-models/data/nnUNet_preprocessed/Dataset101_FeTA/`
- Pretrained checkpoint: `~/ssl3d_checkpoints/checkpoint_best.pth`
- Apptainer image: `~/containers/ssl3d_nnunet.sif` (built from `petermcgor/nnunetv2:1.1.0-nnssl`)

## 1. Upload the script (Windows PowerShell, from repo root)

```powershell
scp scripts/test_feta_encoders.py bouattou@izar.hpc.epfl.ch:~/test_feta_encoders.py
```

Alternatively, clone/pull the GitHub repo on Izar and run the script from the repo; no `scp` needed in that case.

## 2. Request a GPU (SSH session on `izar1`)

```bash
srun --qos=debug --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --pty bash -l
nvidia-smi
```

Do not run the GPU computation on the login node. A debug job is temporary; request a new session when it expires.

## 3. Run both pretrained and random forward passes (GPU node)

```bash
apptainer exec --nv --bind "$HOME:$HOME" \
  ~/containers/ssl3d_nnunet.sif \
  python ~/test_feta_encoders.py \
  --data-dir ~/brain-mri-foundation-models/data/nnUNet_preprocessed/Dataset101_FeTA/SSL3D_val \
  --checkpoint ~/ssl3d_checkpoints/checkpoint_best.pth \
  --num-cases 3 --patch-size 96
```

Expected: 3 MRI/mask pairs are read for each of the two modes, 6 finite feature maps per case with the same shapes, and `SUCCESS: both frozen encoders passed the FeTA smoke test.`

## Notes for segmentation-head implementation

- `make_frozen_encoder("pretrained", checkpoint_path)` or `make_frozen_encoder("random", checkpoint_path)` returns a frozen `torch.nn.Module`.
- `encoder(x)` returns six multiscale tensors (one per encoder stage). The head can consume these directly in the training loop.
- The centered 96³ patches in this script are **only for a smoke test**. Proper random/foreground patch sampling, label handling (including ignore label `-1`), batching, validation, and native-resolution evaluation must be designed for training.
- Dataset files, SIF image, and pretrained checkpoint remain outside Git. Commit the Python script and this guide only.


scp scripts/test_bobs_pretrained.py scripts/test_bobs_random.py bouattou@izar.hpc.epfl.ch:~/
apptainer exec --nv --bind "$HOME:$HOME" \
  ~/containers/ssl3d_nnunet.sif \
  python ~/test_bobs_pretrained.py
  or 
apptainer exec --nv --bind "$HOME:$HOME" \
~/containers/ssl3d_nnunet.sif \
python ~/test_bobs_random.py
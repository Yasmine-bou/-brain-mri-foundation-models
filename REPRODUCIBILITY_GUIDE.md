# Reproducing the Brain MRI Encoder Tests

This guide explains how to run the current FeTA and BOBs frozen-encoder checks. It does **not** reproduce segmentation training or final evaluation yet.

## 1. What you need

- Access to an EPFL Izar account with GPU allocation rights.
- Access to the FeTA and BOBs datasets, preprocessed with the project's nnU-Net pipeline.
- The SSL3D pretrained checkpoint (`checkpoint_best.pth`).
- The Apptainer image (`ssl3d_nnunet.sif`).

The commands below use the team's **existing Izar setup**. A new user must provide their own authorized data, checkpoint, and container. These are not stored in Git.

## 2. Run the encoder checks on Izar

Run these commands in an Izar terminal. Replace `<your-username>` with your own account name. The repository must be accessible to you on GitHub.

| Task | Command | Check |
| --- | --- | --- |
| Clone the code (first time only) | `git clone https://github.com/Yasmine-bou/-brain-mri-foundation-models.git brain-mri-foundation-models-git` | A new project folder exists. The repo is private so you'll need GitHub authentication is required. |
| Open the project | `cd ~/brain-mri-foundation-models-git` | You are at the project root. |
| Update the code | `git pull --ff-only origin main` | The current `main` branch is downloaded. |
| Record the code version | `git rev-parse HEAD` | Save the printed commit hash with your results. |
| Check the scripts | `ls scripts/tests` | `test_ssl3d_integration.py` is listed. |
| Check the input data | `ls data/nnUNet_preprocessed` | `Dataset101_FeTA` and `Dataset102_BOBs` are listed. |
| Check the checkpoint | `ls -lh ~/ssl3d_checkpoints/checkpoint_best.pth` | The file exists. |
| Check the container | `ls -lh ~/containers/ssl3d_nnunet.sif` | The file exists. |
| Request one GPU | `srun --qos=debug --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:30:00 --pty bash -l` | A shell opens on a GPU node. |
| Check GPU access | `nvidia-smi` | A GPU is listed. |
| Return to the project | `cd ~/brain-mri-foundation-models-git` | You are at the project root. |
| Run the encoder test | `apptainer exec --nv --bind "$HOME:$HOME" ~/containers/ssl3d_nnunet.sif python -m scripts.tests.test_ssl3d_integration --patch-size 160` | Four tests should succeed: FeTA/BOBs × pretrained/random. Each encoder returns six feature maps. |
| Release the GPU session | `exit` | You return to the previous shell. |

**Data path note:** On the original team account, the preprocessed datasets live in `~/brain-mri-foundation-models/data/nnUNet_preprocessed`. If the new project does not already have `data/nnUNet_preprocessed`, the following command links those *existing* files without copying them:

```bash
ln -s ~/brain-mri-foundation-models/data/nnUNet_preprocessed ~/brain-mri-foundation-models-git/data/nnUNet_preprocessed
```

This shortcut is specific to that Izar account; it is **not** a data-download or preprocessing step. Do not use it if the destination already exists.

## 3. Run the preprocessing checks

On a machine where Docker and the project's preprocessed datasets are already available, start Docker and run from the repository root:

| Task | Command | Expected result on the prepared team environment |
| --- | --- | --- |
| Check FeTA | `python -m scripts.run_pipeline validate --dataset feta` | 80/80 cases validated. |
| Check BOBs | `python -m scripts.run_pipeline validate --dataset bobs` | 71/71 sessions validated. |
| Check the data loaders | `python -m scripts.run_pipeline test` | All six dataset-split loader tests pass. |

These commands have been tested on the prepared Windows/Docker environment. They are not a complete installation procedure.

## 4. What is still needed for a clean-room reproduction

The following steps need tested, portable instructions before someone can reproduce the full project from an empty machine:

- [ ] Provide official dataset access instructions and the exact expected FeTA/BOBs directory layout.
- [ ] Document the SSL3D checkpoint source, version, and expected file location.
- [ ] Provide a reproducible Docker/Apptainer build with pinned dependencies.
- [ ] Document the exact raw-to-nnU-Net conversion and preprocessing commands for both datasets.
- [ ] Explain how to use and verify the committed train/validation/test split files.
- [ ] Add the team's decoder training and evaluation commands, configurations, seeds, and output paths.
- [ ] Re-run the GPU test with the current `main` commit and record the result.

**A successful encoder check is not a segmentation experiment.** Save the commit hash, dataset/split, encoder type, checkpoint version, command, and output logs for each run.

# Evaluating the Benefits and Limits of Brain Foundation Models in Developmental MRI

**EPFL — CS-433 Machine Learning | Fall 2026**

Machine Learning for Science (ML4Science) project conducted in collaboration with the **Medical Image Analysis Lab (MIAL)**, affiliated with the University of Lausanne (UNIL) and Lausanne University Hospital (CHUV).

---

## 1. Project Overview

Pretrained brain MRI foundation models have shown promising results across various medical imaging tasks. However, their effectiveness for fetal and infant brain tissue segmentation remains insufficiently explored.

This project investigates whether pretrained brain MRI models provide meaningful advantages over models trained from scratch, particularly when limited annotated data is available.

Our main research question is:

**Under which conditions does pretraining improve developmental brain MRI segmentation performance and reduce the need for labelled training data?**

A particular challenge is the difference in anatomical structures, MRI contrast, and annotation protocols between fetal and infant datasets, which complicates direct comparisons and model transfer.

## 2. Research Objectives

We aim to:

- Compare segmentation models trained from scratch with models initialized using pretrained representations.
- Investigate different adaptation strategies, including frozen pretrained encoders and full fine-tuning.
- Evaluate how segmentation performance changes with the amount of labelled training data.
- Assess segmentation quality using Dice Similarity Coefficient (DSC), 95th-percentile Hausdorff Distance (HD95), and Volumetric Similarity (VS), following the evaluation principles described in the FeTA benchmark.
- Investigate the impact of developmental differences, MRI characteristics, and dataset annotation protocols.
- Establish a fair and reproducible experimental pipeline.

## 3. Datasets

We consider two publicly available developmental brain MRI datasets.

### 3.1 FeTA — Fetal Tissue Annotation Dataset

The FeTA dataset contains T2-weighted fetal brain MRI volumes and corresponding manually annotated multi-tissue segmentation masks.

The segmentation includes seven brain tissue categories, in addition to the background:

1. External cerebrospinal fluid (eCSF)
2. Grey matter
3. White matter
4. Ventricles
5. Cerebellum
6. Deep grey matter
7. Brainstem

The dataset includes fetal brains across different gestational ages and contains both neurotypical and pathological cases.

**Reference:** Payette et al. (2021).

**Dataset website:** https://fetachallenge.github.io/

### 3.2 Baby Open Brains (BOBs)

The Baby Open Brains dataset provides expert-reviewed infant brain MRI segmentation annotations.

The dataset includes:

- 71 MRI sessions from 51 participants.
- Infants aged approximately 1–9 months.
- T1-weighted (T1w) and T2-weighted (T2w) MRI images.
- Manually curated segmentation masks covering cortical and subcortical brain structures.

The dataset captures significant developmental changes in brain anatomy, tissue contrast, and myelination during early infancy.

**Reference:** Feczko et al. (2025).

**Dataset DOI:** https://doi.org/10.1038/s41597-025-05404-y

### 3.3 Dataset Compatibility

FeTA and BOBs differ in developmental stage, MRI characteristics, and segmentation annotation protocols.

Consequently, their original segmentation labels cannot be assumed to be directly interchangeable.

Dataset-specific label definitions and any necessary label harmonization will be documented before cross-dataset comparisons.

Evaluation will account for these differences to avoid misleading comparisons between fetal and infant segmentation performance.

For BOBs, particular attention will also be given to participant-level data splitting, since multiple MRI sessions may belong to the same infant.

**Data availability:** MRI datasets are not stored in this repository. Please refer to [`data/README.md`](data/README.md) for dataset information and setup instructions.

## 4. Foundation Model

This project investigates the transferability of pretrained brain MRI representations, with particular interest in the foundation models presented by Gordaliza et al. (2026).

Their work introduces brain MRI foundation model approaches based on convolutional U-Net architectures, incorporating anatomical priors and neuroimaging domain knowledge.

The models achieved first-place results in tracks of the SSL3D and FOMO25 brain MRI foundation model challenges at MICCAI 2025.

**Reference:** Gordaliza et al. (2026).

**Paper:** https://arxiv.org/abs/2601.13166

**Official GitHub repository:** https://github.com/jbanusco/BrainFM4Challenges

The specific pretrained checkpoint, architecture, and initialization procedure used in our experiments will be documented once the experimental configuration has been finalized.

## 5. Methodology

We plan to investigate three main training strategies:

| Strategy | Description |
|---|---|
| Training from scratch | Train a segmentation network using randomly initialized weights. |
| Frozen pretrained encoder | Use a pretrained encoder as a fixed feature extractor and train a segmentation head or decoder. |
| Fine-tuning | Initialize the network with pretrained weights and adapt its parameters to the target segmentation task. |

To ensure a fair comparison, the experimental configurations will be kept as consistent as possible across training strategies.

### 5.1 Data Loading and Preprocessing

The experimental pipeline uses MRI volumes and corresponding segmentation masks stored in the SSL3D/nnU-Net preprocessed format.

A custom PyTorch Dataset, `SSL3DDataset`, provides a common data-loading interface for FeTA and BOBs.

The implementation supports:

- Loading paired MRI and segmentation volumes from `.b2nd` files.
- Configurable 3D patch sizes or complete-volume loading.
- Random patch sampling for training.
- Optional foreground-based sampling.
- Deterministic center-patch extraction for validation and testing.
- Spatially aligned image and segmentation patches.
- Validation of image dimensions, numerical values, and segmentation labels.
- Conversion to PyTorch tensors for model training.

For the currently supported preprocessed labels:

- `0` represents the background.
- `1–7` represent foreground segmentation classes.
- `-1` represents voxels excluded from the segmentation loss.

The anatomical meaning of foreground labels is dataset-dependent and must be verified against the corresponding preprocessing configuration.

When patches are larger than the available image dimensions, zero padding is applied to MRI intensities and ignore-label padding (`-1`) is applied to segmentation masks.

The dataset loader does not modify or overwrite the original preprocessed volumes.

**Important:** The training loss must explicitly ignore voxels labelled `-1`.

Additional preprocessing details, including spatial resolution, intensity normalization, and label mappings, will be documented as the experimental configuration is finalized.

### 5.2 Labelled Data Efficiency

To assess whether pretraining reduces annotation requirements, we will evaluate segmentation performance using different fractions of the available labelled training data.

The same training subsets will be used across model initialization strategies whenever possible, allowing a controlled comparison of data efficiency.

Training subset sizes, sampling procedures, and random seeds will be documented to ensure reproducibility.

### 5.3 Evaluation Metrics

We plan to compare the different approaches using:

- **Dice Similarity Coefficient (DSC):** Measures the overlap between predicted and ground-truth segmentations.
- **95th-percentile Hausdorff Distance (HD95):** Measures segmentation boundary discrepancies while reducing sensitivity to extreme outliers.
- **Volumetric Similarity (VS):** Measures agreement between predicted and reference segmentation volumes.
- **Additional analyses:** Segmentation errors, generalization, and computational requirements when feasible.

Evaluation will account for dataset-specific anatomical labels and annotation protocols.

### 5.4 Experimental Fairness

To ensure meaningful comparisons, experiments will aim to use:

- Consistent training, validation, and test splits.
- Identical training data subsets across compared strategies.
- Comparable preprocessing and data augmentation procedures.
- Consistent evaluation metrics.
- Documented model architectures, hyperparameters, and random seeds.

For BOBs, participant-level splitting will be used to prevent sessions from the same infant from appearing across different data splits.

The final experimental setup will be documented as implementation progresses.

## 6. Repository Structure

```text
brain-mri-foundation-models/
├── data/                     # Dataset documentation and local data
│   ├── README.md
│   ├── splits/               # Predefined dataset splits
│   ├── nnUNet_raw/           # Raw nnU-Net datasets (Git-ignored)
│   ├── nnUNet_preprocessed/  # Preprocessed MRI volumes (Git-ignored)
│   └── nnUNet_results/       # nnU-Net outputs (Git-ignored)
│
├── notebooks/                # Exploratory data analysis
│
├── results/                  # Experimental results and visualizations
│
├── scripts/                  # Experiment execution scripts
│
├── src/                      # Reusable Python modules
│   ├── datasets/             # Dataset loading and patch extraction
│   ├── metrics/              # Segmentation evaluation metrics
│   ├── models/               # Model architectures and initialization
│   └── training/             # Training and validation utilities
│
├── .gitignore
├── .pre-commit-config.yaml
└── README.md
```

The repository separates reusable model and training code from exploratory notebooks and experiment execution scripts.

Large MRI datasets, preprocessed volumes, model checkpoints, and generated experiment outputs are excluded from version control.

Only source code, configuration files, dataset documentation, and reproducibility-related metadata are intended to be tracked.

Some directories and modules will be populated progressively as implementation continues.

## 7. Installation and Setup

The project is developed in Python and uses PyTorch for deep learning and 3D medical image segmentation.

### 7.1 Main Dependencies

The primary libraries include:

- **PyTorch:** Deep learning, automatic differentiation, and model training.
- **MONAI:** Medical image analysis and 3D segmentation utilities.
- **NumPy:** Numerical computations and array manipulation.
- **Blosc2:** Reading preprocessed `.b2nd` MRI volumes.
- **NiBabel:** Loading and manipulating NIfTI medical images.
- **Matplotlib:** Data visualization.

Additional dependencies may be introduced as the training and evaluation pipelines develop.

### 7.2 Repository Setup

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd brain-mri-foundation-models
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment using:

```bash
.venv\Scripts\activate
```

A reproducible dependency specification and complete installation instructions will be provided once the software environment has been finalized.

### 7.3 Dataset Setup

MRI datasets must be obtained separately and organized according to the expected SSL3D/nnU-Net directory structure.

The custom dataset loader expects preprocessed FeTA and BOBs directories containing paired MRI and segmentation `.b2nd` files.

For example:

```text
data/
└── nnUNet_preprocessed/
    ├── Dataset101_FeTA/
    │   ├── Spacing__1.00_1.00_1.00___Norm__Z/
    │   ├── SSL3D_val/
    │   └── SSL3D_test/
    │
    └── Dataset102_BOBs/
        ├── Spacing__1.00_1.00_1.00___Norm__Z/
        ├── SSL3D_val/
        └── SSL3D_test/
```

Each split directory contains paired files of the form:

```text
case_id.b2nd
case_id_seg.b2nd
```

Detailed dataset download and preparation instructions are maintained in [`data/README.md`](data/README.md).

### 7.4 Hardware

Training 3D segmentation networks is computationally demanding and generally benefits from GPU acceleration.

GPU resources, hardware specifications, and environment configuration will be documented once the final training infrastructure is established.

## 8. Experiments and Results

*Work in progress.*

This section will document the completed experiments and their corresponding results.

Planned analyses include:

- Baseline segmentation performance using randomly initialized models.
- Performance with frozen pretrained encoders.
- Performance after fine-tuning pretrained models.
- Segmentation quality across different labelled training-set sizes.
- Comparisons of DSC, HD95, and VS.
- Visual comparisons between predicted and ground-truth segmentations.
- Analysis of the benefits and limitations of pretraining.

Results will be reported separately for FeTA and BOBs where appropriate.

Cross-dataset comparisons will explicitly consider differences in anatomy, image characteristics, and annotation protocols.

Only completed experiments and measured results will be reported.

## 9. Reproducibility

Reproducibility is a central objective of this project.

We aim to provide a documented experimental pipeline covering:

- Dataset preparation and preprocessing.
- Dataset-specific label definitions and mappings.
- Subject-level training, validation, and test splits.
- Model architectures and pretrained checkpoint selection.
- Training configurations and hyperparameters.
- Random seeds and data sampling procedures.
- Evaluation scripts and metrics.
- Experimental results and comparisons.

### 9.1 Data Splitting

Dataset splits will be defined to minimize data leakage and enable consistent comparisons.

For BOBs, all MRI sessions from the same participant must remain within the same split.

The same predefined splits will be reused across training strategies whenever possible.

### 9.2 Training Reproducibility

Experimental configurations will document:

- Model initialization strategy.
- Optimizer and learning rate.
- Batch size and patch size.
- Number of training epochs.
- Loss functions.
- Data augmentation procedures.
- Random seeds.
- Hardware and software environment.

### 9.3 Evaluation Reproducibility

Evaluation procedures will be kept consistent across compared methods.

Segmentation metrics, handling of ignored labels, aggregation procedures, and dataset-specific evaluation settings will be documented.

Detailed instructions for reproducing the experiments will be provided as the project develops.

## 10. Team and Supervision

**Student team — EPFL CS-433**

- Yasmine Bouattour
- Gwenaëlle Beulque
- Florent Guesde

**Hosting laboratory:** Medical Image Analysis Lab (MIAL), UNIL / CHUV.

**Academic supervision:** Prof. Meritxell Bach Cuadra, with guidance from the hosting laboratory team.

## 11. References

1. Gordaliza, P. M., Banus, J., Gérin, B., Wynen, M., Molchanova, N., Richiardi, J., & Bach Cuadra, M. (2026). *From 100,000+ images to winning the first brain MRI foundation model challenges: Sharing lessons and models.* arXiv:2601.13166. https://doi.org/10.48550/arXiv.2601.13166

2. Payette, K., de Dumast, P., Kebiri, H., et al. (2021). *An automatic multi-tissue human fetal brain segmentation benchmark using the Fetal Tissue Annotation Dataset.* Scientific Data, 8, 167. https://doi.org/10.1038/s41597-021-00946-3

3. Feczko, E., Stoyell, S. M., Moore, L. A., et al. (2025). *Baby Open Brains: An open-source dataset of infant brain segmentations.* Scientific Data, 12, 1423. https://doi.org/10.1038/s41597-025-05404-y

---

**Project status:** In progress — Fall 2026.

**Course:** EPFL CS-433 Machine Learning.

**Submission deadline:** December 17, 2026.

# Evaluating the Benefits and Limits of Brain Foundation Models in Developmental MRI

**EPFL — CS-433 Machine Learning | Fall 2026**

Machine Learning for Science (ML4Science) project conducted in collaboration with the **Medical Image Analysis Lab (MIAL)**, affiliated with the University of Lausanne (UNIL) and Lausanne University Hospital (CHUV).

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

1. Cerebrospinal fluid (CSF)
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

**Data availability:** MRI datasets are not stored in this repository. Please refer to [`data/README.md`](data/README.md) for dataset information and setup instructions.

## 4. Foundation Model

This project investigates the transferability of pretrained brain MRI representations, with particular interest in the foundation models presented by Gordaliza et al. (2026).

Their work introduces a brain MRI foundation model approach based on a convolutional U-Net architecture, incorporating anatomical priors and neuroimaging domain knowledge.

The models achieved first-place results in tracks of the SSL3D and FOMO25 brain MRI foundation model challenges at MICCAI 2025.

**Reference:** Gordaliza et al. (2026).

**Paper:** https://arxiv.org/abs/2601.13166

**Official GitHub repository:** https://github.com/jbanusco/BrainFM4Challenges

The specific pretrained checkpoint and architecture used in our experiments will be documented once the experimental configuration has been finalized.

## 5. Methodology

We plan to investigate three main training strategies:

| Strategy | Description |
|---|---|
| Training from scratch | Train a segmentation network using randomly initialized weights. |
| Frozen pretrained encoder | Use a pretrained encoder as a fixed feature extractor and train a segmentation head or decoder. |
| Fine-tuning | Initialize the network with pretrained weights and adapt its parameters to the target segmentation task. |

### Labelled Data Efficiency

To assess whether pretraining reduces annotation requirements, we will evaluate segmentation performance using different fractions of the available labelled training data.

### Evaluation

We plan to compare the different approaches using:

- Dice Similarity Coefficient (DSC) for segmentation overlap.
- 95th-percentile Hausdorff Distance (HD95) for segmentation boundary accuracy.
- Volumetric Similarity (VS) for differences in segmented tissue volumes.
- Additional analyses of segmentation errors, generalization, and computational requirements when feasible.

The experiments will use consistent training, validation, and testing procedures to ensure fair comparisons.

The experimental setup, hyperparameters, and preprocessing pipeline will be documented as implementation progresses.

## 6. Repository Structure

```text
brain-mri-foundation-models/
├── data/                  # Dataset setup instructions (data ignored by Git)
│   └── README.md
├── notebooks/             # Exploratory data analysis and experiments
│   └── 01_explore_feta.ipynb
├── results/               # Experimental results and visualizations
├── scripts/               # Experiment execution scripts
├── src/                   # Reusable Python code
├── .gitignore
└── README.md
```

## 7. Installation and Setup

Installation instructions, dependencies, and environment configuration will be added as implementation progresses.

The project will primarily use Python, with libraries such as:

- **PyTorch:** Deep learning and model training.
- **MONAI:** Medical image analysis and 3D segmentation.
- **NiBabel:** Loading and manipulating NIfTI medical images.
- **NumPy:** Numerical computations.
- **Matplotlib:** Data visualization.

Hardware requirements and instructions for reproducing the experiments will be documented once the training environment is finalized.

## 8. Experiments and Results

*Work in progress.*

This section will document:

- Baseline segmentation performance.
- Frozen-encoder and fine-tuning experiments.
- Performance across different labelled training-set sizes.
- Segmentation metrics and visual comparisons.
- Analysis of the benefits and limitations of pretraining.

Only completed experiments and measured results will be reported.

## 9. Reproducibility

Our goal is to provide a reproducible research pipeline, including:

- Dataset preparation and preprocessing.
- Subject-level training, validation, and testing splits.
- Model architectures and configurations.
- Training hyperparameters and random seeds.
- Evaluation scripts and metrics.
- Experimental results and comparisons.

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
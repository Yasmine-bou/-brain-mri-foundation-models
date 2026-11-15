# Data

This directory contains the datasets used in our EPFL CS-433 ML4Science project: **Evaluating the Benefits and Limits of Brain Foundation Models in Developmental MRI**.

## Datasets

### 1. FeTA — Fetal Tissue Annotation Dataset

- **Description:** T2-weighted fetal brain MRI volumes and corresponding tissue segmentation masks.
- **Reference:** Payette et al. (2021).
- **Website:** https://fetachallenge.github.io/

### 2. Baby Open Brains (BOBs)

- **Description:** Infant brain MRI volumes and associated brain segmentation annotations.
- **Reference:** Feczko et al. (2025).
- **DOI:** https://doi.org/10.1038/s41597-025-05404-y

## Setup

1. Obtain the datasets from their official sources, following the applicable access and licensing conditions.
2. Place the downloaded datasets in the `data/` directory.
3. Organize the files as follows:

```text
data/
├── README.md
├── FeTA/
│   └── feta_2.4/
└── bobs/
```

## Important

The MRI volumes and segmentation masks are **not tracked by Git**.

Only this README is included in the repository.

Dataset files must be downloaded separately and used in accordance with their respective licenses and data usage conditions.

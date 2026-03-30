# Pereira 2018 Dataset

**Toward a universal decoder of linguistic meaning from brain activation**

Paper Link: [https://www.nature.com/articles/s41467-018-03068-4](https://www.nature.com/articles/s41467-018-03068-4)

- **Authors**: Francisco Pereira, Bin Lou, Brianna Pritchett, Samuel Ritter, Samuel J. Gershman, Nancy Kanwisher, Matthew Botvinick & Evelina Fedorenko
- **Published**: 06 March 2018, *Nature Communications*

## Overview
The Pereira 2018 dataset is one of the most comprehensive sentence-level fMRI datasets. It includes brain activations (beta images) associated with reading sentences across three separate experiments:
- **Experiment 1**: 180 sentences (presented in variants: sentences, pictures+words, word clouds).
- **Experiment 2**: 384 sentences.
- **Experiment 3**: 243 sentences.

The sentences belong to 96 (Expt 2) and 72 (Expt 3) thematic passages.

## Data Characteristics
- **Format**: fMRI beta coefficients per stimulus (sentence/concept).
- **Parcellation**: Brain data is typically provided either in a whole-head voxel mask or pooled into anatomical atlases (AAL or Gordon).
- **Alignment**: This is a **Sentence-Level (Stimulus-Level)** dataset. Each sample consists of one whole-brain activation vector per sentence. It does NOT contain the temporal word-by-word progression of the BOLD signal.

## Metadata Information
- `examples`: Beta coefficient images for each stimuli (N_stimuli x N_voxels/parcels).
- `keySentences`/`labelsSentences`: Identifiers and text for each stimulus.
- `meta`: Struct containing voxel coordinates and ROI mapping (Gordon and AAL atlases).

## Usage in ConvMinds
In this library, the dataset is loaded via `PereiraBenchmark`. It handles:
1.  **ROI Pooling**: Averaging voxel activations within Gordon atlas parcels (default: 333 parcels).
2.  **Coordinate Mapping**: Propagating the 3D centroids of the ROI parcels for spatial encoding.
3.  **Stimulus Uniqueness**: Handling cross-subject identification to keep subject-specific responses distinct.

# DAFI-Net

## DAFI-Net: A Dual-Domain Adaptive Frequency Integration Transformer Network for Accelerated MRI Reconstruction

This repository contains the implementation of **DAFI-Net**, a dual-domain deep learning framework for accelerated MRI reconstruction.

DAFI-Net integrates spatial-domain and frequency-domain feature representations through a **Multi-scale VisionMambaLight Frequency Feature Extraction Branch (MVFFB)**, a **Swin Transformer-based Spatial Feature Extraction Branch (SFEB)**, an **Adaptive Weight Generation Module (AWGM)**, and **Adaptive Frequency Integration (AFI)**.

---

## Overview

Accelerated MRI reconstruction aims to recover high-quality MR images from undersampled k-space measurements.

DAFI-Net processes undersampled k-space data through two complementary domains:

1. **Frequency domain** – frequency-domain features are extracted using the Multi-scale VisionMambaLight Frequency Feature Extraction Branch (MVFFB).

2. **Spatial domain** – undersampled k-space is converted into an image using an inverse Fourier transform and processed using the Swin Transformer-based Spatial Feature Extraction Branch (SFEB).

The features from both domains are adaptively integrated using the **Adaptive Weight Generation Module (AWGM)** and **Adaptive Frequency Integration (AFI)**.

### Main Components

- **Multi-scale VisionMambaLight Frequency Feature Extraction Branch (MVFFB)**
- **Swin Transformer-based Spatial Feature Extraction Branch (SFEB)**
- **Adaptive Weight Generation Module (AWGM)**
- **Adaptive Frequency Integration (AFI)**
- **Reconstruction Head**

> **Note:** Set `DATA_DIR` in `train.py` to the location of your preprocessed fastMRI dataset.

## Dataset

DAFI-Net was trained and evaluated using the **fastMRI** knee dataset.

The fastMRI dataset is a collaborative research project by Meta AI (formerly Facebook AI Research) and NYU Langone Health. It provides publicly available MRI data for research on accelerated MRI reconstruction.

The dataset can be accessed from the official fastMRI website:

- **[fastMRI Dataset](https://fastmri.med.nyu.edu/)**

> **Note:** Users must follow the fastMRI dataset access requirements and terms of use when downloading and using the dataset.

## Acknowledgements

The spatial-domain branch of DAFI-Net uses code adapted from
[SwinIR](https://github.com/JingyunLiang/SwinIR), released under the
Apache License 2.0.

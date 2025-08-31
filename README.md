# Automated Segmentation of Peripheral Arteries in 3D CT Data Based on a Single-Depth Vascular Network

## Introduction
This repository contains source code for the automated segmentation of peripheral arteries from 3D CT scans using a Single-Depth Vascular Network (SDVN). The pipeline allows preprocessing, training, and evaluation of segmentation models for medical imaging research.

## Abstract
Peripheral artery disease (PAD), if left untreated, can lead to cardiovascular mortality. The diagnosis of PAD typically requires complex and costly procedures, such as computed tomography (CT), followed by extensive examinations. In addition, segmenting peripheral arteries through deep networks still remains challenging due to their small and complex structure, motion artifacts, class imbalance, and variations in field of view in different datasets. Furthermore, handling large-scale whole-body scans efficiently complicates the segmentation process. To address these limitations, this study employs three-dimensional CT scans to segment peripheral arteries. Firstly, a YOLOv8 network is developed to extract the region of interest (ROI) from whole-body scans. This approach optimizes processing by focusing on ROI while reducing memory usage and processing time. Afterwards, ROI is fed into a novel Single-Depth Vascular Network for peripheral artery segmentation. The Single-Depth Vascular Network encapsulates an encoder–decoder architecture with a skip connection to improve the accuracy and precision of segmentation. Through this approach, the network achieves an average Dice similarity coefficient of 0.91, a Hausdorff distance of 12.60 mm, and a 95th percentile Hausdorff distance of 1.12 mm against unseen data. This approach provides a tool for medical professionals to diagnose and treat PAD with greater precision.

## SDVN
![SDVN Architecture](network.png)


## Pipeline
The pipeline performs:
- Preprocessing of CT scans (normalization, resampling, ROI extraction)
- Training of the SDVN model
- Model evaluation and inference on test CT scans

## Requirements
- PyTorch
- SimpleITK
- NumPy, SciPy


## Citation
If you use this work in your research, please cite:

Manahil Zulfiqar, Maciej Stanuch, Sylvia Vagena, Fragiska Sigala, Andrzej Skalski,
Automated segmentation of peripheral arteries in 3D CT data based on a Single-Depth vascular network,
Biomedical Signal Processing and Control,
Volume 112, Part A,
2026,
108410,
ISSN 1746-8094,
https://doi.org/10.1016/j.bspc.2025.108410.
(https://www.sciencedirect.com/science/article/pii/S1746809425009218)

## Acknowledgements
This study is funded with European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 956470.

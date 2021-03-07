# Visualization and Medical Image Analysis*

This repository aims to develop a CNN-based 2D semantic segmentation module for brain tumor detection on BraTS 2019 dataset.

Architecture: U-Net

Installation:

pip install -e .

Train:

python visualizer/train.py -c config/train.gin

Evaluation:

python visualizer/evaluate/evaluate.py -c config/evaluate.gin

References:

[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694.

[2] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117.

[3] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018).

[4] Rachel Lea Ballantyne Draelos, Best Use of Train/Val/Test Splits with Tips for Medical Data, Glass Box, Available at: https://glassboxmedicine.com/2019/09/15/best-use-of-train-val-test-splits-with-tips-for-medical-data/, Accessed on: 19. 12. 2020.

[5] Captum, 'Semantic Segmentation with Captum', Captum Tutorials, Available at: https://captum.ai/tutorials/Segmentation_Interpret, Accessed on: 19. 12. 2020.

\* - Under development phase
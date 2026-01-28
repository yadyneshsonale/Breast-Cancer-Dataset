# Breast Cancer Thermal ROI Dataset

A curated dataset of breast cancer thermal images with manually identified and segmented Regions of Interest (ROIs), developed in collaboration with medical professionals.

## 🚀 How to Use

1. ⭐ **Star** this repository if you find it useful!
2. 🍴 **Fork** the repository to your own GitHub account
3. 📥 **Clone** your forked repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Breast-Cancer-Dataset.git
   ```
4. 📂 Access the data in the `data/` folder organized by patient

> If you use this dataset in your research, please consider citing the publications listed below.

## Overview

This dataset contains thermal breast imaging data from **86 patients** (labeled `p1` through `p109`, with some patient numbers not included). Each patient folder contains paired ROI images used for breast cancer analysis and research.

The ROIs were segmented from the publicly available [DMR-IR dataset](https://data.mendeley.com/datasets/mhrt4svjxc/3) hosted on Mendeley Data.

## Clinical Methodology

During initial breast cancer screening using thermal imaging, clinicians identify regions with high metabolic activity — these appear as **hotspots** (whiter areas compared to surrounding tissue). Once a hotspot is identified on one breast, the corresponding **symmetric region** on the contralateral (other) breast is examined.

- **If metabolic activity is similar** in both regions → **Possibly Benign**
- **If metabolic activity differs** between regions → **Possibly Malignant**

This is why the images are provided in **pairs** — each pair consists of the ROI from one breast and its symmetric counterpart from the other breast.

## Dataset Structure

```
Breast-Cancer-Dataset/
├── medical_image_classifier.py    # Neural network classifier for PB/PM classification
├── data/
│   ├── p1/
│   │   ├── b1.jpg    # ROI from one breast (Possibly Benign)
│   │   └── b2.jpg    # Symmetric ROI from contralateral breast (Possibly Benign)
│   ├── p19/
│   │   ├── m1.jpg    # ROI from one breast (Possibly Malignant)
│   │   └── m2.jpg    # Symmetric ROI from contralateral breast (Possibly Malignant)
│   ├── ...
│   └── p109/
│       └── ...
├── LICENSE
└── README.md
```

### Naming Convention

- **Patient Folders**: `pX` where `X` is the patient number (e.g., `p1` = Patient 1, `p109` = Patient 109)
- **Image Files**:
  - `b1.jpg` & `b2.jpg` — **Possibly Benign** ROI pair (symmetric regions from both breasts)
  - `m1.jpg` & `m2.jpg` — **Possibly Malignant** ROI pair (symmetric regions from both breasts)

## Data Collection

The ROIs in this dataset were manually identified and segmented in collaboration with a medical doctor to ensure clinical accuracy and relevance for research purposes.

## Source Dataset

This dataset was created by segmenting ROIs from the DMR-IR database:

> **DMR-IR Database**: [https://data.mendeley.com/datasets/mhrt4svjxc/3](https://data.mendeley.com/datasets/mhrt4svjxc/3)

## Related Publications

This dataset has not been used in any publications yet. However, it was developed as part of the research presented in the following papers:

1. **Yadynesh D Sonale**, et al. *"Deep learning-based classification of breast abnormalities using thermal imaging and ResNet-50."* In Proceedings of the **ASME International Mechanical Engineering Congress & Exposition (IMECE), 2025**.  
   📄 [Paper Link](https://asme.pinetec.com/imece-india2025/data/pdfs/trk-9/IMECE-INDIA2025-161705.pdf)

2. **Yadynesh D Sonale**, et al. *"A comparative study of pre-trained deep learning models with and without pre-processing for multi-class classification of thermal breast images in early cancer detection."* In Proceedings of the **ASME International Mechanical Engineering Congress & Exposition (IMECE), 2025**.  
   📄 [Paper Link](https://asme.pinetec.com/imece-india2025/data/pdfs/trk-6/IMECE-INDIA2025-161724.pdf)

## Usage

This dataset can be used for:
- Breast cancer detection and classification research
- Training machine learning models for thermal medical image analysis
- ROI segmentation and symmetry analysis tasks
- Educational purposes in medical imaging

## Code

### `medical_image_classifier.py`

A simplified neural network classifier for medical images that categorizes images into two classes ("PB" - Possibly Benign and "PM" - Possibly Malignant).

**Features:**
- Processes grayscale image histograms as 256-dimensional feature vectors
- Uses a feedforward neural network with dropout layers for classification
- Includes data loading, preprocessing, and model training
- Provides evaluation metrics and visualization of accuracy/loss curves
- Generates ROC analysis for model performance assessment

**Quick Start:**
```bash
python medical_image_classifier.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Yadynesh D Sonale

## Disclaimer

This dataset is intended for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

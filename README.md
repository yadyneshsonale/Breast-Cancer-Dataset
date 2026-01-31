# Breast Cancer Thermal ROI Dataset

A curated dataset of breast cancer thermal images with manually identified and segmented Regions of Interest (ROIs), developed in collaboration with medical professionals.

## 🚀 How to Use

1. ⭐ **Star** this repository if you find it useful!
2. 🍴 **Fork** the repository to your own GitHub account
3. 📥 **Clone** your forked repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Breast-Cancer-Dataset.git
   cd Breast-Cancer-Dataset
   ```
4. 📂 Access the symmetry data in the `symmetry data/` folder organized by patient

> If you use this dataset in your research, please consider citing the publications listed below.

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setting Up a Virtual Environment

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   - **On macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
   - **On Windows:**
     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the symmetry analysis code:**
   ```bash
   cd code
   python symmetry_code.py
   ```

5. **Deactivate the virtual environment (when done):**
   ```bash
   deactivate
   ```

### Troubleshooting
- If you encounter issues installing TensorFlow on Apple Silicon Macs, you may need to use conda or install a compatible version
- Ensure your virtual environment is activated before running any Python scripts
- Use `pip list` to verify all packages are installed correctly

## Overview

This dataset contains thermal breast imaging data from **86 patients** (labeled `p1` through `p109`, with some patient numbers not included). Each patient folder contains paired ROI images used for breast cancer analysis and research.

The ROIs were segmented from the publicly available [DMR-IR dataset](https://data.mendeley.com/datasets/mhrt4svjxc/3) hosted on Mendeley Data.

### How the Dataset Was Created

1. **Hotspot Identification**: A region of interest (ROI) showing high metabolic activity (hotspot) was identified on one breast in the thermal image using the labels provided in the DMR-IR dataset
2. **Bounding Box Extraction**: A rectangular bounding box was drawn around the identified ROI to extract the region
3. **Symmetric Region Extraction**: A bounding box of the **same dimensions** was placed at the **symmetric location** on the contralateral (opposite) breast
4. **Pair Creation**: Both extracted regions were saved as an image pair for comparison

This symmetric extraction approach allows for direct comparison between the suspicious region and its anatomically corresponding region on the healthy breast.

## Clinical Methodology

During initial breast cancer screening using thermal imaging, clinicians identify regions with high metabolic activity — these appear as **hotspots** (whiter areas compared to surrounding tissue). Once a hotspot is identified on one breast, the corresponding **symmetric region** on the contralateral (other) breast is examined.

- **If metabolic activity is similar** in both regions → **Possibly Benign**
- **If metabolic activity differs** between regions → **Possibly Malignant**

This is why the images are provided in **pairs** — each pair consists of the ROI from one breast and its symmetric counterpart from the other breast.

## Dataset Structure

```
Breast-Cancer-Dataset/
├── code/
│   └── symmetry_code.py           # Neural network classifier for symmetry analysis
├── paper_data/
│   ├── left/                      # Segmented left breast images
│   │   ├── IIR0001.png - IIR0118.png
│   │   └── left.xlsx              # Metadata/labels
│   └── right/                     # Segmented right breast images
│       ├── IIR0001.png - IIR0118.png
│       └── right.xlsx             # Metadata/labels
├── symmetry data/
│   ├── p1/
│   │   ├── b1.jpg                 # ROI from one breast (Possibly Benign)
│   │   └── b2.jpg                 # Symmetric ROI from contralateral breast
│   ├── p19/
│   │   ├── m1.jpg                 # ROI from one breast (Possibly Malignant)
│   │   └── m2.jpg                 # Symmetric ROI from contralateral breast
│   ├── ...
│   └── p109/
│       └── ...
├── requirements.txt
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

## Paper Data (`paper_data/`)

The `paper_data/` folder contains segmented breast images created to replicate the dataset methodology described in our research papers. **Note:** This is not the original data used in the papers, but rather data prepared by us following the same segmentation pipeline.

### Segmentation Process

This segmentation process involved delineating the boundaries of each breast to isolate them from the surrounding tissue and background. This step is crucial for ensuring that the subsequent analysis focuses solely on the breast tissue, thereby improving the accuracy and reliability of the results.

By focusing exclusively on segmenting the breast tissue, the study aimed to eliminate potential sources of false positives, such as axillary lymph nodes, which can often appear in thermal images and may be mistakenly identified as areas of concern.

### Folder Structure

```
paper_data/
├── left/                    # Segmented left breast images
│   ├── IIR0001.png          # Image files (IIR0001 - IIR0118)
│   ├── ...
│   └── left.xlsx            # Metadata/labels
└── right/                   # Segmented right breast images
    ├── IIR0001.png          # Image files (IIR0001 - IIR0118)
    ├── ...
    └── right.xlsx           # Metadata/labels
```

## Source Dataset

This dataset was created by segmenting ROIs from the DMR-IR database:

> **DMR-IR Database**: [https://data.mendeley.com/datasets/mhrt4svjxc/3](https://data.mendeley.com/datasets/mhrt4svjxc/3)

## Usage

This dataset can be used for:
- Breast cancer detection and classification research
- Training machine learning models for thermal medical image analysis
- ROI segmentation and symmetry analysis tasks
- Educational purposes in medical imaging

## Code

### `symmetry_code.py`

A neural network classifier for medical images that categorizes thermal breast ROIs into two classes: **PB** (Possibly Benign) and **PM** (Possibly Malignant) based on symmetry analysis.

**Features:**
- Extracts grayscale image histograms as 256-dimensional feature vectors
- Computes symmetry features by combining histogram data from paired ROIs
- Uses a feedforward neural network with dropout layers for classification
- Includes data loading from the `symmetry data/` folder
- Provides evaluation metrics and visualization of accuracy/loss curves
- Generates ROC curve analysis for model performance assessment

**Quick Start:**
```bash
cd code
python symmetry_code.py
```

### `resnet50_classifier.py`

A deep learning classifier using pre-trained **ResNet-50** architecture for breast tumor classification into three categories: **N** (Normal), **PB** (Possibly Benign), and **PM** (Possibly Malignant).

**Data Preprocessing:**
- All 238 segmented images are resized to a fixed dimension of **244 × 244 pixels** with **3 channels (RGB)** to ensure uniformity and compatibility with deep learning architectures
- Dataset is resampled to **550 samples** (440 training, 110 testing) to handle class imbalance
- Oversampling techniques are used to balance the dataset
- Data augmentation (rotation, flipping, cropping) is applied to reduce overfitting

**Model Architecture:**
- Pre-trained ResNet-50 on ImageNet as the base model
- Additional custom layers: GlobalAveragePooling2D → Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → BatchNorm → Dropout(0.3) → Dense(3, softmax)
- Training: 80% training / 20% testing split, 70 epochs

**Features:**
- Loads images from `paper_data/` folder with labels from Excel files
- Class imbalance handling through oversampling
- Data augmentation for improved generalization
- Training/validation accuracy and loss visualization
- ROC curve analysis for multi-class classification
- Confusion matrix generation

**Quick Start:**
```bash
cd code
python resnet50_classifier.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Yadynesh D Sonale

## Disclaimer

This dataset is intended for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Dataset Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{sonale2026breastcancer,
  author    = {Sonale, Yadynesh D; Kumar, Avinash},
  title     = {Breast Cancer Thermal ROI Dataset},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/YOUR_USERNAME/Breast-Cancer-Dataset}
}
```

## Related Publications

This dataset was developed as part of the research presented in the following papers:

1. **Yadynesh D Sonale**, et al. *"Deep learning-based classification of breast abnormalities using thermal imaging and ResNet-50."* In Proceedings of the **ASME International Mechanical Engineering Congress & Exposition (IMECE), 2025**.  
   📄 [Paper Link](https://asme.pinetec.com/imece-india2025/data/pdfs/trk-9/IMECE-INDIA2025-161705.pdf)

2. **Yadynesh D Sonale**, et al. *"A comparative study of pre-trained deep learning models with and without pre-processing for multi-class classification of thermal breast images in early cancer detection."* In Proceedings of the **ASME International Mechanical Engineering Congress & Exposition (IMECE), 2025**.  
   📄 [Paper Link](https://asme.pinetec.com/imece-india2025/data/pdfs/trk-6/IMECE-INDIA2025-161724.pdf)

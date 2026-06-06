# Explainable COVID-19 Detection from Chest X-Rays

> An Explainable AI system that detects COVID-19 and lung abnormalities from chest X-ray images using DenseNet121, visualizes decision-making through Grad-CAM, and estimates disease severity for clinical decision support.

---

## Overview

Medical imaging models often provide highly accurate predictions but fail to explain *why* a prediction was made.

This project addresses that limitation by combining **deep learning** and **explainable AI (XAI)** techniques to create an interpretable COVID-19 screening system.

The model analyzes chest X-rays and provides:

* Disease classification
* Prediction confidence
* Visual explanation of model attention
* Abnormality localization
* Severity estimation
* Clinical assistance summary

The objective is to bridge the gap between AI predictions and clinical trust.

---

## Key Features

### Multi-Class Disease Detection

Classifies chest X-ray images into:

* COVID-19
* Lung Opacity
* Normal

### Explainable AI

Generates Grad-CAM heatmaps to reveal the lung regions that influenced the prediction.

### Attention-Based Localization

Automatically identifies and highlights the most significant abnormal region.

### Severity Assessment

Estimates disease involvement and categorizes findings as:

* Mild
* Moderate
* Severe

### Interactive Web Application

Built using Streamlit for real-time inference and visualization.

---

## System Pipeline

```text
Chest X-Ray
      │
      ▼
Image Preprocessing
      │
      ▼
DenseNet121 Classification
      │
      ▼
Prediction + Confidence
      │
      ▼
Grad-CAM Generation
      │
      ▼
Attention Localization
      │
      ▼
Severity Analysis
      │
      ▼
Clinical Summary
```

---

## Model Architecture

### DenseNet121

The classification backbone is DenseNet121, chosen for:

* Dense feature reuse
* Efficient gradient propagation
* Reduced information loss
* Strong performance on medical imaging tasks

Dense connectivity allows each layer to receive information from all previous layers, improving representation learning for subtle pulmonary abnormalities.

---

## Explainability Layer

To improve transparency, Grad-CAM is integrated into the inference pipeline.

The system:

* Extracts activation maps from the final convolutional layers
* Computes class-specific gradients
* Produces visual attention maps
* Highlights diagnostically relevant lung regions

This allows users to understand the rationale behind each prediction.

---

## Dataset

**COVID-19 Radiography Dataset**

Classes:

| Class        | Description             |
| ------------ | ----------------------- |
| COVID        | COVID-19 Positive Cases |
| Lung Opacity | Pulmonary Opacity Cases |
| Normal       | Healthy Chest X-rays    |

Source:

Kaggle COVID-19 Radiography Database

---

## Results

| Metric              | Value       |
| ------------------- | ----------- |
| Training Accuracy   | 96.33%      |
| Validation Accuracy | 94.17%      |
| Classes             | 3           |
| Architecture        | DenseNet121 |

The model demonstrated strong classification performance while maintaining visual interpretability through Grad-CAM-based explanations.

---

## Tech Stack

| Category                | Technologies          |
| ----------------------- | --------------------- |
| Deep Learning           | PyTorch               |
| Architecture            | DenseNet121           |
| Explainability          | Grad-CAM              |
| Computer Vision         | OpenCV                |
| Frontend                | Streamlit             |
| Data Processing         | NumPy, Pillow         |
| Training Environment    | Google Colab (T4 GPU) |
| Development Environment | VS Code               |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sriyamvs8/MLmodelforCoviddetection.git
cd MLmodelforCoviddetection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Improvements

* Lung Segmentation using U-Net
* Vision Transformer (ViT) Integration
* Multi-Disease Thoracic Screening
* DICOM Support
* Real-Time Hospital Deployment
* Radiologist Feedback Loop

---

## Author

**Sriya M**

B.Tech



---

### If this project interests you, feel free to star the repository and connect for collaboration.

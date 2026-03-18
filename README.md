## Non-Invasic Anemia Detection via Conjunctiva Image Analysis

A research-based deep learning pipeline for the non-invasive screening of Anemia using the **Eye-Defy-Anemia** dataset (India & Italy cohorts). This project implements a clinical-to-image labeling engine and evaluates multiple CNN architectures to achieve high-sensitivity diagnostic results.

---

## Project Overview
Traditional anemia diagnosis requires invasive blood sampling. This project explores a **computer vision-based approach** by analyzing the pallor of the conjunctiva. 

### Key Contributions:
* **Automated Clinical Labeling:** Programmatically maps raw Hemoglobin (Hgb) levels to image folders based on WHO-standard thresholds ($14\text{ g/dL}$ for men, $12\text{ g/dL}$ for women).
* **Multi-Model Benchmarking:** Comparative analysis of **VGG16**, **InceptionV3**, and **DenseNet121**.
* **Weighted Ensemble Strategy:** Combines the feature-extraction strengths of DenseNet and Inception to minimize False Negatives.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `dataPrep.py` | Main script for cleaning and mapping Excel data to image paths. |
| `images.py` | Logic for applying age/gender-specific Hgb thresholds. |
| `DenseNet.py` | Training and evaluation for the DenseNet121 architecture. |
| `Inception.py` | Training and evaluation for the InceptionV3 architecture. |
| `ensembleAll.py` | Final weighted ensemble of all trained models. |
| `imageCheck.py` | Utility to verify data integrity and path existence. |

---

## Getting Started

### 1. Prerequisites
* Python 3.8+
* TensorFlow / Keras
* Pandas / NumPy
* Matplotlib (for visualization)

### 2. Data Preparation
Place your `India.xlsx` and `Italy.xlsx` files in the root directory. Ensure your images are structured as:
`dataset/[Country]/[PatientID]/*.jpg`

Run the labeling script:
```bash
python dataPrep.py
```

### 3. Training & Evaluation
To train a specific model (e.g., DenseNet):
```bash
python DenseNet.py
```

To run the final ensemble prediction:
```bash
python ensembleIncepDen.py
```

---

## Results & Performance
The project utilizes **Confusion Matrices** to prioritize **Recall (Sensitivity)**, as missing an anemic patient (False Negative) is the highest risk in a clinical setting. 

Current benchmarking shows that the **DenseNet + Inception Ensemble** provides the most robust generalization across both the India and Italy datasets.

---

## Future Work
* Integration of **Image Segmentation** to isolate the conjunctiva region automatically.
* Deployment via a mobile-friendly API for real-time field screening.
* Expansion to include "Older Adult" specific Hgb threshold adjustments.

---
### **Focus:** Medical AI & Computer Vision
---

## License

MIT License - Free to use with attribution.

---

## Contact

- **Email**: manrek.shivani8@gmail.com

---

<div align="center">


[⬆ Back to Top](#vigileyes-)

</div>

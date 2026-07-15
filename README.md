# 🧠 Graph-Enhanced 3D U-Net for Brain Tumor Segmentation on BraTS 2020

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## 📌 Overview

This repository presents a **Graph-Enhanced 3D U-Net** for automated brain tumor segmentation on the **BraTS 2020** dataset.

Unlike conventional CNN-based segmentation methods, this work introduces a **Graph Refinement Module** that enhances the bottleneck representations of a 3D U-Net using graph-inspired feature aggregation, enabling better structural reasoning across tumor regions.

The proposed architecture segments three clinically significant tumor subregions:

- **Whole Tumor (WT)**
- **Tumor Core (TC)**
- **Enhancing Tumor (ET)**

This project was developed as part of a **Research Internship on Graph Neural Networks for Medical Image Analysis**.

---

# 🏆 Highlights

- ✅ Graph-enhanced 3D U-Net Architecture
- ✅ End-to-End Brain Tumor Segmentation
- ✅ BraTS 2020 Dataset
- ✅ MONAI-based Medical Imaging Pipeline
- ✅ Dice + CrossEntropy Combined Loss
- ✅ Automatic Evaluation Metrics
- ✅ WT / TC / ET Dice Score Computation
- ✅ Publication-quality Visualizations
- ✅ GUI-ready Deployment Architecture

---

# 📖 Motivation

Brain tumor segmentation is an essential step in computer-aided diagnosis.

Traditional CNN-based segmentation methods capture local spatial information effectively but often struggle to model long-range structural relationships among tumor regions.

Graph Neural Networks provide an elegant solution by modeling feature interactions globally.

This work combines:

- CNN-based volumetric feature extraction
- Graph-inspired feature refinement
- Medical image segmentation

to improve segmentation accuracy while maintaining computational efficiency.

---

# 🧬 Dataset

## BraTS 2020

The project uses the **Brain Tumor Segmentation Challenge (BraTS 2020)** dataset.

Each patient contains four MRI modalities:

- T1
- T1ce
- T2
- FLAIR

Ground truth labels:

| Label | Description |
|--------|-------------|
| 0 | Background |
| 1 | Necrotic / Non-enhancing Tumor |
| 2 | Peritumoral Edema |
| 3 | Enhancing Tumor |

Clinical evaluation regions:

- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

---

# 🏗 Proposed Architecture

```
                MRI Volumes
                     │
                     ▼
             MONAI Preprocessing
                     │
                     ▼
               3D U-Net Encoder
                     │
                     ▼
         Graph Feature Refinement
                     │
                     ▼
              3D U-Net Decoder
                     │
                     ▼
           Multi-class Segmentation
                     │
                     ▼
          WT • TC • ET Predictions
```

---

# 📂 Project Structure

```
BraTS2020/

│
├── datasets/
│
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── graph_refiner.py
│   └── unet_graph.py
│
├── losses/
│   └── combined_loss.py
│
├── train/
│   └── train_unet_graph.py
│
├── evaluate/
│   └── evaluate_unet_graph.py
│
├── preprocessing/
│
├── utils/
│
├── notebooks/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/brats2020.git

cd brats2020
```

Create environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Dependencies

Major libraries

- Python
- PyTorch
- MONAI
- NumPy
- SciPy
- scikit-learn
- nibabel
- matplotlib
- tqdm

---

# 🚀 Training

```python
from train.train_unet_graph import train_unet_graph

model = train_unet_graph(
    train_loader,
    val_loader,
    epochs=200,
    lr=1e-4
)
```

---

# 📈 Evaluation

```python
from evaluate.evaluate_unet_graph import evaluate

evaluate(
    model,
    test_loader
)
```

Metrics computed

- Accuracy
- Precision
- Recall
- F1 Score
- Dice Score
- Confusion Matrix
- Classification Report

Clinical Metrics

- WT Dice
- TC Dice
- ET Dice

---

# 📊 Experimental Results

## Final Proposed Model

| Metric | Score |
|---------|--------|
| Accuracy | **99.46%** |
| Precision | **77.56%** |
| Recall | **76.57%** |
| F1 Score | **76.73%** |

### Dice Scores

| Region | Dice |
|---------|------|
| Whole Tumor | **0.8915** |
| Tumor Core | **0.7435** |
| Enhancing Tumor | **0.7334** |

---

# 📉 Comparison with Baseline Models

| Model | WT | TC | ET |
|--------|----|----|----|
| GCN | 0.6972 | 0.6195 | 0.6160 |
| CNN-GCN | 0.6268 | 0.5379 | 0.5973 |
| GAT | 0.6406 | 0.5231 | 0.4727 |
| GraphSAGE | 0.6638 | 0.5910 | 0.5557 |
| ISGNN | Poor | Poor | Poor |
| **Graph-Enhanced 3D U-Net (Proposed)** | **0.8915** | **0.7435** | **0.7334** |

---

# 📊 Visualizations

The repository contains scripts for generating

- Training Loss Curve
- Validation Accuracy
- Dice Comparison
- WT/TC/ET Bar Chart
- Radar Chart
- Confusion Matrix
- Model Comparison Charts

---

# 💡 Key Contributions

✔ Developed a Graph-Enhanced 3D U-Net architecture.

✔ Integrated graph-based feature refinement into volumetric segmentation.

✔ Achieved significant improvements over conventional graph baselines.

✔ Built a complete training and evaluation pipeline.

✔ Computed clinically meaningful BraTS metrics.

✔ Designed a deployment-ready architecture suitable for GUI and embedded systems.

---

# 🔬 Future Work

Future improvements include

- Graph Attention Networks
- Dynamic Graph Construction
- Transformer-based Encoders
- Explainable AI
- Edge Deployment using NVIDIA Jetson Nano
- Real-time Clinical GUI

---

# 👩‍💻 Author

**Deepti Yadav**

B.Tech Computer Science Engineering

Research Internship

Graph Neural Networks for Medical Image Segmentation

---

# 📚 References

1. Menze et al., The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS), IEEE TMI.

2. Bakas et al., Advancing The Cancer Genome Atlas Glioma MRI Collections with Expert Segmentation Labels.

3. Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation.

4. Çiçek et al., 3D U-Net: Learning Dense Volumetric Segmentation.

5. Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks.

6. Hamilton et al., GraphSAGE.

7. Veličković et al., Graph Attention Networks.

---

# ⭐ Acknowledgements

- BraTS Challenge Organizers
- MONAI Development Team
- PyTorch Community
- Research Internship Mentor
- Open Source Community

---

## 📜 License

This project is intended for **research and educational purposes**.

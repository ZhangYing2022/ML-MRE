# ML-MRE: Multi-Label Multimodal Relation Extraction

This repository provides the **M2RE dataset** and baseline implementations for the task of **Multi-Label Multimodal Relation Extraction**.

## 🔍 Overview

ML-MRE is a benchmark designed to reflect the complexity and ambiguity of real-world image-text pairs. Unlike prior MRE datasets that assume one relation per entity pair or focus on unimodal inputs, ML-MRE supports:
- Multi-label relations per entity pair
- Multimodal input reasoning (text + image)
- Fine-grained relation semantics across 24 relation types

## 📂 Repository Structure
```bash
.
├── data/ # Processed data files (JSON format)
├── models/ # Baseline models and configurations
├── processor/ # Preprocessing scripts
├── trainer/ # Training
├── run.py
└── README.md
```

## 📦 Dataset

The dataset can be obtained in [Google Drive](https://drive.google.com/drive/folders/1wgaydIUgMine0WWF4xxaWE-p_QqdTEBV?usp=sharing), it is constructed from news-style image-text pairs, with human-verified annotations. Each sample contains:
- A news article (caption/sentence)
- A corresponding image
- Entity pair
- Multiple relation labels per entity pair


## 🚀 Getting Started
1. Clone the repo
```bash
git clone https://github.com/ZhangYing2022/ML-MRE.git
```
2. Install the environment
```bash
cd ML-MRE
pip install -r requirements.txt
```
3. Train
```bash
python train_hvformer.py
```



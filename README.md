# ML-MRE: Multi-Label Multimodal Relation Extraction

This repository provides the **M2RE dataset** and baseline implementations for the task of **Multi-Label Multimodal Relation Extraction**.

## 🔍 Overview

ML-MRE is a benchmark designed to reflect the complexity and ambiguity of real-world image-text pairs. Unlike prior MRE datasets that assume one relation per entity pair or focus on unimodal inputs, ML-MRE supports:
- **Multi-label relations per entity pair**
- **Multimodal input reasoning (text + image)**
- Fine-grained relation semantics across 24 relation types

## 📂 Repository Structure
```bash
.
├── data/ # Processed data files (JSON format)
├── models/ # Baseline models and configurations
├── scripts/ # Training, evaluation, and preprocessing scripts
├── results/ # Sample output logs or prediction results
└── README.md
```

## 📦 Dataset

The dataset is constructed from news-style image-text pairs, with human-verified annotations. Each sample contains:
- A news article (caption/sentence)
- A corresponding image
- Entity pair
- Multiple relation labels per entity pair


## 🚀 Getting Started

```bash
git clone https://github.com/your-username/ml-mre.git
cd ml-mre
pip install -r requirements.txt
bash scripts/train_hvformer.sh
```

## Cite
```bash
@inproceedings{your2024mlmre,
  title={ML-MRE: A Multi-Label Multimodal Relation Extraction Benchmark},
  author={Your, Name and Others},
  booktitle={Proceedings of XYZ Conference},
  year={2024}
}
```

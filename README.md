# 🧠 MedMNIST v2 Deep Learning Project

This project is focused on replicating and extending the MedMNIST v2 paper, which provides a standardized benchmark for lightweight biomedical image classification. The goal is to build baseline and advanced CNN models, evaluate them across multiple MedMNIST datasets, and present results with clarity and reproducibility.

---

## 🚀 Current Progress

### ✅ Dataset: PathMNIST
- Colon tissue classification (9 classes)
- Image size: 28×28 RGB
- Split: 89,996 train / 10,004 val / 7,180 test

### ✅ Model: ResNet-18
- Trained for 10 epochs
- Val Accuracy: **98.04%**
- Val AUC: **0.9997**
- Model saved in `results/`
- Logs and plots generated

---

## 📁 Project Structure

MedMNIST/
│
├── scripts/ # Training, evaluation, plotting scripts
├── results/ # Logs and visualizations
├── models/ # Model definitions
├── data/ # Downloaded datasets (ignored by Git)
├── .gitignore
├── requirements.txt
└── README.md

# Hybrid CNN–Swin Transformer with FDCL

This repository provides an implementation of a *Hybrid CNN–Swin Transformer architecture* enhanced with a *Cross-Fusion Block (CFB)* and *Frequency-Domain Contrastive Learning (FDCL)* for remote sensing image classification.  
The model effectively combines local and global feature representations, improving robustness and classification accuracy on benchmark datasets.

---

## 🚀 Features
- *Tailored CNN* backbone for local texture representation  
- *Swin Transformer* backbone for capturing global context  
- *Cross-Fusion Block (CFB)* with residual and improved channel attention to unify features  
- *Frequency-Domain Contrastive Learning (FDCL)* to enhance discriminative power  
- Ablation-ready: supports training with and without FDCL, various loss functions, and optimizers  

---
## 📦 Installation

Clone this repository and install dependencies:

bash
pip install -r requirements.txt


Dependencies:
- TensorFlow 2.13.0  
- scikit-learn 1.0  
- matplotlib 3.7.3  
- numpy 1.24.3  
- pillow  

---

## 📂 Data Preparation

Datasets should be organized into class-subfolders, for example:


data/eurosat/
   ├── Forest/
   ├── River/
   └── ...


Resize and prepare with:

bash
python scripts/prepare_data.py --source /path/to/dataset --target data/eurosat256 --img_size 256


Works with *EuroSAT, **UC Merced, and **NWPU-RESISC45*.

---
## 🏋 Training

Run supervised training with FDCL:

bash
python -m src.training.train_fdcl   --data_root data/eurosat256   --epochs 50   --batch 32   --lr 0.00143   --momentum 0.0912   --lambda_ccl 0.5   --warmup_epochs 5


Key arguments:
- --lambda_ccl → weight for FDCL loss  
- --warmup_epochs → delay before enabling FDCL  
- --proj_dim, --temperature, --cutoff → FDCL parameters  

---

## 📊 Evaluation

Evaluate the best checkpoint:

bash
python -m src.eval.evaluate   --data_root data/eurosat256   --model_path experiments/ckpts/best_model


Outputs:
- Accuracy, Precision, Recall, F1  
- Confusion matrix  

---
## 📁 Project Structure


src/
 ├── datasets/loader.py
 ├── models/
 │   ├── tailored_cnn.py
 │   ├── swin_transformer.py
 │   ├── cross_fusion_block.py
 │   ├── heads.py
 │   └── fdcl_losses.py
 ├── training/train_fdcl.py
 └── eval/evaluate.py
scripts/prepare_data.py
requirements.txt
README.md


---

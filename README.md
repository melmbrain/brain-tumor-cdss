# Brain Tumor CDSS: Multimodal Deep Learning for Glioma Prognosis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.0+-green.svg)](https://monai.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **MRI + Gene Expression + Protein 데이터를 통합한 뇌종양(Glioma) 예후 예측 시스템**

멀티모달 딥러닝과 자체 전이학습 파이프라인을 활용하여 뇌종양 환자의 생존 예측, IDH 변이, MGMT 메틸화 상태를 예측합니다.

---

## 🎯 Key Features

- **자체 전이학습 파이프라인**: 외부 사전학습(ImageNet) 없이 뇌종양 특화 Encoder 구축
- **멀티모달 융합**: MRI (768-dim) + Gene (64-dim) + Protein (229-dim) 통합
- **VAE 기반 Gene Encoder**: 확률적 잠재 공간으로 노이즈 강건성 확보
- **Cross-Modal Attention**: 8-head attention으로 모달리티 간 상호작용 학습
- **분리 전략**: 학습(VAE) + 해석(Pathway) 분리로 성능과 해석력 모두 확보

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Brain Tumor CDSS Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Stage 1] Pre-training on Large-scale Data                     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  M1-Seg (1,242명) → SwinUNETR → 768-dim MRI features   │     │
│  │  MG (~1,000명)    → VAE Encoder → 64-dim Gene features │     │
│  └────────────────────────────────────────────────────────┘     │
│                              │                                   │
│                              ▼  Transfer Learning                │
│  [Stage 2] Multimodal Fusion (72명)                             │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  MM: 768 + 64 + 229 → Cross-Modal Attention → Predict  │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

![Architecture](docs/images/architecture.png)

---

## 📊 Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| **M1-Seg** | Tumor Segmentation | Dice Score | 0.766 |
| **M1-Cls** | IDH Mutation | AUC | 0.878 |
| **M1-Cls** | Grade Classification | Accuracy | 83.8% |
| **M1-Cls** | Survival Prediction | C-Index | 0.660 |
| **MG** | Survival Risk | C-Index | 0.780 |
| **MG** | Event Prediction | AUC | 0.850 |
| **MM** | Multimodal Survival | C-Index | 0.610 |

### Multimodal Fusion Effect
| Modality | C-Index |
|----------|---------|
| MRI only | 0.55 |
| Gene only | 0.58 |
| **MM (MRI+Gene+Protein)** | **0.61 (+5~6%p)** |

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, MONAI, transformers
- **MRI Processing**: SwinUNETR, nibabel, SimpleITK
- **Gene Analysis**: VAE, ssGSEA, Pathway Analysis
- **Survival Analysis**: Cox Proportional Hazards, lifelines
- **Backend**: FastAPI, Redis
- **Frontend**: React, Material-UI

---

## 📁 Project Structure

```
brain-tumor-cdss/
├── models/
│   ├── m1/                    # MRI Encoder (SwinUNETR)
│   │   ├── segmentation.py    # Tumor segmentation
│   │   └── classification.py  # IDH, MGMT, Grade, Survival
│   ├── mg/                    # Gene VAE Encoder
│   │   ├── vae.py             # VAE architecture
│   │   └── pathway.py         # Pathway interpretation
│   └── mm/                    # Multimodal Fusion
│       ├── attention.py       # Cross-Modal Attention
│       └── fusion.py          # Feature fusion
│
├── preprocessing/
│   ├── mri_preprocessing.py   # MRI normalization, skull stripping
│   ├── gene_preprocessing.py  # Gene expression normalization
│   └── data_pipeline.py       # Data loading utilities
│
├── training/
│   ├── train_m1.py            # M1 training script
│   ├── train_mg.py            # MG training script
│   └── train_mm.py            # MM training script
│
├── inference/
│   ├── predict.py             # Inference pipeline
│   └── demo.ipynb             # Interactive demo
│
├── configs/
│   └── default.yaml           # Hyperparameters
│
├── docs/
│   ├── ARCHITECTURE.md        # Detailed architecture
│   └── EXPERIMENTS.md         # Experiment results
│
├── samples/                   # Sample data for demo
└── weights/                   # Pretrained weights (see below)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/brain-tumor-cdss.git
cd brain-tumor-cdss
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

```bash
# Download from GitHub Releases
wget https://github.com/yourusername/brain-tumor-cdss/releases/download/v1.0/weights.zip
unzip weights.zip -d weights/
```

### 3. Run Inference

```python
from inference.predict import BrainTumorPredictor

predictor = BrainTumorPredictor(
    m1_weights="weights/m1_seg.pth",
    mg_weights="weights/mg_vae.pth",
    mm_weights="weights/mm_fusion.pth"
)

result = predictor.predict(
    mri_path="samples/patient_001/mri.nii.gz",
    gene_expression="samples/patient_001/gene.csv",
    protein_data="samples/patient_001/protein.csv"
)

print(result)
# {
#     "survival_risk": 0.65,
#     "idh_mutation": "Mutant",
#     "mgmt_methylation": "Methylated",
#     "grade": "Grade IV",
#     "pathway_interpretation": {...}
# }
```

---

## 📈 Experiments

### Gene Encoder Comparison

| Method | C-Index | Event AUC | Notes |
|--------|---------|-----------|-------|
| DEG + Pathway | 0.766 | 0.864 | Overfitting risk |
| Gene2Vec + DEG | 0.786 | 0.844 | High variance |
| **VAE (Ours)** | **0.780** | **0.850** | Stable, transfer-friendly |

### Transfer Learning Validation

M1 모델이 BraTS 데이터로 학습 후 TCGA 데이터에서도 동등한 성능 유지:
- BraTS Validation: Dice 0.766
- TCGA (MM data): Dice ≥ 0.766 ✓

→ 일반화 성공, 표준화된 segmentation 파이프라인 확보

### Separation Strategy (학습/해석 분리)

```
[Training] Gene → VAE Encoder → Predictions
                    ↓
              64-dim latent → Transfer to MM

[Inference] Gene → VAE (frozen) → Predictions
              ↓
           ssGSEA → 50 Hallmark Pathways → Interpretation
```

---

## 📚 Dataset

| Dataset | Patients | Usage |
|---------|----------|-------|
| BraTS 2021 | 1,251 | M1 pre-training (MRI) |
| CGGA | ~1,000 | MG pre-training (Gene) |
| TCGA-GBM/LGG | 72 | MM multimodal fusion |

**Data Leakage Prevention**: MM의 72명은 M1 학습에서 제외

---

## 🔬 Technical Highlights

### 1. Self-built Transfer Learning
외부 사전학습 모델(ImageNet) 대신 뇌종양 데이터로 직접 Encoder 학습
- 도메인 특화된 feature 추출
- 소규모 MM 데이터(72명)에도 효과적 전이

### 2. VAE-based Gene Encoder
```python
# Reparameterization Trick
z = mu + sigma * epsilon  # epsilon ~ N(0, 1)
```
- 확률적 잠재 공간으로 노이즈 강건성 확보
- KL Divergence로 정규화 → 일반화 향상

### 3. Cross-Modal Attention
```python
Attention(Q, K, V) = softmax(QK^T / √d) × V
```
- 8 attention heads
- 모달리티 간 상호작용 학습
- 상호보완적 정보 활용

### 4. Cox Proportional Hazards
```
h(t|x) = h₀(t) × exp(risk_score)
```
- C-Index: 두 환자 중 누가 먼저 사망할지 맞추는 정확도
- 0.5 = Random, 1.0 = Perfect

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **Author**: [Your Name]
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn]

---

## 🙏 Acknowledgments

- [MONAI](https://monai.io/) for medical imaging deep learning
- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats/) for MRI dataset
- [CGGA](http://www.cgga.org.cn/) for gene expression data

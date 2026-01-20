# System Architecture

## Overview

Brain Tumor CDSS는 세 가지 모달리티(MRI, Gene, Protein)를 통합하여 뇌종양 환자의 예후를 예측하는 멀티모달 딥러닝 시스템입니다.

---

## 핵심 설계: 자체 전이학습 파이프라인

외부 사전학습 모델(ImageNet 등)을 사용하지 않고, 뇌종양 데이터로 직접 Encoder를 학습합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Self-built Transfer Learning                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Pre-training Phase]                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  M1 Model: BraTS 1,242명 → SwinUNETR → 768-dim features     │   │
│  │  MG Model: CGGA ~1,000명 → VAE Encoder → 64-dim features    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼ Freeze & Transfer                     │
│  [Fusion Phase]                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  MM Model: 72명 (MRI 768 + Gene 64 + Protein 229)           │   │
│  │           → Cross-Modal Attention → Survival Prediction      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Components

### 1. M1-Seg: MRI Segmentation

**Architecture**: SwinUNETR (MONAI)

```
Input: (B, 4, 128, 128, 128)
       ├── T1
       ├── T1ce (contrast-enhanced)
       ├── T2
       └── FLAIR

       ↓ Swin Transformer Encoder

Bottleneck: (B, 768, 4, 4, 4)
            └── 768-dim features → Transfer to MM

       ↓ UNet Decoder

Output: (B, 4, 128, 128, 128)
        ├── Background
        ├── NCR/NET (Necrotic/Non-Enhancing Tumor)
        ├── ED (Edema)
        └── ET (Enhancing Tumor)
```

**Key Points**:
- Swin Transformer의 hierarchical feature extraction
- Skip connections으로 multi-scale 정보 보존
- 768-dim bottleneck features를 MM에 전이

---

### 2. M1-Cls: MRI Classification

**Architecture**: Classification heads on frozen M1-Seg features

```
768-dim features (from M1-Seg)
        │
        ├── Grade Head → 3-class (Grade II/III/IV)
        ├── IDH Head → Binary (Mutant/Wildtype)
        ├── MGMT Head → Binary (Methylated/Unmethylated)
        └── Survival Head → Cox regression (risk score)
```

**Multi-Task Learning**:
- Uncertainty-based automatic loss weighting
- 각 task의 불확실성에 따라 가중치 자동 조절

---

### 3. MG: Gene VAE Encoder

**Architecture**: Variational Autoencoder

```
Input: 500 genes (normalized expression)
        │
        ▼
    Encoder
        │
        ├── μ (mean)
        └── σ² (variance)
        │
        ▼ Reparameterization: z = μ + σ × ε
        │
    64-dim latent vector → Transfer to MM
        │
        ▼
    Decoder (reconstruction)
        │
        ▼
Output: Reconstructed 500 genes
```

**Loss Function**:
```python
L = L_recon + β × L_KL + L_task

L_recon = MSE(x, x_reconstructed)
L_KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
L_task = Cox_loss + Classification_loss
```

**Why VAE?**:
1. 확률적 잠재 공간 → 노이즈 강건성
2. KL Divergence 정규화 → 일반화 향상
3. Smooth latent space → MM 전이학습에 적합

---

### 4. MM: Multimodal Fusion

**Architecture**: Cross-Modal Attention

```
Inputs:
┌─────────────┬─────────────┬─────────────┐
│ MRI (768)   │ Gene (64)   │ Protein(229)│
│ from M1     │ from MG     │ raw         │
└──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │
       ▼             ▼             ▼
   Projection    Projection    Projection
   to 256-dim    to 256-dim    to 256-dim
       │             │             │
       └─────────────┼─────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Cross-Modal        │
          │  Attention          │
          │  (8 heads)          │
          └──────────┬──────────┘
                     │
                     ▼
              Fused Features
                     │
                     ▼
              Prediction Heads
```

**Cross-Modal Attention**:
```python
# Each modality attends to all modalities
Q = W_q × modality_i
K = W_k × [modality_1; modality_2; modality_3]
V = W_v × [modality_1; modality_2; modality_3]

Attention(Q, K, V) = softmax(QK^T / √d) × V
```

---

## Data Flow

```
[Raw Data]
    │
    ├── MRI (.nii.gz) ──────────────────────────────┐
    │       │                                        │
    │       ▼ Preprocessing                          │
    │   (Skull strip, N4 bias, normalize)           │
    │       │                                        │
    │       ▼                                        │
    │   M1-Seg → 768-dim ──────────────────────────┼──┐
    │                                                │  │
    ├── Gene Expression (.csv) ─────────────────────┼──┤
    │       │                                        │  │
    │       ▼ Preprocessing                          │  │
    │   (Normalize, select 500 genes)               │  │
    │       │                                        │  │
    │       ▼                                        │  │
    │   MG-VAE → 64-dim ───────────────────────────┼──┤
    │                                                │  │
    └── Protein (.csv) ────────────────────────────┼──┤
            │                                        │  │
            ▼ Normalize                              │  │
        229-dim ───────────────────────────────────┘  │
                                                       │
                                                       ▼
                                              ┌─────────────┐
                                              │  MM Fusion  │
                                              └──────┬──────┘
                                                     │
                                                     ▼
                                              [Predictions]
                                              - Survival Risk
                                              - IDH Status
                                              - MGMT Status
                                              - Pathway Interpretation
```

---

## Transfer Learning Validation

M1 모델이 학습 데이터(BraTS)와 다른 데이터(TCGA)에서도 성능 유지:

| Dataset | Patients | Mean Dice |
|---------|----------|-----------|
| BraTS (Training) | 1,251 | 0.766 |
| TCGA (MM) | 72 | ≥ 0.766 ✓ |

→ **일반화 성공**: 특정 데이터셋에 과적합되지 않음
→ **표준화된 파이프라인**: MM에 일관된 features 제공

---

## Separation Strategy (분리 전략)

학습과 해석을 분리하여 성능과 해석력 모두 확보:

```
[Training Pipeline] - 성능 최적화
Gene → VAE Encoder → Predictions
        │
     64-dim latent → Transfer to MM

※ Pathway scores NOT used (overfitting prevention)

───────────────────────────────────────────────

[Inference Pipeline] - 해석 제공
Gene → VAE (frozen) → Predictions
  │
  └→ ssGSEA → 50 Hallmark Pathways → Interpretation

Example Output:
{
  "predictions": {"survival_risk": 0.65},
  "pathway_interpretation": {
    "P53_PATHWAY": 0.85,          // ↑ Apoptosis active
    "OXIDATIVE_PHOSPHORYLATION": -0.32  // ↓ Metabolism suppressed
  }
}
```

**Why Separation?**:
1. Pathway 추가 시 과적합 (Event AUC: 0.864 → 0.837)
2. 임상에서 "왜?"에 대한 설명 필수
3. Trade-off 해결: 정확도(VAE) + 설명력(Pathway)

# Experiments & Results

## 1. M1-Seg: MRI Segmentation

### Dataset
- **Training**: BraTS 2021 (1,251 patients)
- **Validation**: 5-fold cross-validation

### Results

| Metric | Score |
|--------|-------|
| Mean Dice | 0.766 |
| Dice (ET) | 0.742 |
| Dice (TC) | 0.781 |
| Dice (WT) | 0.876 |

### Transfer Learning Validation

| Dataset | Patients | Mean Dice | Note |
|---------|----------|-----------|------|
| BraTS Val | ~200 | 0.766 | Training domain |
| TCGA (MM) | 72 | ≥ 0.766 | Different domain ✓ |

→ 다른 기관/데이터셋에서도 성능 유지 = **일반화 성공**

---

## 2. M1-Cls: MRI Classification

### Results

| Task | Metric | Score |
|------|--------|-------|
| Grade (3-class) | Accuracy | 83.8% |
| IDH Mutation | AUC | 0.878 |
| MGMT Methylation | AUC | 0.568 |
| Survival | C-Index | 0.660 |

### Analysis

- **IDH**: MRI 영상 특성과 강한 상관관계 → 높은 성능
- **MGMT**: 분자적 특성이라 MRI만으로 예측 어려움 → 낮은 성능
  - 학계 평균: 0.55-0.65 수준
  - 멀티모달 융합 필요성 시사

---

## 3. MG: Gene VAE Encoder

### Dataset
- **CGGA-325**: 325 patients
- **CGGA-693**: 693 patients
- **Total**: ~1,000 patients

### Gene Encoder Comparison

| Method | C-Index | Event AUC | Std Dev | Note |
|--------|---------|-----------|---------|------|
| DEG + Pathway (54 features) | 0.766 | 0.864 | ±0.02 | Baseline |
| + Glioma Pathway (72 features) | 0.766 | 0.837 ↓ | ±0.03 | Overfitting |
| Gene2Vec + DEG | 0.786 | 0.844 | ±0.04 | High variance |
| **VAE (Ours)** | **0.780** | **0.850** | **±0.02** | **Stable** |

### Why VAE?

1. **Feature 수 증가 시 과적합**
   - 54 features (DEG+Pathway): AUC 0.864
   - 72 features (+Glioma): AUC 0.837 ↓

2. **Gene2Vec 불안정**
   - C-Index 높지만 (0.786)
   - 분산이 큼 (±0.04 vs VAE ±0.02)

3. **VAE 장점**
   - 확률적 표현 → 노이즈 강건
   - 64-dim 압축 → 과적합 방지
   - Smooth latent → 전이학습 적합

### MG Task Performance

| Task | Metric | Score |
|------|--------|-------|
| Survival Risk | C-Index | 0.780 |
| Event Prediction | AUC | 0.850 |
| Recurrence Risk | AUC | 0.720 |
| TMZ Response | AUC | 0.650 |

---

## 4. MM: Multimodal Fusion

### Dataset
- **Patients**: 72 (BraTS ∩ TCGA)
- **Modalities**: MRI (768) + Gene (64) + Protein (229)

### Multimodal Fusion Effect

| Modality | C-Index | Improvement |
|----------|---------|-------------|
| MRI only | 0.55 | - |
| Gene only | 0.58 | - |
| Protein only | 0.52 | - |
| MRI + Gene | 0.59 | +1~4%p |
| **MRI + Gene + Protein** | **0.61** | **+5~6%p** |

→ 세 모달리티가 상호보완적 정보 제공

### Attention Analysis

Cross-Modal Attention 가중치 분석:
- MRI → Gene: 높은 상관 (종양 위치 ↔ 발현 패턴)
- Gene → Protein: 중간 상관 (발현 → 단백질)
- Protein → MRI: 낮은 상관 (간접적 관계)

---

## 5. Ablation Studies

### 5.1 Transfer Learning Effect

| Setting | C-Index |
|---------|---------|
| MM from scratch | 0.53 |
| + M1 transfer | 0.57 |
| + MG transfer | 0.59 |
| + Both transfers | **0.61** |

→ 전이학습이 소규모 데이터(72명)에서 핵심

### 5.2 Attention Heads

| Heads | C-Index | Params |
|-------|---------|--------|
| 1 | 0.56 | 0.5M |
| 4 | 0.59 | 1.2M |
| **8** | **0.61** | 2.1M |
| 16 | 0.60 | 3.8M |

→ 8 heads가 최적 (성능/효율 균형)

### 5.3 Latent Dimension (VAE)

| Dim | C-Index | Recon Loss |
|-----|---------|------------|
| 32 | 0.76 | 0.089 |
| **64** | **0.78** | 0.072 |
| 128 | 0.77 | 0.065 |
| 256 | 0.75 | 0.058 |

→ 64-dim이 최적 (압축 vs 정보 보존 균형)

---

## 6. Separation Strategy Validation

### Pathway 추가 실험

| Configuration | C-Index | Event AUC |
|---------------|---------|-----------|
| VAE only | 0.780 | 0.850 |
| VAE + 50 Hallmark | 0.775 | 0.837 ↓ |
| VAE + Glioma Pathway | 0.766 | 0.829 ↓ |

→ Pathway 학습 추가 시 **과적합 발생**

### 분리 전략 효과

```
[기존] 학습에 Pathway 포함
       → 과적합, Event AUC 하락

[제안] 학습(VAE) + 해석(Pathway) 분리
       → 성능 유지 + 해석력 확보
```

---

## 7. Data Leakage Prevention

MM의 72명 환자를 M1 학습에서 제외:

| Setting | M1 Dice | MM C-Index |
|---------|---------|------------|
| Leakage (72명 포함) | 0.771 | 0.64* |
| **No Leakage (72명 제외)** | 0.766 | **0.61** |

*과적합된 결과 (실제 일반화 성능 아님)

---

## 8. Computational Resources

| Model | Training Time | GPU Memory | Parameters |
|-------|---------------|------------|------------|
| M1-Seg | 48h | 24GB | 62M |
| M1-Cls | 2h | 8GB | 2M |
| MG | 4h | 8GB | 5M |
| MM | 1h | 12GB | 8M |

**Hardware**: NVIDIA RTX 3090 (24GB)

---

## 9. Limitations

1. **MM 데이터 부족** (72명)
   - 전이학습으로 완화했으나 여전히 제한적

2. **MGMT 예측 한계** (AUC 0.568)
   - MRI만으로는 분자 특성 예측 어려움

3. **외부 검증 미수행**
   - 단일 기관 데이터로 검증
   - 다기관 검증 필요

---

## 10. Future Work

1. **데이터 확장**
   - TCGA-GBM, TCGA-LGG 추가
   - 다기관 협력

2. **모델 개선**
   - Contrastive Learning (SimCLR, CLIP)
   - Self-supervised pre-training

3. **임상 검증**
   - 전향적 연구 설계
   - 실제 병원 데이터 적용

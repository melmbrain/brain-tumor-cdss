# MM Model: Multimodal Fusion

## Overview

MM is a **Cross-Modal Attention** model that fuses:

- MRI features (from M1)
- Gene expression features (from MG)
- Protein data (RPPA)
- Clinical information

## Architecture

```mermaid
flowchart TB
    subgraph Inputs
        MRI[MRI Features<br/>768-dim]
        GENE[Gene Features<br/>64-dim]
        PROT[Protein<br/>229-dim]
        CLIN[Clinical<br/>10-dim]
    end

    subgraph Projection
        MRI_P[Project to 256]
        GENE_P[Project to 256]
        PROT_P[Project to 256]
        CLIN_P[Project to 256]
    end

    subgraph Attention["Cross-Modal Attention (8 heads)"]
        ATT[Multi-Head<br/>Self-Attention]
    end

    subgraph Outputs
        SURV[Survival<br/>C-Index 0.61]
        IDH[IDH<br/>AUC 0.878]
        MGMT[MGMT]
        GRADE[Grade]
    end

    MRI --> MRI_P
    GENE --> GENE_P
    PROT --> PROT_P
    CLIN --> CLIN_P

    MRI_P & GENE_P & PROT_P & CLIN_P --> ATT
    ATT --> SURV & IDH & MGMT & GRADE

    style ATT fill:#9933FF
```

## Model Specifications

| Parameter | Value |
|-----------|-------|
| MRI Input | 768-dim |
| Gene Input | 64-dim |
| Protein Input | 229-dim (167 RPPA) |
| Clinical Input | 10-dim |
| Hidden Dimension | 256 |
| Attention Heads | 8 |
| Output Tasks | 7 classification + survival |
| Parameters | ~2M |

## Cross-Modal Attention

The attention mechanism learns which modality combinations are informative:

```python
# Multi-head attention
attention = MultiheadAttention(
    embed_dim=256,
    num_heads=8,
    batch_first=True
)

# Stack modality embeddings as sequence
# [batch, 4, 256] - 4 modalities
modal_embeddings = torch.stack([mri, gene, protein, clinical], dim=1)

# Self-attention across modalities
attended, attention_weights = attention(
    modal_embeddings,
    modal_embeddings,
    modal_embeddings
)
```

## Usage

```python
from models.mm import MultimodalFusionModel, MMInference

# Initialize
mm = MMInference(checkpoint_path='weights/mm_best.pt')

# Predict
result = mm.predict(
    mri_features=mri_feat,      # From M1
    gene_features=gene_feat,    # From MG
    protein_data=protein_tensor,
    clinical_data=clinical_tensor
)

print(f"IDH: {result['idh_prediction']}")
print(f"MGMT: {result['mgmt_prediction']}")
print(f"Survival: {result['survival_prediction']}")
```

## Output Tasks

| Task | Classes | Metric |
|------|---------|--------|
| Grade | 4 (G1-G4) | Accuracy |
| IDH | 2 (WT/Mut) | AUC |
| MGMT | 2 (Unmeth/Meth) | AUC |
| 1p/19q | 2 (Intact/Codel) | AUC |
| Histology | 3 | Accuracy |
| Gender | 2 | Accuracy |
| Race | 3 | Accuracy |
| Survival | Continuous | C-Index |

## Performance

| Task | Metric | Score |
|------|--------|-------|
| Survival | C-Index | 0.610 |
| IDH | AUC | 0.878 |
| MGMT | AUC | 0.65 |
| Grade | Accuracy | 0.70 |

## Training

The MM model is trained with frozen M1 and MG encoders:

```python
# Freeze encoders
for param in m1_encoder.parameters():
    param.requires_grad = False
for param in mg_encoder.parameters():
    param.requires_grad = False

# Train only fusion layers
optimizer = Adam(mm_model.parameters(), lr=1e-4)
```

## Attention Visualization

```python
# Get attention weights
outputs = mm_model(mri, gene, protein, clinical, return_attention=True)

# Visualize which modalities attend to each other
attention_weights = outputs['modal_attention']  # [batch, heads, 4, 4]
```

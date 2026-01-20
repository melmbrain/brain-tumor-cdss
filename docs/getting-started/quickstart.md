# Quick Start

This guide will help you get started with Brain Tumor CDSS in just a few steps.

## 1. Basic Usage

### Load Models

```python
from models.m1 import M1Inference
from models.mg import MGInference
from models.mm import MMInference

# Initialize inference engines
m1 = M1Inference(checkpoint_path='weights/m1_best.pth', device='cuda')
mg = MGInference(checkpoint_path='weights/mg_best.pt', device='cuda')
mm = MMInference(checkpoint_path='weights/mm_best.pt', device='cuda')
```

### MRI Analysis (M1)

```python
# Analyze MRI scan
result = m1.analyze(
    patient_id='patient_001',
    mri_path='/path/to/mri_folder/',  # Contains T1, T1ce, T2, FLAIR
    include_segmentation=True
)

print(f"Segmentation shape: {result['segmentation'].shape}")
print(f"IDH prediction: {result['idh_prediction']}")
print(f"Grade: {result['grade_prediction']}")
```

### Gene Expression Analysis (MG)

```python
import pandas as pd

# Load gene expression data
gene_df = pd.read_csv('patient_expression.csv', index_col=0)
gene_expression = gene_df.iloc[0].to_dict()

# Analyze
result = mg.analyze(
    patient_id='patient_001',
    gene_expression=gene_expression,
    include_explainability=True
)

print(f"Survival Risk: {result['survival_risk']['category']}")
print(f"Grade: {result['grade_prediction']['predicted']}")
```

### Multimodal Fusion (MM)

```python
import torch

# Get features from M1 and MG
mri_features = m1.extract_features(mri_path)
gene_features = mg.extract_features(gene_expression)

# Protein data (RPPA)
protein_data = torch.randn(1, 167)  # Your actual protein data

# Clinical data
clinical_data = torch.tensor([[
    65,    # age
    1,     # sex (0: female, 1: male)
    90,    # KPS
    1,     # tumor location
    # ... other clinical features
]])

# Run multimodal fusion
result = mm.predict(
    mri_features=mri_features,
    gene_features=gene_features,
    protein_data=protein_data,
    clinical_data=clinical_data
)

print(f"IDH: {result['idh_prediction']}")
print(f"MGMT: {result['mgmt_prediction']}")
print(f"Survival Risk: {result['survival_risk']}")
```

## 2. Using the Demo Notebook

The easiest way to explore the system is through our demo notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melmbrain/brain-tumor-cdss/blob/main/notebooks/demo.ipynb)

## 3. Command Line Interface

```bash
# Run inference
python inference/predict.py \
    --mri_path /path/to/mri \
    --gene_file /path/to/expression.csv \
    --output_dir /path/to/output
```

## 4. Sample Data

Test with our sample data:

```python
from sample_data import load_sample_data

# Load synthetic sample data
sample = load_sample_data()

# Run analysis
result = mg.analyze(
    patient_id='sample_patient',
    gene_expression=sample['gene_expression'],
    pathway_scores=sample['pathway_scores']
)
```

## Next Steps

- Read the [Architecture Overview](../architecture/overview.md) to understand the system design
- Check the [API Reference](../api/inference.md) for detailed documentation
- Explore [Experiments](../experiments.md) to see our training results

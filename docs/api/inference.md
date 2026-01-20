# Inference API Reference

## M1Inference

MRI analysis and feature extraction.

### Constructor

```python
M1Inference(
    checkpoint_path: str = None,
    device: str = 'cuda',
    include_segmentation: bool = True
)
```

**Parameters:**

- `checkpoint_path`: Path to model weights (.pth file)
- `device`: 'cuda' or 'cpu'
- `include_segmentation`: Whether to generate segmentation masks

### Methods

#### analyze()

```python
def analyze(
    self,
    patient_id: str,
    mri_path: str,
    include_segmentation: bool = True,
    include_features: bool = True
) -> Dict
```

**Parameters:**

- `patient_id`: Unique patient identifier
- `mri_path`: Path to folder containing MRI files (T1, T1ce, T2, FLAIR)
- `include_segmentation`: Generate tumor segmentation
- `include_features`: Extract 768-dim features

**Returns:**

```python
{
    'patient_id': str,
    'segmentation': np.ndarray,  # (H, W, D) with labels 0-3
    'features': np.ndarray,      # (768,) feature vector
    'idh_prediction': str,       # 'Wildtype' or 'Mutant'
    'idh_probability': float,
    'grade_prediction': str,     # 'G2', 'G3', or 'G4'
    'grade_probabilities': dict,
    'survival_risk': float
}
```

#### extract_features()

```python
def extract_features(self, mri_path: str) -> torch.Tensor
```

Extract features only, without full analysis.

---

## MGInference

Gene expression analysis.

### Constructor

```python
MGInference(
    checkpoint_path: str = None,
    gene_names: List[str] = None,
    device: str = 'cuda'
)
```

### Methods

#### analyze()

```python
def analyze(
    self,
    patient_id: str,
    gene_expression: Dict[str, float],
    pathway_scores: Dict[str, float] = None,
    include_explainability: bool = True
) -> Dict
```

**Parameters:**

- `patient_id`: Unique patient identifier
- `gene_expression`: Dict mapping gene names to expression values
- `pathway_scores`: Optional pre-computed pathway scores
- `include_explainability`: Include pathway interpretation

**Returns:**

```python
{
    'patient_id': str,
    'survival_risk': {
        'score': float,
        'category': str  # 'Low', 'Medium', or 'High'
    },
    'grade_prediction': {
        'predicted': str,
        'confidence': float,
        'probabilities': dict
    },
    'survival_time': {
        'predicted_months': float
    },
    'recurrence': {
        'prediction': str,
        'probability': float
    },
    'latent_features': np.ndarray,  # (64,)
    'pathway_analysis': dict  # If include_explainability=True
}
```

#### extract_features()

```python
def extract_features(self, gene_expression: Dict[str, float]) -> torch.Tensor
```

Extract 64-dim latent features.

---

## MMInference

Multimodal fusion analysis.

### Constructor

```python
MMInference(
    checkpoint_path: str = None,
    m1_checkpoint: str = None,
    mg_checkpoint: str = None,
    device: str = 'cuda'
)
```

### Methods

#### predict()

```python
def predict(
    self,
    mri_features: torch.Tensor = None,
    gene_features: torch.Tensor = None,
    protein_data: torch.Tensor = None,
    clinical_data: torch.Tensor = None,
    return_attention: bool = False
) -> Dict
```

**Parameters:**

- `mri_features`: (batch, 768) from M1
- `gene_features`: (batch, 64) from MG
- `protein_data`: (batch, 167) RPPA data
- `clinical_data`: (batch, 10) clinical features
- `return_attention`: Include attention weights

**Returns:**

```python
{
    'grade_prediction': str,
    'grade_probabilities': dict,
    'idh_prediction': str,
    'idh_probability': float,
    'mgmt_prediction': str,
    'mgmt_probability': float,
    'survival_risk': float,
    'survival_category': str,
    'attention_weights': np.ndarray  # If return_attention=True
}
```

---

## CLI Interface

```bash
# Full pipeline
python inference/predict.py \
    --patient_id patient_001 \
    --mri_path /path/to/mri \
    --gene_file /path/to/expression.csv \
    --protein_file /path/to/rppa.csv \
    --clinical_file /path/to/clinical.csv \
    --output_dir /path/to/output \
    --device cuda

# MRI only
python inference/predict.py \
    --patient_id patient_001 \
    --mri_path /path/to/mri \
    --mode mri_only

# Gene only
python inference/predict.py \
    --patient_id patient_001 \
    --gene_file /path/to/expression.csv \
    --mode gene_only
```

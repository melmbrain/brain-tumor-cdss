# Sample Data

This directory contains synthetic sample data for testing and demonstration purposes.

## Usage

```python
from sample_data import load_sample_data, load_sample_mri, load_sample_protein

# Load all sample data
data = load_sample_data()
print(data['gene_expression'])  # 100 genes
print(data['pathway_scores'])   # 6 pathways
print(data['clinical_data'])    # Clinical features

# Load sample MRI (synthetic)
mri = load_sample_mri()
print(mri.shape)  # (4, 128, 128, 128)

# Load sample protein data
protein = load_sample_protein()
print(protein.shape)  # (167,)
```

## Data Description

| Type | Description | Shape/Size |
|------|-------------|------------|
| Gene Expression | Z-score normalized expression values | 100 genes |
| Pathway Scores | Hallmark pathway enrichment scores | 6 pathways |
| Clinical Data | Patient demographics and clinical features | Dict |
| MRI | Synthetic 4-channel MRI volume | (4, 128, 128, 128) |
| Protein | Synthetic RPPA protein levels | (167,) |

## Important Note

**This is synthetic data generated for testing purposes only.**

- Gene expression values are randomly generated
- MRI data is random noise with a simulated tumor sphere
- Protein values are randomly generated
- Clinical data represents a hypothetical patient

For real analysis, use actual patient data that has been:
1. Properly preprocessed
2. IRB-approved for research use
3. De-identified according to HIPAA guidelines

## Real Data Sources

For actual research, consider these public datasets:

| Dataset | Type | Access |
|---------|------|--------|
| [BraTS](https://www.synapse.org/brats) | MRI | Registration required |
| [TCGA-GBM](https://portal.gdc.cancer.gov/) | Multi-omics | Open access |
| [CGGA](http://www.cgga.org.cn/) | Gene expression | Open access |

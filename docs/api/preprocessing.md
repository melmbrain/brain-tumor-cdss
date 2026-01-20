# Preprocessing API Reference

## MRIPreprocessor

Preprocessing pipeline for brain MRI data.

### Constructor

```python
MRIPreprocessor(
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_size: Tuple[int, int, int] = (128, 128, 128),
    normalize_method: str = 'zscore',
    apply_n4: bool = True,
    apply_skull_strip: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| target_spacing | Tuple[float, float, float] | (1.0, 1.0, 1.0) | Target voxel spacing in mm |
| target_size | Tuple[int, int, int] | (128, 128, 128) | Target volume dimensions |
| normalize_method | str | 'zscore' | Normalization: 'zscore', 'minmax', or 'percentile' |
| apply_n4 | bool | True | Apply N4 bias field correction |
| apply_skull_strip | bool | False | Apply skull stripping |

### Methods

#### process_patient()

```python
def process_patient(
    self,
    patient_dir: str,
    output_dir: str = None,
    save_intermediate: bool = False
) -> Dict[str, np.ndarray]
```

Process a single patient's MRI data.

**Parameters:**

- `patient_dir`: Directory containing MRI files
- `output_dir`: Optional output directory
- `save_intermediate`: Save intermediate processing results

**Returns:**

```python
{
    'image': np.ndarray,        # (4, H, W, D) - T1, T1ce, T2, FLAIR
    'segmentation': np.ndarray, # (H, W, D) if available
    'affine': np.ndarray,       # (4, 4) affine matrix
    'patient_id': str,
    'original_shape': tuple
}
```

### Example

```python
from preprocessing.mri_preprocessing import MRIPreprocessor

# Initialize
preprocessor = MRIPreprocessor(
    target_size=(128, 128, 128),
    normalize_method='zscore',
    apply_n4=True
)

# Process single patient
result = preprocessor.process_patient(
    patient_dir='/path/to/BraTS2021_00001',
    output_dir='/path/to/output'
)

print(f"Processed shape: {result['image'].shape}")
# Output: Processed shape: (4, 128, 128, 128)
```

---

## batch_process()

Process multiple patients in parallel.

```python
batch_process(
    input_dir: str,
    output_dir: str,
    n_workers: int = 4,
    **kwargs
)
```

**Parameters:**

- `input_dir`: Directory containing patient folders
- `output_dir`: Output directory
- `n_workers`: Number of parallel workers
- `**kwargs`: Arguments passed to MRIPreprocessor

### Example

```python
from preprocessing.mri_preprocessing import batch_process

batch_process(
    input_dir='/data/BraTS2021',
    output_dir='/data/processed',
    n_workers=8,
    target_size=(128, 128, 128),
    normalize_method='zscore'
)
```

---

## Normalization Methods

### Z-score Normalization

```python
# Default method
# Normalizes to zero mean and unit variance within brain mask
normalized = (volume - mean) / std
```

### Min-Max Normalization

```python
# Scales to [0, 1] range
normalized = (volume - min) / (max - min)
```

### Percentile Normalization

```python
# Clips to 1-99 percentile, then scales to [0, 1]
p1, p99 = np.percentile(volume[mask], [1, 99])
normalized = np.clip(volume, p1, p99)
normalized = (normalized - p1) / (p99 - p1)
```

---

## CLI Usage

```bash
# Single patient
python preprocessing/mri_preprocessing.py \
    --input_dir /path/to/patient \
    --output_dir /path/to/output \
    --normalize zscore

# Batch processing
python preprocessing/mri_preprocessing.py \
    --input_dir /data/BraTS2021 \
    --output_dir /data/processed \
    --n_workers 8 \
    --target_size 128 128 128 \
    --normalize zscore
```

---

## File Naming Conventions

The preprocessor supports multiple naming conventions:

| Pattern | Example |
|---------|---------|
| BraTS style | `BraTS2021_00001_t1.nii.gz` |
| Simple | `t1.nii.gz` |
| Upper case | `T1.nii.gz` |

Segmentation files:
- `*_seg.nii.gz`
- `seg.nii.gz`

---

## Dependencies

Required packages:

```bash
pip install nibabel SimpleITK scipy numpy
```

Optional (for advanced skull stripping):

```bash
pip install antspy  # ANTs Python wrapper
```

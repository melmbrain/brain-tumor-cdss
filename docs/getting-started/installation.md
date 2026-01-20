# Installation

## Requirements

- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA 11.0+ (for GPU support)

## Installation Methods

### Option 1: pip install (Recommended)

```bash
pip install brain-tumor-cdss
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/melmbrain/brain-tumor-cdss.git
cd brain-tumor-cdss

# Install in development mode
pip install -e .

# Or install with all extras
pip install -e .[dev,docs,api]
```

### Option 3: Docker

```bash
# Pull the image
docker pull melmbrain/brain-tumor-cdss:latest

# Or build locally
docker build -t brain-tumor-cdss .

# Run
docker run -it brain-tumor-cdss
```

## Download Pretrained Weights

Download the pretrained model weights from [GitHub Releases](https://github.com/melmbrain/brain-tumor-cdss/releases/tag/v1.0.0):

```bash
# Create weights directory
mkdir -p weights

# Download weights (using curl)
curl -L -o weights/m1_best.pth https://github.com/melmbrain/brain-tumor-cdss/releases/download/v1.0.0/m1_best.pth
curl -L -o weights/mg_best.pt https://github.com/melmbrain/brain-tumor-cdss/releases/download/v1.0.0/mg_4tasks_best.pt
curl -L -o weights/mm_best.pt https://github.com/melmbrain/brain-tumor-cdss/releases/download/v1.0.0/mm_best.pt
```

## Verify Installation

```python
import torch
from models.m1 import MRIMultiTaskModel
from models.mg import GeneExpressionCDSS
from models.mm import MultimodalFusionModel

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test model loading
m1 = MRIMultiTaskModel(in_channels=4)
print(f"M1 model loaded: {sum(p.numel() for p in m1.parameters()):,} parameters")
```

## Dependencies

Core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥1.12.0 | Deep learning framework |
| monai | ≥1.0.0 | Medical imaging |
| nibabel | ≥4.0.0 | NIfTI file handling |
| numpy | ≥1.21.0 | Numerical computing |
| pandas | ≥1.3.0 | Data manipulation |
| scikit-learn | ≥1.0.0 | Machine learning utilities |

Optional dependencies:

| Package | Purpose |
|---------|---------|
| SimpleITK | Advanced image processing |
| gseapy | Gene set enrichment |
| lifelines | Survival analysis |

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

```python
# Use smaller batch size
batch_size = 1

# Or use CPU
device = 'cpu'
```

### MONAI Import Error

If MONAI is not installed:

```bash
pip install monai[all]
```

### Missing nibabel

```bash
pip install nibabel
```

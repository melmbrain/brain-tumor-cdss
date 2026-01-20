# Release v1.0.0

## Brain Tumor CDSS - Initial Release

### Model Weights

| File | Size | Description |
|------|------|-------------|
| `m1_best.pth` | 247 MB | MRI Encoder (SwinUNETR) - Segmentation & Classification |
| `mg_4tasks_best.pt` | 2.1 MB | Gene VAE Encoder - 4 Task Predictions |
| `mm_best.pt` | 5.9 MB | Multimodal Fusion Model |

### Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| M1-Seg | Tumor Segmentation | Dice | 0.766 |
| M1-Cls | IDH Mutation | AUC | 0.878 |
| M1-Cls | Grade Classification | Accuracy | 83.8% |
| MG | Survival Prediction | C-Index | 0.780 |
| MM | Multimodal Survival | C-Index | 0.610 |

### Training Data

- **M1**: BraTS 2021 (1,242 patients)
- **MG**: CGGA (~1,000 patients)
- **MM**: TCGA (72 patients with all modalities)

### Installation

```bash
# Clone repository
git clone https://github.com/melmbrain/brain-tumor-cdss.git
cd brain-tumor-cdss

# Install dependencies
pip install -r requirements.txt

# Download weights
mkdir -p weights
curl -L -o weights/m1_best.pth https://github.com/melmbrain/brain-tumor-cdss/releases/download/v1.0.0/m1_best.pth
curl -L -o weights/mg_best.pt https://github.com/melmbrain/brain-tumor-cdss/releases/download/v1.0.0/mg_4tasks_best.pt
curl -L -o weights/mm_best.pt https://github.com/melmbrain/brain-tumor-cdss/releases/download/v1.0.0/mm_best.pt
```

### What's Included

- Full model weights for M1, MG, and MM models
- Inference pipeline for single and batch predictions
- Demo Jupyter notebook
- MkDocs documentation
- Docker support
- PyPI package support

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU inference)

### License

MIT License

# Pretrained Weights

## Download

Pretrained weights are available from GitHub Releases:

```bash
# Download from releases
wget https://github.com/yourusername/brain-tumor-cdss/releases/download/v1.0/weights.zip
unzip weights.zip -d weights/
```

Or download manually from the [Releases](https://github.com/yourusername/brain-tumor-cdss/releases) page.

## Files

| File | Model | Size | Description |
|------|-------|------|-------------|
| `m1_seg.pth` | M1-Seg | ~250MB | MRI Segmentation (SwinUNETR) |
| `m1_cls.pth` | M1-Cls | ~10MB | MRI Classification heads |
| `mg_vae.pth` | MG | ~5MB | Gene VAE Encoder |
| `mg_embeddings.pth` | MG | ~2MB | Gene2Vec embeddings |
| `mm_fusion.pth` | MM | ~15MB | Multimodal Fusion |

## Training Data

- **M1**: BraTS 2021 (1,251 patients)
- **MG**: CGGA (1,018 patients)
- **MM**: TCGA (72 patients with MRI + Gene + Protein)

## Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| M1-Seg | Segmentation | Dice | 0.766 |
| M1-Cls | IDH | AUC | 0.878 |
| MG | Survival | C-Index | 0.780 |
| MM | Multimodal Survival | C-Index | 0.610 |

## Usage

```python
from models.m1 import M1Inference
from models.mg import MGInference
from models.mm import MMInference

# Load models
m1 = M1Inference(model_path="weights/m1_seg.pth")
mg = MGInference(
    model_path="weights/mg_vae.pth",
    gene_embeddings_path="weights/mg_embeddings.pth"
)
mm = MMInference(
    model_path="weights/mm_fusion.pth",
    m1_weights_path="weights/m1_seg.pth",
    mg_weights_path="weights/mg_vae.pth"
)
```

## Note

If weights are not available, the models will run in **demo mode** with random initialization.
This is useful for testing the pipeline structure but will not produce meaningful predictions.

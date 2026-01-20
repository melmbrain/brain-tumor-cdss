"""
M1 Model: MRI Multi-Task Model
Based on MODEL_SPECIFICATION.md Section 3.1

Core diagnostic model for brain tumor analysis:
- Grade prediction (3-class: G2, G3, G4)
- IDH mutation status (binary)
- MGMT methylation status (binary)
- Survival prediction (regression in days)
- Segmentation (optional, 3 tumor regions)

Architecture: SwinUNETR Encoder + Multi-Task Heads
Learning Type: Partial Multi-Task Learning (different labels per dataset)
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==================== Model Architecture ====================

class SwinEncoder(nn.Module):
    """
    SwinUNETR-style 3D encoder for MRI feature extraction
    Simplified version for inference
    """

    def __init__(self, in_channels=4, embed_dim=48, depths=[2, 2, 2, 2]):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, kernel_size=4, stride=4),
            nn.LayerNorm([embed_dim, 1, 1, 1]),  # Placeholder
        )

        # Hierarchical feature extraction
        self.stages = nn.ModuleList()
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = nn.Sequential(
                *[SwinBlock(dim) for _ in range(depth)]
            )
            self.stages.append(stage)

            # Downsample between stages (except last)
            if i < len(depths) - 1:
                self.stages.append(
                    nn.Sequential(
                        nn.Conv3d(dim, dim * 2, kernel_size=2, stride=2),
                        nn.BatchNorm3d(dim * 2),
                        nn.GELU()
                    )
                )

        self.final_dim = dims[-1]

    def forward(self, x):
        # Initial patch embedding
        x = self.patch_embed[0](x)

        # Hierarchical encoding
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return x, features


class SwinBlock(nn.Module):
    """Simplified Swin Transformer Block"""

    def __init__(self, dim, num_heads=4):
        super().__init__()

        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv3d(dim * 4, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SegmentationDecoder(nn.Module):
    """U-Net style decoder for segmentation"""

    def __init__(self, encoder_dims, num_classes=4):
        super().__init__()

        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()

        dims = encoder_dims[::-1]  # Reverse for decoder

        for i in range(len(dims) - 1):
            self.ups.append(
                nn.ConvTranspose3d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.convs.append(
                nn.Sequential(
                    nn.Conv3d(dims[i + 1] * 2, dims[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm3d(dims[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )

        self.final = nn.Conv3d(dims[-1], num_classes, kernel_size=1)

    def forward(self, x, skip_features):
        for i, (up, conv) in enumerate(zip(self.ups, self.convs)):
            x = up(x)
            skip = skip_features[-(i + 2)]  # Get matching skip connection
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return self.final(x)


class AttentionPooling(nn.Module):
    """Attention-based pooling for survival prediction"""

    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, D*H*W, C]

        attn_weights = F.softmax(self.attention(x_flat), dim=1)  # [B, D*H*W, 1]
        pooled = torch.sum(x_flat * attn_weights, dim=1)  # [B, C]

        return pooled


class MRIMultiTaskModel(nn.Module):
    """
    M1: MRI Multi-Task Model

    Primary diagnostic model for brain tumors
    Outputs:
    - Grade: 3-class (G2, G3, G4)
    - IDH: binary
    - MGMT: binary
    - Survival: regression (days)
    - Segmentation: 4-class (optional)
    """

    GRADE_CLASSES = ['G2', 'G3', 'G4']
    IDH_CLASSES = ['Wildtype', 'Mutant']
    MGMT_CLASSES = ['Unmethylated', 'Methylated']

    def __init__(
        self,
        in_channels=4,
        embed_dim=48,
        include_segmentation=True
    ):
        super().__init__()

        self.include_segmentation = include_segmentation

        # Encoder
        self.encoder = SwinEncoder(in_channels, embed_dim)
        encoder_dim = self.encoder.final_dim

        # Global pooling for classification
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Attention pooling for survival
        self.attention_pool = AttentionPooling(encoder_dim, 256)

        # Classification heads
        self.grade_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # G2, G3, G4
        )

        self.idh_head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Binary
        )

        self.mgmt_head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Binary
        )

        # Survival head (from attention-pooled features)
        self.survival_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # [mean, log_var] for uncertainty
        )

        # Segmentation decoder (optional)
        if include_segmentation:
            encoder_dims = [embed_dim * (2 ** i) for i in range(4)]
            self.seg_decoder = SegmentationDecoder(encoder_dims, num_classes=4)

    def forward(self, x, return_features=False):
        # Encode
        features, skip_features = self.encoder(x)

        # Global pooling for classification
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1).squeeze(-1)

        # Attention pooling for survival
        attn_pooled = self.attention_pool(features)

        # Classification outputs
        grade_logits = self.grade_head(pooled)
        idh_logit = self.idh_head(pooled)
        mgmt_logit = self.mgmt_head(pooled)

        # Survival output
        survival_out = self.survival_head(attn_pooled)
        survival_mean = survival_out[:, 0]
        survival_log_var = survival_out[:, 1]

        outputs = {
            'grade_logits': grade_logits,
            'idh_logit': idh_logit,
            'mgmt_logit': mgmt_logit,
            'survival_mean': survival_mean,
            'survival_log_var': survival_log_var
        }

        # Segmentation output (optional)
        if self.include_segmentation:
            seg_output = self.seg_decoder(features, skip_features)
            outputs['seg_logits'] = seg_output

        if return_features:
            outputs['features'] = pooled

        return outputs


# ==================== Inference Service ====================

class M1Inference:
    """
    M1 Model Inference Service
    MRI Multi-Task Analysis with Clinical Decision Support
    """

    GRADE_CLASSES = ['G2', 'G3', 'G4']
    IDH_CLASSES = ['Wildtype', 'Mutant']
    MGMT_CLASSES = ['Unmethylated', 'Methylated']

    def __init__(self, model_path: str = None, device: str = None):
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Load model
        self.model = MRIMultiTaskModel(
            in_channels=4,
            embed_dim=48,
            include_segmentation=True
        )

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded M1 model from {model_path}")
        else:
            logger.warning("M1 model weights not found, using random initialization (demo mode)")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Results directory
        self.results_dir = Path(os.environ.get('RESULTS_PATH', './results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        mri_path: str,
        patient_id: str,
        patient_info: Optional[Dict] = None,
        include_segmentation: bool = True
    ) -> Dict:
        """
        Run M1 (MRI Multi-Task) analysis

        Args:
            mri_path: Path to MRI data
            patient_id: Patient identifier
            patient_info: Optional patient metadata
            include_segmentation: Include segmentation output

        Returns:
            dict: Multi-task predictions with clinical recommendations
        """
        start_time = datetime.now()

        # Create result directory
        result_id = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_dir = self.results_dir / result_id
        result_dir.mkdir(parents=True, exist_ok=True)

        # Load and preprocess MRI
        mri_tensor = self._load_mri(mri_path)
        mri_tensor = mri_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(mri_tensor, return_features=True)

        # Process classification outputs
        grade_probs = F.softmax(outputs['grade_logits'], dim=-1).squeeze().cpu().numpy()
        idh_prob = torch.sigmoid(outputs['idh_logit']).squeeze().cpu().item()
        mgmt_prob = torch.sigmoid(outputs['mgmt_logit']).squeeze().cpu().item()

        # Get predictions
        grade_idx = int(np.argmax(grade_probs))
        idh_prediction = 'Mutant' if idh_prob > 0.5 else 'Wildtype'
        mgmt_prediction = 'Methylated' if mgmt_prob > 0.5 else 'Unmethylated'

        # Process survival prediction
        survival_mean = float(outputs['survival_mean'].cpu().item())
        survival_std = float(np.exp(0.5 * outputs['survival_log_var'].cpu().item()))

        # Convert to months (model outputs normalized days)
        survival_days = max(0, survival_mean * 730 + 365)  # Scale to reasonable range
        survival_months = survival_days / 30.0
        ci_low = max(0, (survival_days - 1.96 * survival_std * 180) / 30.0)
        ci_high = (survival_days + 1.96 * survival_std * 180) / 30.0

        # Process segmentation if available
        seg_result = None
        if include_segmentation and 'seg_logits' in outputs:
            seg_mask = torch.argmax(outputs['seg_logits'], dim=1).squeeze().cpu().numpy()
            seg_result = self._process_segmentation(seg_mask, result_dir)

        # Generate WHO classification
        who_class = self._generate_who_classification(
            self.GRADE_CLASSES[grade_idx],
            idh_prediction
        )

        # Generate clinical recommendations
        clinical_support = self._generate_clinical_support(
            grade=self.GRADE_CLASSES[grade_idx],
            idh=idh_prediction,
            mgmt=mgmt_prediction,
            survival_months=survival_months,
            patient_info=patient_info
        )

        # Build result
        result = {
            'grade': {
                'prediction': self.GRADE_CLASSES[grade_idx],
                'confidence': round(float(grade_probs[grade_idx]), 3),
                'probabilities': {
                    g: round(float(grade_probs[i]), 3)
                    for i, g in enumerate(self.GRADE_CLASSES)
                }
            },
            'idh': {
                'prediction': idh_prediction,
                'confidence': round(float(idh_prob if idh_prediction == 'Mutant' else 1 - idh_prob), 3)
            },
            'mgmt': {
                'prediction': mgmt_prediction,
                'confidence': round(float(mgmt_prob if mgmt_prediction == 'Methylated' else 1 - mgmt_prob), 3)
            },
            'survival': {
                'predicted_months': round(survival_months, 1),
                'confidence_interval': {
                    'low': round(ci_low, 1),
                    'high': round(ci_high, 1)
                },
                'risk_category': self._get_risk_category(survival_months, self.GRADE_CLASSES[grade_idx])
            },
            'who_classification': who_class,
            'clinical_recommendation': clinical_support
        }

        # Add segmentation if available
        if seg_result:
            result['segmentation'] = seg_result

        # Add processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result['processing_time_ms'] = int(processing_time)

        logger.info(f"M1 analysis completed for {patient_id}: {who_class}")

        return result

    def _load_mri(self, mri_path: str) -> torch.Tensor:
        """Load and preprocess MRI data"""
        import nibabel as nib

        mri_path = Path(mri_path)
        modalities = ['t1', 't1ce', 't2', 'flair']
        volumes = []

        if mri_path.is_dir():
            for mod in modalities:
                found = False
                # Try different naming conventions
                patterns = [
                    f'*_{mod}.nii.gz', f'*_{mod.upper()}.nii.gz',
                    f'*_{mod}.nii', f'{mod}.nii.gz',
                    f'*_T1c.nii.gz' if mod == 't1ce' else None
                ]
                patterns = [p for p in patterns if p]

                for pattern in patterns:
                    files = list(mri_path.glob(pattern))
                    if files:
                        nii = nib.load(str(files[0]))
                        volumes.append(nii.get_fdata())
                        found = True
                        break

                if not found:
                    logger.warning(f"Could not find {mod} modality, using zeros")
                    volumes.append(np.zeros((240, 240, 155)))
        else:
            nii = nib.load(str(mri_path))
            data = nii.get_fdata()
            if data.ndim == 4:
                for i in range(min(4, data.shape[-1])):
                    volumes.append(data[..., i])

        # Stack and normalize
        mri_data = np.stack(volumes, axis=0)

        for i in range(4):
            vol = mri_data[i]
            if np.any(vol > 0):
                p1, p99 = np.percentile(vol[vol > 0], [1, 99])
                vol = np.clip(vol, p1, p99)
                vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
            mri_data[i] = vol

        return torch.from_numpy(mri_data).float().unsqueeze(0)

    def _process_segmentation(self, seg_mask: np.ndarray, result_dir: Path) -> Dict:
        """Process segmentation output"""
        try:
            import nibabel as nib

            # Calculate volumes
            voxel_volume_mm3 = 1.0  # 1mm isotropic
            voxel_volume_cm3 = voxel_volume_mm3 / 1000

            volumes = {
                'necrosis_cm3': round(float(np.sum(seg_mask == 1)) * voxel_volume_cm3, 2),
                'edema_cm3': round(float(np.sum(seg_mask == 2)) * voxel_volume_cm3, 2),
                'enhancing_cm3': round(float(np.sum(seg_mask == 4)) * voxel_volume_cm3, 2),
            }
            volumes['whole_tumor_cm3'] = round(sum(volumes.values()), 2)
            volumes['tumor_core_cm3'] = round(
                volumes['necrosis_cm3'] + volumes['enhancing_cm3'], 2
            )

            # Save mask
            mask_path = result_dir / 'segmentation.nii.gz'
            nii = nib.Nifti1Image(seg_mask.astype(np.int16), np.eye(4))
            nib.save(nii, str(mask_path))

            return {
                'mask_path': str(mask_path),
                'volumes': volumes
            }
        except Exception as e:
            logger.warning(f"Failed to process segmentation: {e}")
            return None

    def _generate_who_classification(self, grade: str, idh: str) -> str:
        """Generate WHO 2021 CNS tumor classification"""
        if grade == 'G4':
            if idh == 'Wildtype':
                return 'Glioblastoma, IDH-wildtype'
            else:
                return 'Astrocytoma, IDH-mutant, Grade 4'
        elif grade == 'G3':
            if idh == 'Mutant':
                return 'Astrocytoma, IDH-mutant, Grade 3'
            else:
                return 'High-grade glioma, IDH-wildtype, Grade 3'
        else:  # G2
            if idh == 'Mutant':
                return 'Astrocytoma, IDH-mutant, Grade 2'
            else:
                return 'Low-grade glioma, IDH-wildtype, Grade 2'

    def _get_risk_category(self, survival_months: float, grade: str) -> str:
        """Determine risk category"""
        if grade == 'G4':
            if survival_months < 12:
                return 'Very High'
            elif survival_months < 18:
                return 'High'
            else:
                return 'Moderate'
        elif grade == 'G3':
            if survival_months < 24:
                return 'High'
            elif survival_months < 48:
                return 'Moderate'
            else:
                return 'Low'
        else:  # G2
            if survival_months < 60:
                return 'Moderate'
            else:
                return 'Low'

    def _generate_clinical_support(
        self,
        grade: str,
        idh: str,
        mgmt: str,
        survival_months: float,
        patient_info: Optional[Dict]
    ) -> Dict:
        """Generate clinical recommendations"""
        recommendations = {
            'primary': '',
            'treatment_options': [],
            'monitoring': '',
            'prognosis_summary': ''
        }

        # Primary recommendation based on grade and molecular markers
        if grade == 'G4' and idh == 'Wildtype':
            if mgmt == 'Methylated':
                recommendations['primary'] = (
                    'Standard Stupp protocol recommended: '
                    'Maximal safe resection + RT (60Gy) + concurrent/adjuvant Temozolomide'
                )
                recommendations['treatment_options'] = [
                    'Stupp protocol (RT + TMZ)',
                    'TTFields (Optune) consideration',
                    'Clinical trial enrollment'
                ]
            else:
                recommendations['primary'] = (
                    'Consider alternative to standard TMZ due to unmethylated MGMT. '
                    'Maximal safe resection + RT recommended'
                )
                recommendations['treatment_options'] = [
                    'RT alone or RT + alternative agent',
                    'Clinical trial for MGMT-unmethylated GBM',
                    'Immunotherapy consideration'
                ]
            recommendations['monitoring'] = 'MRI every 2-3 months'

        elif idh == 'Mutant':
            if grade == 'G4':
                recommendations['primary'] = (
                    'IDH-mutant Grade 4 astrocytoma: '
                    'Maximal safe resection + RT +/- TMZ'
                )
                recommendations['treatment_options'] = [
                    'RT + TMZ (modified Stupp)',
                    'IDH inhibitor clinical trials',
                    'Vorasidenib consideration'
                ]
            else:
                recommendations['primary'] = (
                    'IDH-mutant lower grade glioma: '
                    'Maximal safe resection, consider watchful waiting for Grade 2'
                )
                recommendations['treatment_options'] = [
                    'Observation with serial imaging (Grade 2)',
                    'RT + PCV for higher risk features',
                    'IDH inhibitor trials'
                ]
            recommendations['monitoring'] = 'MRI every 3-6 months'

        else:  # Lower grade, IDH-wildtype
            recommendations['primary'] = (
                'IDH-wildtype lower grade glioma: '
                'Consider more aggressive approach due to molecular profile'
            )
            recommendations['treatment_options'] = [
                'Maximal safe resection',
                'Adjuvant RT consideration',
                'Molecular re-testing recommended'
            ]
            recommendations['monitoring'] = 'MRI every 3-4 months'

        # Prognosis summary
        risk = self._get_risk_category(survival_months, grade)
        recommendations['prognosis_summary'] = (
            f"Estimated median survival: {survival_months:.1f} months. "
            f"Risk category: {risk}. "
        )
        if idh == 'Mutant':
            recommendations['prognosis_summary'] += "IDH mutation is a favorable prognostic factor. "
        if mgmt == 'Methylated' and grade in ['G3', 'G4']:
            recommendations['prognosis_summary'] += "MGMT methylation suggests benefit from alkylating agents. "

        return recommendations

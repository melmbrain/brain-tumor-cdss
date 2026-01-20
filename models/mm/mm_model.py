"""
MM Model: Multimodal Fusion for Comprehensive Glioma Analysis
Based on MODEL_SPECIFICATION.md Section 3.3

Features:
- Multi-modal data integration (MRI + Genomics + Proteomics + Clinical)
- Uses pretrained M1 (MRI) and MG (Gene) encoders for feature extraction
- WHO 2021 Classification prediction (7 tasks)
- Survival prediction with uncertainty
- Explainability (Attention maps, Gene/Protein importance)
- Clinical Decision Support

Updated Dimensions (based on actual pretrained encoders):
- M1 (MRI): 768-dim (SwinUNETR feature_size=48 → 48*16=768)
- MG (Gene): 64-dim (VAE latent_dim=64)
- Protein: 167-dim (RPPA after NA>20% drop)
- Clinical: 10-dim
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add scripts directory for feature extractor imports
SCRIPT_DIR = Path(__file__).parent.parent.parent / 'scripts'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

logger = logging.getLogger(__name__)


# ==================== Model Architecture ====================

class MRIEncoder(nn.Module):
    """
    3D CNN Encoder for MRI feature extraction
    Produces spatial attention maps for explainability
    """

    def __init__(self, in_channels=4, feature_dim=512):
        super().__init__()

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, 32, 3, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(256, 512, 3, stride=2, padding=1),
                nn.BatchNorm3d(512),
                nn.ReLU(inplace=True)
            )
        ])

        # Spatial attention for explainability
        self.attention = nn.Sequential(
            nn.Conv3d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 1, 1),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x, return_attention=False):
        for block in self.conv_blocks:
            x = block(x)

        # Generate attention map
        attention_map = self.attention(x)
        x_attended = x * attention_map

        # Global pooling
        x_pooled = self.global_pool(x_attended)
        x_flat = x_pooled.view(x_pooled.size(0), -1)

        features = self.fc(x_flat)

        if return_attention:
            return features, attention_map
        return features


class GenomicEncoder(nn.Module):
    """
    MLP Encoder for genomic (gene expression) data
    With attention for gene importance
    """

    def __init__(self, input_dim=1000, feature_dim=256, hidden_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Gene attention for importance
        self.gene_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        attention = self.gene_attention(x)
        x_attended = x * attention

        features = self.encoder(x_attended)

        if return_attention:
            return features, attention
        return features


class ProteomicEncoder(nn.Module):
    """
    MLP Encoder for proteomic (protein expression) data
    With attention for protein importance
    """

    def __init__(self, input_dim=200, feature_dim=128, hidden_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Protein attention for importance
        self.protein_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        attention = self.protein_attention(x)
        x_attended = x * attention

        features = self.encoder(x_attended)

        if return_attention:
            return features, attention
        return features


class ClinicalEncoder(nn.Module):
    """
    MLP Encoder for clinical features
    """

    def __init__(self, input_dim=10, feature_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for multi-modal fusion
    """

    def __init__(self, feature_dims: List[int], fusion_dim=256):
        super().__init__()

        total_dim = sum(feature_dims)

        # Project all modalities to same dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in feature_dims
        ])

        # Cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(fusion_dim * len(feature_dims), fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, features: List[torch.Tensor]):
        # Project all features to same dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]

        # Stack as sequence
        stacked = torch.stack(projected, dim=1)  # [B, num_modalities, fusion_dim]

        # Self-attention across modalities
        attended, attn_weights = self.attention(stacked, stacked, stacked)

        # Flatten and project
        fused = attended.reshape(attended.size(0), -1)
        output = self.output(fused)

        return output, attn_weights


class MultimodalFusionModelLegacy(nn.Module):
    """
    Legacy MM Model: Multimodal Fusion for Glioma Analysis
    (Uses raw data with own encoders - kept for backward compatibility)
    """

    def __init__(
        self,
        mri_channels=4,
        genomic_dim=1000,
        proteomic_dim=200,
        clinical_dim=10,
        fusion_dim=256
    ):
        super().__init__()

        # Modality-specific encoders
        self.mri_encoder = MRIEncoder(mri_channels, 512)
        self.genomic_encoder = GenomicEncoder(genomic_dim, 256)
        self.proteomic_encoder = ProteomicEncoder(proteomic_dim, 128)
        self.clinical_encoder = ClinicalEncoder(clinical_dim, 64)

        # Cross-modal fusion
        self.fusion = CrossModalAttention(
            feature_dims=[512, 256, 128, 64],
            fusion_dim=fusion_dim
        )

        # Classification heads
        self.grade_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        self.idh_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.mgmt_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.codel_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.cdkn2ab_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.atrx_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.survival_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

    def forward(self, mri, genomics, proteomics, clinical, return_attention=False):
        mri_feat, mri_attn = self.mri_encoder(mri, return_attention=True)
        genomic_feat, gene_attn = self.genomic_encoder(genomics, return_attention=True)
        proteomic_feat, protein_attn = self.proteomic_encoder(proteomics, return_attention=True)
        clinical_feat = self.clinical_encoder(clinical)

        fused, modal_attn = self.fusion([mri_feat, genomic_feat, proteomic_feat, clinical_feat])

        survival_out = self.survival_head(fused)

        outputs = {
            'grade_logits': self.grade_head(fused),
            'idh_logits': self.idh_head(fused),
            'mgmt_logits': self.mgmt_head(fused),
            'codel_logits': self.codel_head(fused),
            'cdkn2ab_logits': self.cdkn2ab_head(fused),
            'atrx_logits': self.atrx_head(fused),
            'survival_mean': survival_out[:, 0],
            'survival_log_var': survival_out[:, 1],
            'risk_score': survival_out[:, 2]
        }

        if return_attention:
            outputs['mri_attention'] = mri_attn
            outputs['gene_attention'] = gene_attn
            outputs['protein_attention'] = protein_attn
            outputs['modal_attention'] = modal_attn

        return outputs


class MultimodalFusionModel(nn.Module):
    """
    MM Model v2: Multimodal Fusion with Pretrained M1/MG Encoders

    Uses pre-extracted features from:
    - M1 encoder: 768-dim (SwinUNETR feature_size=48)
    - MG encoder: 64-dim (VAE latent)

    Inputs:
    - mri_feat: Pre-extracted MRI features [B, 768]
    - gene_feat: Pre-extracted gene features [B, 64]
    - protein: RPPA protein expression [B, 167]
    - clinical: Clinical features [B, 10]

    Outputs:
    - 7 classification tasks (Grade, IDH, MGMT, 1p19q, CDKN2A/B, ATRX)
    - Survival prediction with uncertainty
    """

    def __init__(
        self,
        mri_dim=768,      # M1 encoder output (SwinUNETR)
        gene_dim=64,       # MG encoder output (VAE latent)
        protein_dim=167,   # RPPA after NA handling
        clinical_dim=10,
        fusion_dim=256,
        dropout=0.5
    ):
        super().__init__()

        self.mri_dim = mri_dim
        self.gene_dim = gene_dim

        # MRI Projection: 768 → 512 → 256 (2-layer projection)
        self.mri_proj = nn.Sequential(
            nn.Linear(mri_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )

        # Gene Projection: 64 → 128 → 256 (2-layer projection)
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )

        # Protein Encoder: 167 → 128 → 64 → 256
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        self.protein_proj = nn.Linear(64, fusion_dim)

        # Clinical Encoder: 10 → 32 → 256
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, fusion_dim)
        )

        # Cross-Attention Fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Classification Heads (7 tasks)
        self.grade_head = self._make_head(fusion_dim, 4)
        self.idh_head = self._make_head(fusion_dim, 2)
        self.mgmt_head = self._make_head(fusion_dim, 2)
        self.codel_head = self._make_head(fusion_dim, 2)
        self.cdkn2ab_head = self._make_head(fusion_dim, 2)
        self.atrx_head = self._make_head(fusion_dim, 2)

        # Survival head
        self.survival_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # [mean, log_var, risk_score]
        )

    def _make_head(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )

    def forward(
        self,
        mri_feat: torch.Tensor,
        gene_feat: torch.Tensor,
        protein: torch.Tensor,
        clinical: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass with pre-extracted features

        Args:
            mri_feat: Pre-extracted MRI features from M1 encoder [B, 768]
            gene_feat: Pre-extracted gene features from MG encoder [B, 64]
            protein: RPPA protein expression [B, 167]
            clinical: Clinical features [B, 10]
            return_attention: Whether to return attention weights

        Returns:
            Dict with predictions for all tasks
        """
        # Project all modalities to fusion_dim=256
        mri_proj = self.mri_proj(mri_feat)
        gene_proj = self.gene_proj(gene_feat)
        prot_feat = self.protein_encoder(protein)
        prot_proj = self.protein_proj(prot_feat)
        clin_proj = self.clinical_encoder(clinical)

        # Stack for cross-attention [B, 4, 256]
        stacked = torch.stack([mri_proj, gene_proj, prot_proj, clin_proj], dim=1)

        # Cross-attention fusion
        attended, attn_weights = self.cross_attention(stacked, stacked, stacked)

        # Flatten and fuse
        fused = attended.reshape(attended.size(0), -1)
        fused = self.fusion_mlp(fused)

        # Survival prediction
        survival_out = self.survival_head(fused)

        outputs = {
            'grade_logits': self.grade_head(fused),
            'idh_logits': self.idh_head(fused),
            'mgmt_logits': self.mgmt_head(fused),
            'codel_logits': self.codel_head(fused),
            'cdkn2ab_logits': self.cdkn2ab_head(fused),
            'atrx_logits': self.atrx_head(fused),
            'survival_mean': survival_out[:, 0],
            'survival_log_var': survival_out[:, 1],
            'risk_score': survival_out[:, 2]
        }

        if return_attention:
            outputs['modal_attention'] = attn_weights

        return outputs


# ==================== Inference Service ====================

class MMInference:
    """
    MM Model Inference Service
    Multimodal Fusion with Clinical Decision Support
    """

    # Grade classes
    GRADE_CLASSES = ['G1', 'G2', 'G3', 'G4']

    # IDH classes
    IDH_CLASSES = ['Wildtype', 'Mutant']

    # MGMT classes
    MGMT_CLASSES = ['Unmethylated', 'Methylated']

    # 1p19q classes
    CODEL_CLASSES = ['Non-codel', 'Codel']

    # CDKN2A/B classes (per MODEL_SPECIFICATION.md Section 3.3)
    CDKN2AB_CLASSES = ['Intact', 'Deleted']

    # ATRX classes (per MODEL_SPECIFICATION.md Section 3.3)
    ATRX_CLASSES = ['Wildtype', 'Mutant']

    # Top genes for importance tracking
    TOP_GENES = [
        'EGFR', 'TP53', 'PTEN', 'IDH1', 'IDH2',
        'ATRX', 'TERT', 'CDKN2A', 'NF1', 'PIK3CA',
        'RB1', 'PDGFRA', 'MDM2', 'CDK4', 'MET'
    ]

    # Top proteins for importance tracking
    TOP_PROTEINS = [
        'p-AKT', 'p-ERK', 'EGFR', 'p-S6', 'p53',
        'Ki67', 'MGMT', 'IDH1-R132H', 'ATRX', 'p-mTOR'
    ]

    def __init__(
        self,
        model_path: str = None,
        m1_weights_path: str = None,
        mg_weights_path: str = None,
        device: str = None,
        use_pretrained_encoders: bool = True
    ):
        """
        Initialize MM Inference Service

        Args:
            model_path: Path to MM model weights
            m1_weights_path: Path to M1 (MRI) encoder weights
            mg_weights_path: Path to MG (Gene) encoder weights
            device: Device to use (cuda/cpu)
            use_pretrained_encoders: If True, use new model with pretrained encoder features
                                    If False, use legacy model with raw data
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.use_pretrained_encoders = use_pretrained_encoders

        # Load MM model (v2 with pretrained encoder features or legacy)
        if use_pretrained_encoders:
            # New model expects pre-extracted features
            self.model = MultimodalFusionModel(
                mri_dim=768,       # M1 encoder output (SwinUNETR)
                gene_dim=64,       # MG encoder output (VAE latent)
                protein_dim=167,   # RPPA proteins
                clinical_dim=10,
                fusion_dim=256
            )
            logger.info("Using MultimodalFusionModel v2 (pretrained encoder features)")
        else:
            # Legacy model with own encoders
            self.model = MultimodalFusionModelLegacy(
                mri_channels=4,
                genomic_dim=1000,
                proteomic_dim=200,
                clinical_dim=10,
                fusion_dim=256
            )
            logger.info("Using MultimodalFusionModel Legacy (raw data)")

        # Load MM weights if available
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded MM model from {model_path}")
        else:
            logger.warning("MM model weights not found, using random initialization (demo mode)")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Load M1/MG encoders for feature extraction (if using pretrained mode)
        self.m1_encoder = None
        self.mg_encoder = None

        if use_pretrained_encoders:
            try:
                from extract_features import load_m1_encoder, load_mg_encoder

                if m1_weights_path and Path(m1_weights_path).exists():
                    self.m1_encoder = load_m1_encoder(m1_weights_path, str(self.device))
                    logger.info(f"Loaded M1 encoder from {m1_weights_path}")
                else:
                    logger.warning("M1 encoder weights not found - MRI features will be random")

                if mg_weights_path and Path(mg_weights_path).exists():
                    self.mg_encoder = load_mg_encoder(mg_weights_path, str(self.device))
                    logger.info(f"Loaded MG encoder from {mg_weights_path}")
                else:
                    logger.warning("MG encoder weights not found - Gene features will be random")

            except ImportError as e:
                logger.warning(f"Could not import feature extractors: {e}")
                logger.warning("Feature extraction will use random features (demo mode)")

        # Results directory
        self.results_dir = Path(os.environ.get('RESULTS_PATH', './results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        mri_path: str,
        patient_id: str,
        genomic_data: Optional[Dict] = None,
        proteomic_data: Optional[Dict] = None,
        clinical_data: Optional[Dict] = None
    ) -> Dict:
        """
        Run MM (Multimodal Fusion) analysis

        Args:
            mri_path: Path to MRI data
            patient_id: Patient identifier
            genomic_data: Gene expression data (optional)
            proteomic_data: Protein expression data (optional)
            clinical_data: Clinical features (optional)

        Returns:
            dict: Comprehensive analysis with predictions and explanations
        """
        start_time = datetime.now()

        # Create result directory
        result_id = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_dir = self.results_dir / result_id
        result_dir.mkdir(parents=True, exist_ok=True)

        # Load and preprocess data based on mode
        if self.use_pretrained_encoders:
            # Extract features using M1/MG encoders
            mri_feat = self._extract_mri_features(mri_path)
            gene_feat = self._extract_gene_features(genomic_data)
            protein_tensor = self._process_proteomic_v2(proteomic_data)
            clinical_tensor = self._process_clinical(clinical_data)

            # Move to device
            mri_feat = mri_feat.to(self.device)
            gene_feat = gene_feat.to(self.device)
            protein_tensor = protein_tensor.to(self.device)
            clinical_tensor = clinical_tensor.to(self.device)

            # Run inference with pre-extracted features
            with torch.no_grad():
                outputs = self.model(
                    mri_feat,
                    gene_feat,
                    protein_tensor,
                    clinical_tensor,
                    return_attention=True
                )

            # In v2 mode, we don't have gene/protein/MRI attention from the model
            # The attention is modal-level only
            gene_importance = self._generate_dummy_gene_importance(genomic_data)
            protein_importance = self._generate_dummy_protein_importance(proteomic_data)
            mri_attention_path = None  # No MRI attention in v2 mode

        else:
            # Legacy mode: process raw data
            mri_tensor = self._load_mri(mri_path)
            genomic_tensor = self._process_genomic(genomic_data)
            proteomic_tensor = self._process_proteomic(proteomic_data)
            clinical_tensor = self._process_clinical(clinical_data)

            # Move to device
            mri_tensor = mri_tensor.to(self.device)
            genomic_tensor = genomic_tensor.to(self.device)
            proteomic_tensor = proteomic_tensor.to(self.device)
            clinical_tensor = clinical_tensor.to(self.device)

            # Run inference with raw data
            with torch.no_grad():
                outputs = self.model(
                    mri_tensor,
                    genomic_tensor,
                    proteomic_tensor,
                    clinical_tensor,
                    return_attention=True
                )

            # Process attention maps for explainability
            gene_importance = self._extract_gene_importance(
                outputs['gene_attention'].squeeze().cpu().numpy(),
                genomic_data
            )
            protein_importance = self._extract_protein_importance(
                outputs['protein_attention'].squeeze().cpu().numpy(),
                proteomic_data
            )

            # Save MRI attention map
            mri_attention_path = self._save_mri_attention(
                outputs['mri_attention'].squeeze().cpu().numpy(),
                result_dir
            )

        # Process classification outputs (7 tasks per MODEL_SPECIFICATION.md Section 3.3)
        grade_probs = F.softmax(outputs['grade_logits'], dim=-1).squeeze().cpu().numpy()
        idh_probs = F.softmax(outputs['idh_logits'], dim=-1).squeeze().cpu().numpy()
        mgmt_probs = F.softmax(outputs['mgmt_logits'], dim=-1).squeeze().cpu().numpy()
        codel_probs = F.softmax(outputs['codel_logits'], dim=-1).squeeze().cpu().numpy()
        cdkn2ab_probs = F.softmax(outputs['cdkn2ab_logits'], dim=-1).squeeze().cpu().numpy()
        atrx_probs = F.softmax(outputs['atrx_logits'], dim=-1).squeeze().cpu().numpy()

        # Get predictions
        grade_idx = int(np.argmax(grade_probs))
        idh_idx = int(np.argmax(idh_probs))
        mgmt_idx = int(np.argmax(mgmt_probs))
        codel_idx = int(np.argmax(codel_probs))
        cdkn2ab_idx = int(np.argmax(cdkn2ab_probs))
        atrx_idx = int(np.argmax(atrx_probs))

        # Process survival prediction
        survival_mean = float(outputs['survival_mean'].cpu().item())
        survival_std = float(np.exp(0.5 * outputs['survival_log_var'].cpu().item()))
        risk_score = torch.sigmoid(outputs['risk_score']).cpu().item()

        # Convert to months (model outputs normalized value)
        predicted_months = max(0, survival_mean * 36 + 12)  # Scale to 0-48 months range
        ci_low = max(0, predicted_months - 1.96 * survival_std * 12)
        ci_high = predicted_months + 1.96 * survival_std * 12

        # Generate WHO classification
        who_class = self._generate_who_classification(
            self.GRADE_CLASSES[grade_idx],
            self.IDH_CLASSES[idh_idx],
            self.CODEL_CLASSES[codel_idx]
        )

        # Generate clinical decision support (including all 7 markers)
        clinical_support = self._generate_clinical_support(
            grade=self.GRADE_CLASSES[grade_idx],
            idh=self.IDH_CLASSES[idh_idx],
            mgmt=self.MGMT_CLASSES[mgmt_idx],
            codel=self.CODEL_CLASSES[codel_idx],
            cdkn2ab=self.CDKN2AB_CLASSES[cdkn2ab_idx],
            atrx=self.ATRX_CLASSES[atrx_idx],
            survival_months=predicted_months,
            risk_score=risk_score
        )

        # Build result
        result = {
            'classification': {
                'grade': {
                    'prediction': self.GRADE_CLASSES[grade_idx],
                    'confidence': round(float(grade_probs[grade_idx]), 3),
                    'probabilities': {
                        g: round(float(grade_probs[i]), 3)
                        for i, g in enumerate(self.GRADE_CLASSES)
                    }
                },
                'idh': {
                    'prediction': self.IDH_CLASSES[idh_idx],
                    'confidence': round(float(idh_probs[idh_idx]), 3)
                },
                'mgmt': {
                    'prediction': self.MGMT_CLASSES[mgmt_idx],
                    'confidence': round(float(mgmt_probs[mgmt_idx]), 3)
                },
                '1p19q': {
                    'prediction': self.CODEL_CLASSES[codel_idx],
                    'confidence': round(float(codel_probs[codel_idx]), 3)
                },
                'cdkn2ab': {
                    'prediction': self.CDKN2AB_CLASSES[cdkn2ab_idx],
                    'confidence': round(float(cdkn2ab_probs[cdkn2ab_idx]), 3)
                },
                'atrx': {
                    'prediction': self.ATRX_CLASSES[atrx_idx],
                    'confidence': round(float(atrx_probs[atrx_idx]), 3)
                }
            },
            'survival': {
                'predicted_months': round(predicted_months, 1),
                'risk_category': self._get_risk_category(risk_score),
                'confidence_interval': {
                    'low': round(ci_low, 1),
                    'high': round(ci_high, 1)
                },
                'risk_score': round(float(risk_score), 3)
            },
            'explainability': {
                'mri_attention_map': str(mri_attention_path) if mri_attention_path else None,
                'gene_importance': gene_importance,
                'protein_importance': protein_importance
            },
            'clinical_decision_support': {
                'who_classification': who_class,
                'treatment_recommendation': clinical_support['treatment'],
                'clinical_trial_eligibility': clinical_support['trials'],
                'prognosis_summary': clinical_support['prognosis'],
                'risk_factors': clinical_support.get('risk_factors', [])
            }
        }

        # Add processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result['processing_time_ms'] = int(processing_time)

        logger.info(f"MM analysis completed for {patient_id}: {who_class}")

        return result

    def _load_mri(self, mri_path: str) -> torch.Tensor:
        """Load and preprocess MRI data"""
        import nibabel as nib

        mri_path = Path(mri_path)
        modalities = ['t1', 't1ce', 't2', 'flair']
        volumes = []

        if mri_path.is_dir():
            for mod in modalities:
                for pattern in [f'*_{mod}.nii.gz', f'*_{mod}.nii', f'{mod}.nii.gz']:
                    files = list(mri_path.glob(pattern))
                    if files:
                        nii = nib.load(str(files[0]))
                        volumes.append(nii.get_fdata())
                        break
                else:
                    logger.warning(f"Could not find {mod} modality, using zeros")
                    volumes.append(np.zeros((128, 128, 128)))
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

    def _process_genomic(self, genomic_data: Optional[Dict]) -> torch.Tensor:
        """Process genomic data into tensor"""
        if genomic_data is None or 'expression' not in genomic_data:
            # Return random data for demo mode
            return torch.randn(1, 1000)

        expression = genomic_data['expression']
        if isinstance(expression, list):
            expression = np.array(expression)

        # Normalize
        expression = (expression - expression.mean()) / (expression.std() + 1e-8)

        # Pad or truncate to expected dimension
        if len(expression) < 1000:
            expression = np.pad(expression, (0, 1000 - len(expression)))
        else:
            expression = expression[:1000]

        return torch.from_numpy(expression).float().unsqueeze(0)

    def _process_proteomic(self, proteomic_data: Optional[Dict]) -> torch.Tensor:
        """Process proteomic data into tensor"""
        if proteomic_data is None or 'expression' not in proteomic_data:
            # Return random data for demo mode
            return torch.randn(1, 200)

        expression = proteomic_data['expression']
        if isinstance(expression, list):
            expression = np.array(expression)

        # Normalize
        expression = (expression - expression.mean()) / (expression.std() + 1e-8)

        # Pad or truncate to expected dimension
        if len(expression) < 200:
            expression = np.pad(expression, (0, 200 - len(expression)))
        else:
            expression = expression[:200]

        return torch.from_numpy(expression).float().unsqueeze(0)

    def _process_clinical(self, clinical_data: Optional[Dict]) -> torch.Tensor:
        """Process clinical data into tensor"""
        features = np.zeros(10)

        if clinical_data:
            # Age (normalized)
            features[0] = clinical_data.get('age', 50) / 100.0

            # Sex (one-hot)
            features[1] = 1 if clinical_data.get('sex', 'M') == 'M' else 0
            features[2] = 1 if clinical_data.get('sex', 'M') == 'F' else 0

            # KPS (normalized)
            features[3] = clinical_data.get('kps', 80) / 100.0

            # Extent of resection (one-hot)
            eor = clinical_data.get('extent_of_resection', 'unknown')
            features[4] = 1 if eor == 'GTR' else 0
            features[5] = 1 if eor == 'STR' else 0
            features[6] = 1 if eor == 'biopsy' else 0

            # Prior treatment
            features[7] = 1 if clinical_data.get('prior_radiation', False) else 0
            features[8] = 1 if clinical_data.get('prior_chemo', False) else 0

            # Tumor location (simplified)
            features[9] = 1 if clinical_data.get('eloquent_cortex', False) else 0

        return torch.from_numpy(features).float().unsqueeze(0)

    # ==================== V2 Mode Helper Methods ====================

    def _extract_mri_features(self, mri_path: str) -> torch.Tensor:
        """
        Extract MRI features using M1 encoder

        Args:
            mri_path: Path to MRI data

        Returns:
            MRI features [1, 768]
        """
        if self.m1_encoder is None:
            # Return random features if encoder not loaded (demo mode)
            logger.warning("M1 encoder not loaded, using random MRI features")
            return torch.randn(1, 768)

        try:
            import nibabel as nib
            from scipy.ndimage import zoom

            mri_path = Path(mri_path)
            modalities = ['t1', 't1ce', 't2', 'flair']
            volumes = []

            if mri_path.is_dir():
                for mod in modalities:
                    for pattern in [f'*_{mod}.nii.gz', f'*_{mod}.nii', f'{mod}.nii.gz']:
                        files = list(mri_path.glob(pattern))
                        if files:
                            nii = nib.load(str(files[0]))
                            vol = nii.get_fdata().astype(np.float32)
                            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
                            volumes.append(vol)
                            break
                    else:
                        logger.warning(f"Could not find {mod} modality, using zeros")
                        volumes.append(np.zeros((128, 128, 128), dtype=np.float32))
            else:
                nii = nib.load(str(mri_path))
                data = nii.get_fdata().astype(np.float32)
                if data.ndim == 4:
                    for i in range(min(4, data.shape[-1])):
                        vol = data[..., i]
                        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
                        volumes.append(vol)

            # Stack [4, D, H, W]
            mri_data = np.stack(volumes, axis=0)

            # Resize to 128^3
            target_shape = (128, 128, 128)
            if mri_data.shape[1:] != target_shape:
                zoom_factors = [t / s for t, s in zip(target_shape, mri_data.shape[1:])]
                mri_data = np.stack([
                    zoom(mri_data[c], zoom_factors, order=1)
                    for c in range(4)
                ], axis=0)

            # To tensor [1, 4, 128, 128, 128]
            x = torch.tensor(mri_data, device=self.device).unsqueeze(0)

            # Extract features using M1 encoder
            with torch.no_grad():
                features = self.m1_encoder(x)  # [1, 768]

            return features.cpu()

        except Exception as e:
            logger.error(f"Error extracting MRI features: {e}")
            return torch.randn(1, 768)

    def _extract_gene_features(self, genomic_data: Optional[Dict]) -> torch.Tensor:
        """
        Extract gene features using MG encoder

        Args:
            genomic_data: Gene expression data

        Returns:
            Gene features [1, 64]
        """
        if self.mg_encoder is None:
            # Return random features if encoder not loaded (demo mode)
            logger.warning("MG encoder not loaded, using random gene features")
            return torch.randn(1, 64)

        if genomic_data is None or 'expression' not in genomic_data:
            logger.warning("No gene expression data, using random features")
            return torch.randn(1, 64)

        try:
            expression = genomic_data['expression']
            if isinstance(expression, list):
                expression = np.array(expression)

            # Ensure correct number of genes (match MG encoder)
            n_genes = self.mg_encoder.n_genes if hasattr(self.mg_encoder, 'n_genes') else 42
            if len(expression) != n_genes:
                logger.warning(f"Gene count mismatch: got {len(expression)}, expected {n_genes}")
                if len(expression) < n_genes:
                    expression = np.pad(expression, (0, n_genes - len(expression)))
                else:
                    expression = expression[:n_genes]

            # Normalize (Z-score)
            expression = (expression - expression.mean()) / (expression.std() + 1e-8)

            # To tensor [1, n_genes]
            x = torch.tensor(expression, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Extract features using MG encoder
            with torch.no_grad():
                features = self.mg_encoder(x, return_mu_only=True)  # [1, 64]

            return features.cpu()

        except Exception as e:
            logger.error(f"Error extracting gene features: {e}")
            return torch.randn(1, 64)

    def _process_proteomic_v2(self, proteomic_data: Optional[Dict]) -> torch.Tensor:
        """
        Process proteomic data for v2 model (167-dim after NA handling)

        Args:
            proteomic_data: Protein expression data

        Returns:
            Protein features [1, 167]
        """
        if proteomic_data is None or 'expression' not in proteomic_data:
            # Return random data for demo mode
            return torch.randn(1, 167)

        expression = proteomic_data['expression']
        if isinstance(expression, list):
            expression = np.array(expression)

        # Normalize
        expression = (expression - expression.mean()) / (expression.std() + 1e-8)

        # Pad or truncate to 167 (RPPA after NA handling)
        if len(expression) < 167:
            expression = np.pad(expression, (0, 167 - len(expression)))
        else:
            expression = expression[:167]

        return torch.from_numpy(expression).float().unsqueeze(0)

    def _generate_dummy_gene_importance(self, genomic_data: Optional[Dict]) -> List[Dict]:
        """
        Generate placeholder gene importance for v2 mode
        (V2 model uses pretrained encoder, no gene-level attention)
        """
        gene_names = self.TOP_GENES
        if genomic_data and 'gene_names' in genomic_data:
            gene_names = genomic_data['gene_names'][:15]

        return [
            {
                'gene': gene_names[i] if i < len(gene_names) else f'Gene_{i}',
                'importance': round(1.0 / (i + 1), 4)  # Placeholder
            }
            for i in range(min(10, len(gene_names)))
        ]

    def _generate_dummy_protein_importance(self, proteomic_data: Optional[Dict]) -> List[Dict]:
        """
        Generate placeholder protein importance for v2 mode
        (V2 model uses pretrained encoder, no protein-level attention)
        """
        protein_names = self.TOP_PROTEINS
        if proteomic_data and 'protein_names' in proteomic_data:
            protein_names = proteomic_data['protein_names'][:15]

        return [
            {
                'protein': protein_names[i] if i < len(protein_names) else f'Protein_{i}',
                'importance': round(1.0 / (i + 1), 4)  # Placeholder
            }
            for i in range(min(10, len(protein_names)))
        ]

    # ==================== Legacy Mode Helper Methods ====================

    def _extract_gene_importance(
        self,
        attention: np.ndarray,
        genomic_data: Optional[Dict]
    ) -> List[Dict]:
        """Extract gene importance from attention"""
        if genomic_data and 'gene_names' in genomic_data:
            gene_names = genomic_data['gene_names'][:len(attention)]
        else:
            gene_names = self.TOP_GENES

        # Get top indices
        top_k = min(10, len(attention))
        top_indices = np.argsort(attention)[::-1][:top_k]

        return [
            {
                'gene': gene_names[i] if i < len(gene_names) else f'Gene_{i}',
                'importance': round(float(attention[i]), 4)
            }
            for i in top_indices
        ]

    def _extract_protein_importance(
        self,
        attention: np.ndarray,
        proteomic_data: Optional[Dict]
    ) -> List[Dict]:
        """Extract protein importance from attention"""
        if proteomic_data and 'protein_names' in proteomic_data:
            protein_names = proteomic_data['protein_names'][:len(attention)]
        else:
            protein_names = self.TOP_PROTEINS

        # Get top indices
        top_k = min(10, len(attention))
        top_indices = np.argsort(attention)[::-1][:top_k]

        return [
            {
                'protein': protein_names[i] if i < len(protein_names) else f'Protein_{i}',
                'importance': round(float(attention[i]), 4)
            }
            for i in top_indices
        ]

    def _save_mri_attention(
        self,
        attention: np.ndarray,
        result_dir: Path
    ) -> Optional[Path]:
        """Save MRI attention map as image"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Get middle slice
            if attention.ndim == 4:
                attention = attention[0]  # Remove channel dim
            mid_slice = attention.shape[2] // 2

            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(attention[:, :, mid_slice], cmap='hot')
            ax.set_title('MRI Attention Map')
            plt.colorbar(im, ax=ax)
            ax.axis('off')

            save_path = result_dir / 'mri_attention.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            return save_path
        except Exception as e:
            logger.warning(f"Failed to save MRI attention map: {e}")
            return None

    def _get_risk_category(self, risk_score: float) -> str:
        """Convert risk score to category"""
        if risk_score > 0.7:
            return 'High'
        elif risk_score > 0.4:
            return 'Intermediate'
        else:
            return 'Low'

    def _generate_who_classification(
        self,
        grade: str,
        idh: str,
        codel: str
    ) -> str:
        """Generate WHO 2021 CNS tumor classification"""
        if grade == 'G4':
            if idh == 'Wildtype':
                return 'Glioblastoma, IDH-wildtype'
            else:
                return 'Astrocytoma, IDH-mutant, Grade 4'
        elif grade == 'G3':
            if idh == 'Mutant':
                if codel == 'Codel':
                    return 'Oligodendroglioma, IDH-mutant and 1p/19q-codeleted, Grade 3'
                else:
                    return 'Astrocytoma, IDH-mutant, Grade 3'
            else:
                return 'Diffuse glioma, IDH-wildtype, Grade 3'
        elif grade == 'G2':
            if idh == 'Mutant':
                if codel == 'Codel':
                    return 'Oligodendroglioma, IDH-mutant and 1p/19q-codeleted, Grade 2'
                else:
                    return 'Astrocytoma, IDH-mutant, Grade 2'
            else:
                return 'Diffuse glioma, IDH-wildtype, Grade 2'
        else:  # G1
            return 'Low-grade glioma, Grade 1'

    def _generate_clinical_support(
        self,
        grade: str,
        idh: str,
        mgmt: str,
        codel: str,
        cdkn2ab: str = 'Intact',
        atrx: str = 'Wildtype',
        survival_months: float = 12.0,
        risk_score: float = 0.5
    ) -> Dict:
        """Generate clinical decision support recommendations (7 markers per WHO 2021)"""
        treatment = ''
        trials = []
        prognosis = ''
        risk_factors = []

        # Treatment recommendation based on molecular profile
        if grade == 'G4' and idh == 'Wildtype':
            if mgmt == 'Methylated':
                treatment = 'Standard Stupp protocol (maximal safe resection + RT 60Gy + concurrent/adjuvant TMZ)'
                trials = ['NCT04396860 (TTFields)', 'NCT03018288 (Immunotherapy)']
            else:
                treatment = 'Consider RT + alternative alkylating agent or clinical trial'
                trials = ['NCT03632135 (MGMT-unmethylated specific)', 'NCT04396860']
        elif idh == 'Mutant' and codel == 'Codel':
            treatment = 'RT + PCV chemotherapy (consider RT alone for grade 2)'
            trials = ['NCT00887146 (CODEL)', 'NCT04164901']
        elif idh == 'Mutant':
            treatment = 'Maximal safe resection, consider RT +/- TMZ based on grade'
            trials = ['NCT04164901', 'NCT03212274 (IDH inhibitor)']
        else:
            treatment = 'Individualized approach based on complete molecular profile'
            trials = ['Consult tumor board for trial eligibility']

        # Risk factors (per WHO 2021 CNS5 guidelines)
        if cdkn2ab == 'Deleted':
            risk_factors.append('CDKN2A/B homozygous deletion (poor prognosis marker)')
            # CDKN2A/B deletion in IDH-mutant astrocytoma → upgrades to Grade 4
            if idh == 'Mutant' and codel == 'Non-codel' and grade in ['G2', 'G3']:
                treatment += ' NOTE: CDKN2A/B deletion may indicate Grade 4 biology per WHO 2021.'

        if atrx == 'Mutant':
            risk_factors.append('ATRX loss (consistent with astrocytic lineage)')
            # ATRX loss is characteristic of IDH-mutant astrocytomas

        # Prognosis summary
        risk_cat = self._get_risk_category(risk_score)
        prognosis = (
            f"Predicted survival: {survival_months:.1f} months "
            f"(95% CI varies by individual factors). "
            f"Risk category: {risk_cat}. "
        )

        if idh == 'Mutant':
            prognosis += "IDH mutation is associated with improved prognosis. "
        if mgmt == 'Methylated' and grade in ['G3', 'G4']:
            prognosis += "MGMT methylation suggests potential TMZ benefit. "
        if codel == 'Codel':
            prognosis += "1p/19q codeletion is associated with better prognosis and PCV response. "
        if cdkn2ab == 'Deleted':
            prognosis += "CDKN2A/B deletion is associated with worse prognosis. "
        if atrx == 'Mutant' and idh == 'Mutant':
            prognosis += "ATRX loss with IDH mutation confirms astrocytoma diagnosis. "

        return {
            'treatment': treatment,
            'trials': trials,
            'prognosis': prognosis.strip(),
            'risk_factors': risk_factors
        }

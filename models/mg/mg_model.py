"""
MG Model: Gene Expression Clinical Decision Support System
Based on MODEL_SPECIFICATION.md Section 3.4

================================================================================
                            MG MODEL PIPELINE (Updated)
================================================================================
                    ** DEG 제거, Pathway만 사용 **
                    (팀 회의 결정 - 최근 연구 적용)

┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT FEATURES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Gene Expression (Raw)                                                     │
│    └─ Top 2000 genes (by variance) → Gene2Vec SVD → 64-dim latent           │
│    └─ 근거: 89.4% variance 보존 (docs/MG_MODEL_GENE_SELECTION_ANALYSIS.md)  │
│                                                                              │
│ 2. Pathway Scores (Pre-computed, ssGSEA)                                     │
│    └─ HALLMARK_* pathways (48개)                                            │
│    └─ 설명력 제공 + Grade/Survival 예측 개선                                │
│                                                                              │
│ [제거됨] DEG Cluster Scores - 과적합 유발                                    │
│ [제거됨] Clinical Features - 과적합 유발                                     │
│                                                                              │
│ 3. TMZ Signature (Separate Calculation, 학습 X)                              │
│    └─ 6-gene signature score → TMZ 반응 예측                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODEL ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Gene2Vec Encoder ──→ 64-dim  ─┐                                           │
│   (2000 genes + attention)       │                                           │
│                                  ├─→ Fusion Layer ──→ 48-dim shared          │
│   Pathway Encoder ──→ 24-dim ───┘      (88-dim)                             │
│   (48 Hallmark pathways)                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT TASKS (4개)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Survival Risk (Cox Regression)                                            │
│    └─ risk_head → 1-dim score                                               │
│    └─ 해석: score > 0 = High risk, score < 0 = Low risk                     │
│                                                                              │
│ 2. Grade Classification (3-class)                                            │
│    └─ grade_head → 3-dim logits (WHO II, III, IV)                           │
│    └─ softmax → probability                                                  │
│                                                                              │
│ 3. Survival Time (Regression)                                                │
│    └─ surv_time_head → 생존 시간 예측 (days, log-transformed)               │
│                                                                              │
│ 4. Recurrence (Binary Classification)                                        │
│    └─ recurrence_head → 재발 여부 예측 (Primary vs Recurrent)               │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ [별도 계산 - Separate Calculation]                                           │
│                                                                              │
│ 5. TMZ Response (6-gene signature)                                           │
│    └─ 학습 대상 아님, 별도 signature score로 계산                           │
│    └─ Genes: MGMT, MSH2, MSH6, MLH1, PMS2, ALKBH2                           │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ [MM Transfer용 출력]                                                         │
│                                                                              │
│ 6. Gene Latent Vector (64-dim)                                               │
│    └─ Gene2Vec encoder output → MM model fusion                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Performance (Gene2Vec + Pathway):
  - C-Index: 0.7609
  - Grade Accuracy: 0.5935
  - Survival MAE: 396 days
  - Recurrence AUC: 0.6631

Dataset: CGGA 874 patients (stratified by Grade)
Transfer: Gene encoder → MM model
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==================== Model Architecture ====================

class Gene2VecEncoder(nn.Module):
    """
    Gene2Vec-based Gene Expression Encoder
    Uses pre-computed SVD embeddings with attention mechanism

    - SVD: Expression matrix → gene embeddings (co-expression patterns)
    - Attention: Learn which genes are important for each sample
    - Output: 64-dim latent vector for MM transfer
    """

    def __init__(self, gene_embeddings: torch.Tensor, emb_dim: int = 64, dropout: float = 0.4):
        """
        Args:
            gene_embeddings: Pre-computed Gene2Vec embeddings [n_genes, emb_dim]
            emb_dim: Embedding dimension (default 64)
            dropout: Dropout rate
        """
        super().__init__()

        n_genes = gene_embeddings.shape[0]
        self.n_genes = n_genes
        self.emb_dim = emb_dim

        # Gene2Vec embeddings (learnable)
        self.gene_emb = nn.Parameter(gene_embeddings, requires_grad=True)

        # Attention mechanism over genes
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.Tanh(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(32, 1)
        )

        # Gene encoder (after attention pooling)
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, 112),
            nn.LayerNorm(112),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(112, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        self.output_dim = 64

    def forward(self, expr: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            expr: Gene expression values [batch, n_genes]
            return_attention: Whether to return attention weights

        Returns:
            dict with 'z' (latent), optionally 'attention_weights'
        """
        batch_size = expr.shape[0]

        # Weight gene embeddings by expression values
        weighted_emb = expr.unsqueeze(-1) * self.gene_emb.unsqueeze(0)  # [batch, n_genes, emb_dim]

        # Compute attention weights
        attn_scores = self.attention(self.gene_emb.unsqueeze(0).expand(batch_size, -1, -1))
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, n_genes, 1]

        # Attention-weighted pooling
        pooled = (weighted_emb * attn_weights).sum(dim=1)  # [batch, emb_dim]

        # Encode to latent space
        z = self.encoder(pooled)  # [batch, 64]

        outputs = {'z': z}
        if return_attention:
            outputs['attention_weights'] = attn_weights.squeeze(-1)  # [batch, n_genes]

        return outputs


class PathwayEncoder(nn.Module):
    """
    Encoder for ssGSEA Hallmark Pathway scores
    Provides explainable features for CDSS

    - Pathway: ssGSEA Hallmark pathway scores (48 pathways)
    - No DEG (removed to reduce overfitting)
    """

    def __init__(self, n_pathways: int = 48, dropout: float = 0.3):
        super().__init__()

        self.n_pathways = n_pathways

        self.encoder = nn.Sequential(
            nn.Linear(n_pathways, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 24),
            nn.LayerNorm(24)
        )

        # Pathway importance weights for explainability
        self.pathway_importance = nn.Linear(n_pathways, 1, bias=False)

        self.output_dim = 24

    def forward(self, pathway_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pathway_scores: Pathway enrichment scores [batch, n_pathways]

        Returns:
            dict with encoded features and importance scores
        """
        encoded = self.encoder(pathway_scores)

        # Compute importance scores for explainability
        pathway_imp = torch.abs(self.pathway_importance.weight.squeeze())  # [n_pathways]

        return {
            'encoded': encoded,
            'pathway_importance': pathway_imp
        }


class GeneExpressionCDSS(nn.Module):
    """
    MG Model: Gene Expression Clinical Decision Support System

    Architecture (Updated - Gene2Vec + Pathway only):
        Gene2Vec (64-dim) ─┐
                           ├─ Fusion (88-dim) → 48-dim shared → Tasks
        Pathway (24-dim) ──┘

    Tasks (4개):
        1. Survival Risk (Cox regression)
        2. Grade Classification (WHO II/III/IV)
        3. Survival Time (Regression)
        4. Recurrence (Binary)

    Explainability:
        - Gene attention weights → top contributing genes
        - Pathway importance → Hallmark pathways
    """

    GRADE_CLASSES = ['Grade II', 'Grade III', 'Grade IV']

    def __init__(
        self,
        gene_embeddings: torch.Tensor,
        n_pathways: int = 48,
        dropout: float = 0.38
    ):
        super().__init__()

        # Encoders
        self.gene_encoder = Gene2VecEncoder(gene_embeddings, emb_dim=gene_embeddings.shape[1], dropout=dropout)
        self.pathway_encoder = PathwayEncoder(n_pathways, dropout=dropout * 0.75)

        # Fusion dimension: 64 (gene) + 24 (pathway) = 88
        fusion_dim = self.gene_encoder.output_dim + self.pathway_encoder.output_dim

        # Shared fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Task 1: Survival Risk (Cox regression)
        self.risk_head = nn.Linear(48, 1)

        # Task 2: Grade Classification (3-class) - Deeper Head
        self.grade_head = nn.Sequential(
            nn.Linear(48, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 3)
        )

        # Task 3: Survival Time Regression - Deeper Head
        self.surv_time_head = nn.Sequential(
            nn.Linear(48, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

        # Task 4: Recurrence (Binary)
        self.recurrence_head = nn.Linear(48, 1)

    def forward(
        self,
        gene_expression: torch.Tensor,
        pathway_scores: torch.Tensor,
        return_explainability: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            gene_expression: [batch, n_genes] - normalized expression values
            pathway_scores: [batch, n_pathways] - ssGSEA pathway scores
            return_explainability: Whether to return attention/importance weights

        Returns:
            dict with predictions and optionally explainability features
        """
        # Encode each modality
        gene_out = self.gene_encoder(gene_expression, return_attention=return_explainability)
        pathway_out = self.pathway_encoder(pathway_scores)

        # Fuse features
        fused = torch.cat([gene_out['z'], pathway_out['encoded']], dim=-1)

        # Shared representation
        shared = self.fusion(fused)

        # Task predictions
        outputs = {
            'risk': self.risk_head(shared).squeeze(-1),           # [batch] - Task 1
            'grade_logits': self.grade_head(shared),              # [batch, 3] - Task 2
            'surv_time': self.surv_time_head(shared).squeeze(-1), # [batch] - Task 3
            'recurrence': self.recurrence_head(shared).squeeze(-1), # [batch] - Task 4
            'gene_latent': gene_out['z']                          # For MM transfer
        }

        # Add explainability features
        if return_explainability:
            outputs['gene_attention'] = gene_out.get('attention_weights')
            outputs['pathway_importance'] = pathway_out['pathway_importance']
            outputs['gene_contribution'] = gene_out['z'].detach()
            outputs['pathway_contribution'] = pathway_out['encoded'].detach()

        return outputs

    def get_gene_encoder_for_transfer(self) -> nn.Module:
        """Get gene encoder for transfer to MM model"""
        return self.gene_encoder


# ==================== Inference Service ====================

class MGInference:
    """
    MG Model Inference Service
    Gene Expression Clinical Decision Support System

    Features:
    - Gene2Vec-based gene encoding (for MM transfer)
    - Pathway features (for explainability)
    - 4 Tasks: Survival Risk, Grade, Survival Time, Recurrence
    - Clinical recommendations based on predictions
    """

    # Hallmark pathways (48 pathways)
    HALLMARK_PATHWAYS = [
        'HALLMARK_ADIPOGENESIS', 'HALLMARK_ALLOGRAFT_REJECTION', 'HALLMARK_ANDROGEN_RESPONSE',
        'HALLMARK_ANGIOGENESIS', 'HALLMARK_APICAL_JUNCTION', 'HALLMARK_APICAL_SURFACE',
        'HALLMARK_APOPTOSIS', 'HALLMARK_BILE_ACID_METABOLISM', 'HALLMARK_CHOLESTEROL_HOMEOSTASIS',
        'HALLMARK_COAGULATION', 'HALLMARK_COMPLEMENT', 'HALLMARK_DNA_REPAIR',
        'HALLMARK_E2F_TARGETS', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 'HALLMARK_ESTROGEN_RESPONSE_EARLY',
        'HALLMARK_ESTROGEN_RESPONSE_LATE', 'HALLMARK_FATTY_ACID_METABOLISM', 'HALLMARK_G2M_CHECKPOINT',
        'HALLMARK_GLYCOLYSIS', 'HALLMARK_HEDGEHOG_SIGNALING', 'HALLMARK_HEME_METABOLISM',
        'HALLMARK_HYPOXIA', 'HALLMARK_IL2_STAT5_SIGNALING', 'HALLMARK_IL6_JAK_STAT3_SIGNALING',
        'HALLMARK_INFLAMMATORY_RESPONSE', 'HALLMARK_INTERFERON_ALPHA_RESPONSE', 'HALLMARK_INTERFERON_GAMMA_RESPONSE',
        'HALLMARK_KRAS_SIGNALING_DN', 'HALLMARK_KRAS_SIGNALING_UP', 'HALLMARK_MITOTIC_SPINDLE',
        'HALLMARK_MTORC1_SIGNALING', 'HALLMARK_MYC_TARGETS_V1', 'HALLMARK_MYC_TARGETS_V2',
        'HALLMARK_MYOGENESIS', 'HALLMARK_NOTCH_SIGNALING', 'HALLMARK_OXIDATIVE_PHOSPHORYLATION',
        'HALLMARK_P53_PATHWAY', 'HALLMARK_PANCREAS_BETA_CELLS', 'HALLMARK_PEROXISOME',
        'HALLMARK_PI3K_AKT_MTOR_SIGNALING', 'HALLMARK_PROTEIN_SECRETION', 'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY',
        'HALLMARK_SPERMATOGENESIS', 'HALLMARK_TGF_BETA_SIGNALING', 'HALLMARK_TNFA_SIGNALING_VIA_NFKB',
        'HALLMARK_UNFOLDED_PROTEIN_RESPONSE', 'HALLMARK_UV_RESPONSE_DN', 'HALLMARK_UV_RESPONSE_UP'
    ]

    def __init__(
        self,
        model_path: str = None,
        gene_embeddings_path: str = None,
        gene_list_path: str = None,
        device: str = None,
        surv_time_mean: float = 6.654,
        surv_time_std: float = 1.110
    ):
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Configuration
        self.n_genes = 2000
        self.emb_dim = 64
        self.n_pathways = 48

        # Survival time normalization parameters
        self.surv_time_mean = surv_time_mean
        self.surv_time_std = surv_time_std

        # Load gene embeddings (or create placeholder)
        if gene_embeddings_path and Path(gene_embeddings_path).exists():
            gene_emb_data = torch.load(gene_embeddings_path, map_location='cpu')
            self.gene_embeddings = gene_emb_data['embeddings']
            self.gene_list = gene_emb_data.get('genes', [])
            logger.info(f"Loaded gene embeddings: {self.gene_embeddings.shape}")
        else:
            # Placeholder embeddings for demo
            self.gene_embeddings = torch.randn(self.n_genes, self.emb_dim)
            self.gene_list = [f'GENE_{i}' for i in range(self.n_genes)]
            logger.warning("Using random gene embeddings (demo mode)")

        # Initialize model
        self.model = GeneExpressionCDSS(
            gene_embeddings=self.gene_embeddings,
            n_pathways=self.n_pathways
        )

        # Load weights
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            # Load normalization params if available
            if 'surv_time_mean' in checkpoint:
                self.surv_time_mean = checkpoint['surv_time_mean']
                self.surv_time_std = checkpoint['surv_time_std']

            logger.info(f"Loaded MG model from {model_path}")
        else:
            logger.warning("MG model weights not found, using random initialization (demo mode)")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Results directory
        self.results_dir = Path(os.environ.get('RESULTS_PATH', './results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        patient_id: str,
        gene_expression: Dict[str, float],
        pathway_scores: Dict[str, float],
        clinical_data: Dict[str, Any] = None,
        include_explainability: bool = True
    ) -> Dict:
        """
        Run MG (Gene Expression CDSS) analysis

        Args:
            patient_id: Patient identifier
            gene_expression: Dict mapping gene names to expression values
            pathway_scores: Dict mapping pathway names to enrichment scores
            clinical_data: Dict with clinical info (for recommendations only)
            include_explainability: Include detailed explainability outputs

        Returns:
            dict: Comprehensive CDSS analysis results
        """
        start_time = datetime.now()

        if clinical_data is None:
            clinical_data = {}

        # Prepare inputs
        expr_tensor = self._prepare_expression(gene_expression)
        pathway_tensor = self._prepare_pathway_scores(pathway_scores)

        # Move to device
        expr_tensor = expr_tensor.to(self.device)
        pathway_tensor = pathway_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(
                expr_tensor,
                pathway_tensor,
                return_explainability=include_explainability
            )

        # Process predictions
        risk_score = outputs['risk'].cpu().item()
        grade_probs = F.softmax(outputs['grade_logits'], dim=-1).cpu().numpy().squeeze()
        grade_pred = int(np.argmax(grade_probs))

        # Survival time (denormalize from log-scale)
        surv_time_norm = outputs['surv_time'].cpu().item()
        surv_time_log = surv_time_norm * self.surv_time_std + self.surv_time_mean
        surv_time_days = np.expm1(surv_time_log)

        # Recurrence probability
        recurrence_prob = torch.sigmoid(outputs['recurrence']).cpu().item()

        # Build result
        result = {
            'patient_id': patient_id,
            'survival_risk': {
                'score': round(risk_score, 4),
                'category': 'High' if risk_score > 0 else 'Low',
                'interpretation': self._interpret_risk(risk_score)
            },
            'grade_prediction': {
                'predicted': GeneExpressionCDSS.GRADE_CLASSES[grade_pred],
                'confidence': round(float(grade_probs[grade_pred]), 3),
                'probabilities': {
                    cls: round(float(p), 3)
                    for cls, p in zip(GeneExpressionCDSS.GRADE_CLASSES, grade_probs)
                }
            },
            'survival_time': {
                'predicted_days': round(surv_time_days, 0),
                'predicted_months': round(surv_time_days / 30, 1),
                'interpretation': self._interpret_survival_time(surv_time_days)
            },
            'recurrence': {
                'probability': round(recurrence_prob, 3),
                'prediction': 'Recurrent' if recurrence_prob > 0.5 else 'Primary',
                'interpretation': self._interpret_recurrence(recurrence_prob)
            },
            'molecular_profile': {
                'idh_status': 'Mutant' if str(clinical_data.get('idh', '')).lower() in ['mutant', 'mut', '1'] else 'Wildtype',
                '1p19q_status': clinical_data.get('1p19q', 'Unknown'),
                'mgmt_status': clinical_data.get('mgmt', 'Unknown')
            }
        }

        # Add explainability
        if include_explainability:
            result['explainability'] = self._generate_explainability(
                outputs, gene_expression, pathway_scores
            )

        # Generate recommendations
        result['clinical_recommendation'] = self._generate_recommendations(
            result['survival_risk'],
            result['grade_prediction'],
            result['recurrence'],
            clinical_data
        )

        # Add latent for MM transfer
        result['gene_latent'] = outputs['gene_latent'].cpu().numpy().tolist()

        # Processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result['processing_time_ms'] = int(processing_time)

        logger.info(
            f"MG analysis completed for {patient_id}: "
            f"Risk={result['survival_risk']['category']}, "
            f"Grade={result['grade_prediction']['predicted']}, "
            f"Survival={result['survival_time']['predicted_months']:.1f}mo, "
            f"Recurrence={result['recurrence']['prediction']}"
        )

        return result

    def _prepare_expression(self, gene_expr: Dict[str, float]) -> torch.Tensor:
        """Prepare gene expression tensor"""
        values = []
        for gene in self.gene_list[:self.n_genes]:
            values.append(gene_expr.get(gene, 0.0))

        while len(values) < self.n_genes:
            values.append(0.0)

        values = np.array(values, dtype=np.float32)
        if values.std() > 0:
            values = (values - values.mean()) / values.std()

        return torch.from_numpy(values).unsqueeze(0)

    def _prepare_pathway_scores(self, pathway_scores: Dict[str, float]) -> torch.Tensor:
        """Prepare pathway scores tensor"""
        values = [pathway_scores.get(p, 0.0) for p in self.HALLMARK_PATHWAYS[:self.n_pathways]]
        while len(values) < self.n_pathways:
            values.append(0.0)

        # Standardize
        values = np.array(values, dtype=np.float32)
        if values.std() > 0:
            values = (values - values.mean()) / values.std()

        return torch.tensor([values], dtype=torch.float32)

    def _interpret_risk(self, risk_score: float) -> str:
        """Interpret risk score"""
        if risk_score > 0.5:
            return 'High survival risk - aggressive treatment may be needed'
        elif risk_score > 0:
            return 'Moderate-high survival risk - close monitoring recommended'
        elif risk_score > -0.5:
            return 'Moderate-low survival risk - standard treatment protocol'
        else:
            return 'Low survival risk - favorable prognosis'

    def _interpret_survival_time(self, days: float) -> str:
        """Interpret predicted survival time"""
        months = days / 30
        if months < 12:
            return f'Short-term prognosis ({months:.0f} months)'
        elif months < 24:
            return f'Intermediate prognosis ({months:.0f} months)'
        elif months < 60:
            return f'Good prognosis ({months:.0f} months)'
        else:
            return f'Long-term survival expected ({months:.0f} months)'

    def _interpret_recurrence(self, prob: float) -> str:
        """Interpret recurrence probability"""
        if prob > 0.7:
            return 'High recurrence risk - intensive monitoring recommended'
        elif prob > 0.5:
            return 'Moderate recurrence risk - regular follow-up needed'
        elif prob > 0.3:
            return 'Low-moderate recurrence risk'
        else:
            return 'Low recurrence risk - standard follow-up'

    def _generate_explainability(
        self,
        outputs: Dict,
        gene_expr: Dict[str, float],
        pathway_scores: Dict[str, float]
    ) -> Dict:
        """Generate explainability outputs"""
        explain = {}

        # Top genes by attention weight
        if 'gene_attention' in outputs and outputs['gene_attention'] is not None:
            attn = outputs['gene_attention'].cpu().numpy().squeeze()
            top_gene_idx = np.argsort(attn)[-10:][::-1]
            explain['top_genes'] = [
                {
                    'gene': self.gene_list[i] if i < len(self.gene_list) else f'Gene_{i}',
                    'attention': round(float(attn[i]), 4),
                    'expression': round(gene_expr.get(self.gene_list[i], 0.0), 3) if i < len(self.gene_list) else 0.0
                }
                for i in top_gene_idx
            ]

        # Top pathways
        if 'pathway_importance' in outputs:
            path_imp = outputs['pathway_importance'].cpu().numpy()
            top_path_idx = np.argsort(path_imp)[-10:][::-1]
            explain['top_pathways'] = [
                {
                    'pathway': self.HALLMARK_PATHWAYS[i] if i < len(self.HALLMARK_PATHWAYS) else f'Pathway_{i}',
                    'importance': round(float(path_imp[i]), 4),
                    'score': round(list(pathway_scores.values())[i] if i < len(pathway_scores) else 0.0, 3)
                }
                for i in top_path_idx
            ]

        # Feature contributions
        if 'gene_contribution' in outputs and 'pathway_contribution' in outputs:
            gene_contrib = torch.norm(outputs['gene_contribution']).item()
            pathway_contrib = torch.norm(outputs['pathway_contribution']).item()
            total = gene_contrib + pathway_contrib + 1e-8
            explain['feature_contributions'] = {
                'gene2vec': round(gene_contrib / total, 3),
                'pathway': round(pathway_contrib / total, 3)
            }

        return explain

    def _generate_recommendations(
        self,
        survival: Dict,
        grade: Dict,
        recurrence: Dict,
        clinical: Dict
    ) -> Dict:
        """Generate clinical recommendations"""
        recommendations = []

        # Based on survival risk
        if survival['category'] == 'High':
            recommendations.append('Consider aggressive multimodal therapy')
            recommendations.append('Frequent follow-up imaging (every 2-3 months)')
        else:
            recommendations.append('Standard treatment protocol appropriate')
            recommendations.append('Regular follow-up imaging (every 3-4 months)')

        # Based on recurrence
        if recurrence['prediction'] == 'Recurrent':
            recommendations.append('Recurrence pattern detected - consider re-operation evaluation')

        # Based on molecular markers
        if str(clinical.get('mgmt', '')).lower() in ['methylated', '1']:
            recommendations.append('MGMT methylated - TMZ likely beneficial')

        if str(clinical.get('1p19q', '')).lower() in ['codel', 'codeleted', '1']:
            recommendations.append('1p/19q codeleted - PCV chemotherapy consideration')

        # Based on grade
        if 'Grade IV' in grade['predicted']:
            recommendations.append('GBM - Stupp protocol (RT + TMZ) recommended')
            recommendations.append('Consider clinical trial enrollment')

        return {
            'primary': recommendations[0] if recommendations else 'Standard care',
            'all_recommendations': recommendations,
            'monitoring': 'MRI every 2-3 months' if survival['category'] == 'High' else 'MRI every 3-4 months'
        }

    def get_gene_latent_for_mm(self, gene_expression: Dict[str, float]) -> np.ndarray:
        """
        Get gene latent vector for MM model transfer

        Args:
            gene_expression: Dict mapping gene names to expression values

        Returns:
            64-dim numpy array for MM fusion
        """
        expr_tensor = self._prepare_expression(gene_expression).to(self.device)

        with torch.no_grad():
            gene_out = self.model.gene_encoder(expr_tensor)

        return gene_out['z'].cpu().numpy().squeeze()

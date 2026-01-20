"""
Brain Tumor CDSS - Inference Pipeline

Usage:
    from inference.predict import BrainTumorPredictor

    predictor = BrainTumorPredictor(
        m1_weights="weights/m1_model.pth",
        mg_weights="weights/mg_model.pth",
        mm_weights="weights/mm_model.pth"
    )

    result = predictor.predict(
        mri_path="path/to/mri",
        gene_expression={"EGFR": 1.2, "TP53": 0.8, ...},
        protein_data={"p-AKT": 0.5, ...}
    )
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.m1 import M1Inference
from models.mg import MGInference
from models.mm import MMInference

logger = logging.getLogger(__name__)


class BrainTumorPredictor:
    """
    Unified predictor for Brain Tumor CDSS

    Combines M1 (MRI), MG (Gene), and MM (Multimodal) models
    for comprehensive brain tumor analysis.
    """

    def __init__(
        self,
        m1_weights: str = None,
        mg_weights: str = None,
        mm_weights: str = None,
        gene_embeddings_path: str = None,
        device: str = None
    ):
        """
        Initialize predictor with model weights

        Args:
            m1_weights: Path to M1 (MRI encoder) weights
            mg_weights: Path to MG (Gene encoder) weights
            mm_weights: Path to MM (Fusion) weights
            gene_embeddings_path: Path to Gene2Vec embeddings
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.m1 = M1Inference(model_path=m1_weights, device=self.device)
        self.mg = MGInference(
            model_path=mg_weights,
            gene_embeddings_path=gene_embeddings_path,
            device=self.device
        )
        self.mm = MMInference(
            model_path=mm_weights,
            m1_weights_path=m1_weights,
            mg_weights_path=mg_weights,
            device=self.device
        )

        logger.info(f"BrainTumorPredictor initialized on {self.device}")

    def predict(
        self,
        mri_path: str = None,
        gene_expression: Dict[str, float] = None,
        protein_data: Dict[str, float] = None,
        clinical_data: Dict[str, Any] = None,
        patient_id: str = "patient_001"
    ) -> Dict:
        """
        Run full prediction pipeline

        Args:
            mri_path: Path to MRI data (NIfTI format)
            gene_expression: Dict of gene names to expression values
            protein_data: Dict of protein names to expression values
            clinical_data: Dict with clinical features (age, sex, etc.)
            patient_id: Patient identifier

        Returns:
            Dict with comprehensive predictions and explanations
        """
        results = {
            'patient_id': patient_id,
            'modalities_used': []
        }

        # MRI-only analysis (M1)
        if mri_path:
            try:
                m1_result = self.m1.analyze(
                    mri_path=mri_path,
                    patient_id=patient_id,
                    patient_info=clinical_data
                )
                results['mri_analysis'] = m1_result
                results['modalities_used'].append('MRI')
            except Exception as e:
                logger.error(f"M1 analysis failed: {e}")
                results['mri_analysis'] = {'error': str(e)}

        # Gene-only analysis (MG)
        if gene_expression:
            try:
                # Prepare pathway scores (simplified - normally from ssGSEA)
                pathway_scores = self._compute_pathway_scores(gene_expression)

                mg_result = self.mg.analyze(
                    patient_id=patient_id,
                    gene_expression=gene_expression,
                    pathway_scores=pathway_scores,
                    clinical_data=clinical_data
                )
                results['gene_analysis'] = mg_result
                results['modalities_used'].append('Gene')
            except Exception as e:
                logger.error(f"MG analysis failed: {e}")
                results['gene_analysis'] = {'error': str(e)}

        # Multimodal analysis (MM) - requires all modalities
        if mri_path and gene_expression:
            try:
                mm_result = self.mm.analyze(
                    mri_path=mri_path,
                    patient_id=patient_id,
                    genomic_data={'expression': list(gene_expression.values())},
                    proteomic_data={'expression': list(protein_data.values())} if protein_data else None,
                    clinical_data=clinical_data
                )
                results['multimodal_analysis'] = mm_result
                results['modalities_used'].append('Multimodal')
            except Exception as e:
                logger.error(f"MM analysis failed: {e}")
                results['multimodal_analysis'] = {'error': str(e)}

        # Generate summary
        results['summary'] = self._generate_summary(results)

        return results

    def predict_survival(
        self,
        mri_features: np.ndarray = None,
        gene_features: np.ndarray = None,
        protein_features: np.ndarray = None
    ) -> Dict:
        """
        Predict survival using pre-extracted features

        Args:
            mri_features: Pre-extracted MRI features (768-dim)
            gene_features: Pre-extracted gene features (64-dim)
            protein_features: Protein expression features (229-dim)

        Returns:
            Dict with survival prediction
        """
        # Convert to tensors
        mri_feat = torch.tensor(mri_features, dtype=torch.float32).unsqueeze(0)
        gene_feat = torch.tensor(gene_features, dtype=torch.float32).unsqueeze(0)
        protein_feat = torch.tensor(protein_features, dtype=torch.float32).unsqueeze(0)
        clinical_feat = torch.zeros(1, 10)  # Default clinical features

        # Move to device
        mri_feat = mri_feat.to(self.device)
        gene_feat = gene_feat.to(self.device)
        protein_feat = protein_feat.to(self.device)
        clinical_feat = clinical_feat.to(self.device)

        # Run MM model
        with torch.no_grad():
            outputs = self.mm.model(
                mri_feat, gene_feat, protein_feat, clinical_feat
            )

        # Extract survival prediction
        survival_mean = outputs['survival_mean'].cpu().item()
        risk_score = torch.sigmoid(outputs['risk_score']).cpu().item()

        return {
            'survival_months': max(0, survival_mean * 36 + 12),
            'risk_score': risk_score,
            'risk_category': 'High' if risk_score > 0.7 else 'Intermediate' if risk_score > 0.4 else 'Low'
        }

    def _compute_pathway_scores(self, gene_expression: Dict[str, float]) -> Dict[str, float]:
        """
        Compute pathway scores from gene expression
        (Simplified version - full implementation uses ssGSEA)
        """
        # Return placeholder scores for demo
        hallmark_pathways = [
            'HALLMARK_APOPTOSIS', 'HALLMARK_CELL_CYCLE',
            'HALLMARK_DNA_REPAIR', 'HALLMARK_GLYCOLYSIS',
            'HALLMARK_HYPOXIA', 'HALLMARK_INFLAMMATORY_RESPONSE',
            'HALLMARK_P53_PATHWAY', 'HALLMARK_PI3K_AKT_MTOR_SIGNALING'
        ]

        return {p: np.random.randn() for p in hallmark_pathways}

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary from all analyses"""
        summary = {
            'overall_risk': 'Unknown',
            'key_findings': [],
            'recommended_actions': []
        }

        # Determine overall risk from available analyses
        if 'multimodal_analysis' in results and 'error' not in results['multimodal_analysis']:
            mm = results['multimodal_analysis']
            summary['overall_risk'] = mm.get('survival', {}).get('risk_category', 'Unknown')
            summary['key_findings'].append(
                f"WHO Classification: {mm.get('clinical_decision_support', {}).get('who_classification', 'N/A')}"
            )
        elif 'gene_analysis' in results and 'error' not in results['gene_analysis']:
            mg = results['gene_analysis']
            summary['overall_risk'] = mg.get('survival_risk', {}).get('category', 'Unknown')
        elif 'mri_analysis' in results and 'error' not in results['mri_analysis']:
            m1 = results['mri_analysis']
            summary['overall_risk'] = m1.get('survival', {}).get('risk_category', 'Unknown')

        return summary


def main():
    """Demo usage"""
    print("Brain Tumor CDSS - Demo")
    print("=" * 50)

    # Initialize predictor (will use random weights in demo mode)
    predictor = BrainTumorPredictor()

    # Demo with synthetic data
    demo_genes = {f'GENE_{i}': np.random.randn() for i in range(100)}
    demo_proteins = {f'PROTEIN_{i}': np.random.randn() for i in range(50)}

    print("\nRunning gene-only analysis...")
    result = predictor.predict(
        gene_expression=demo_genes,
        protein_data=demo_proteins,
        patient_id="demo_patient"
    )

    print(f"\nModalities used: {result['modalities_used']}")
    print(f"Overall risk: {result['summary']['overall_risk']}")

    if 'gene_analysis' in result and 'error' not in result['gene_analysis']:
        ga = result['gene_analysis']
        print(f"\nGene Analysis:")
        print(f"  - Survival Risk: {ga.get('survival_risk', {}).get('category', 'N/A')}")
        print(f"  - Grade: {ga.get('grade_prediction', {}).get('predicted', 'N/A')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

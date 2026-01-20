"""
Sample Data Module for Brain Tumor CDSS

Provides synthetic sample data for testing and demonstration purposes.
This data is randomly generated and does not represent real patient information.
"""

import numpy as np
from typing import Dict, Any


def load_sample_data() -> Dict[str, Any]:
    """
    Load synthetic sample data for testing.

    Returns:
        Dict containing:
        - gene_expression: Dict[str, float] - 100 gene expression values
        - pathway_scores: Dict[str, float] - 6 pathway enrichment scores
        - clinical_data: Dict[str, Any] - Clinical features
        - mri_shape: Tuple - Expected MRI input shape
    """
    np.random.seed(42)

    # Sample gene expression (100 genes)
    gene_names = [
        'EGFR', 'PTEN', 'TP53', 'IDH1', 'ATRX', 'TERT', 'PIK3CA', 'NF1',
        'RB1', 'CDKN2A', 'MDM2', 'CDK4', 'PDGFRA', 'MET', 'FGFR3', 'MGMT',
    ] + [f'GENE_{i}' for i in range(84)]

    gene_expression = {
        name: float(np.random.randn())
        for name in gene_names
    }

    # Sample pathway scores (Hallmark pathways)
    pathway_names = [
        'HALLMARK_APOPTOSIS',
        'HALLMARK_CELL_CYCLE',
        'HALLMARK_DNA_REPAIR',
        'HALLMARK_GLYCOLYSIS',
        'HALLMARK_HYPOXIA',
        'HALLMARK_P53_PATHWAY',
    ]

    pathway_scores = {
        name: float(np.random.randn())
        for name in pathway_names
    }

    # Sample clinical data
    clinical_data = {
        'age': 55,
        'sex': 'M',
        'kps': 80,
        'tumor_location': 'frontal',
        'extent_of_resection': 'GTR',
        'prior_treatment': False,
    }

    return {
        'gene_expression': gene_expression,
        'pathway_scores': pathway_scores,
        'clinical_data': clinical_data,
        'mri_shape': (4, 128, 128, 128),
        'protein_dim': 167,
    }


def load_sample_mri() -> np.ndarray:
    """
    Generate synthetic MRI data for testing.

    Returns:
        np.ndarray: Shape (4, 128, 128, 128) - T1, T1ce, T2, FLAIR

    Note:
        This is random noise, not real MRI data.
        For actual testing, use real preprocessed MRI data.
    """
    np.random.seed(42)

    # Generate 4-channel random volume
    mri = np.random.randn(4, 128, 128, 128).astype(np.float32)

    # Add some structure (sphere in center to simulate tumor)
    center = np.array([64, 64, 64])
    radius = 20

    x, y, z = np.ogrid[:128, :128, :128]
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    mask = distance < radius

    # Enhance tumor region in T1ce channel
    mri[1][mask] += 2.0

    return mri


def load_sample_protein() -> np.ndarray:
    """
    Generate synthetic protein (RPPA) data for testing.

    Returns:
        np.ndarray: Shape (167,) - RPPA protein levels
    """
    np.random.seed(42)
    return np.random.randn(167).astype(np.float32)


def load_sample_clinical_tensor() -> np.ndarray:
    """
    Generate clinical feature tensor for MM model input.

    Returns:
        np.ndarray: Shape (10,) - Normalized clinical features
    """
    np.random.seed(42)
    return np.random.randn(10).astype(np.float32)


# Sample patient information
SAMPLE_PATIENT = {
    'patient_id': 'SAMPLE_001',
    'description': 'Synthetic sample patient for testing',
    'data': load_sample_data(),
}


if __name__ == '__main__':
    # Test loading
    data = load_sample_data()
    print(f"Gene expression: {len(data['gene_expression'])} genes")
    print(f"Pathway scores: {len(data['pathway_scores'])} pathways")
    print(f"Clinical data: {data['clinical_data']}")

    mri = load_sample_mri()
    print(f"MRI shape: {mri.shape}")

    protein = load_sample_protein()
    print(f"Protein shape: {protein.shape}")

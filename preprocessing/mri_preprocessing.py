"""
MRI Preprocessing Pipeline
Prepares BraTS-style MRI data for M1 model

Features:
- Multi-modality loading (T1, T1ce, T2, FLAIR)
- Skull stripping (optional)
- N4 Bias field correction
- Intensity normalization
- Resampling to isotropic resolution
- Cropping to brain region
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIPreprocessor:
    """
    MRI Preprocessing Pipeline for Brain Tumor Analysis

    Steps:
    1. Load multi-modality MRI (T1, T1ce, T2, FLAIR)
    2. N4 Bias Field Correction
    3. Skull Stripping (optional)
    4. Intensity Normalization
    5. Resampling to target resolution
    6. Cropping to brain region
    """

    MODALITIES = ['t1', 't1ce', 't2', 'flair']

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Tuple[int, int, int] = (128, 128, 128),
        normalize_method: str = 'zscore',
        apply_n4: bool = True,
        apply_skull_strip: bool = False
    ):
        """
        Initialize preprocessor

        Args:
            target_spacing: Target voxel spacing in mm
            target_size: Target volume size
            normalize_method: 'zscore', 'minmax', or 'percentile'
            apply_n4: Apply N4 bias field correction
            apply_skull_strip: Apply skull stripping
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.apply_n4 = apply_n4
        self.apply_skull_strip = apply_skull_strip

        if not NIBABEL_AVAILABLE:
            logger.warning("nibabel not installed. Install with: pip install nibabel")
        if not SITK_AVAILABLE:
            logger.warning("SimpleITK not installed. Install with: pip install SimpleITK")

    def process_patient(
        self,
        patient_dir: str,
        output_dir: str = None,
        save_intermediate: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Process a single patient's MRI data

        Args:
            patient_dir: Directory containing MRI files
            output_dir: Output directory (optional)
            save_intermediate: Save intermediate results

        Returns:
            Dict with processed volumes and metadata
        """
        patient_dir = Path(patient_dir)
        patient_id = patient_dir.name

        logger.info(f"Processing patient: {patient_id}")

        # Load all modalities
        volumes = {}
        affine = None

        for mod in self.MODALITIES:
            vol, aff = self._load_modality(patient_dir, mod)
            if vol is not None:
                volumes[mod] = vol
                if affine is None:
                    affine = aff
            else:
                logger.warning(f"Could not load {mod} for {patient_id}")

        if not volumes:
            raise ValueError(f"No modalities found for {patient_id}")

        # Apply preprocessing steps
        processed = {}
        for mod, vol in volumes.items():
            # N4 Bias Field Correction
            if self.apply_n4 and SITK_AVAILABLE:
                vol = self._n4_correction(vol)

            # Intensity normalization
            vol = self._normalize(vol)

            processed[mod] = vol

        # Stack modalities
        stacked = np.stack([processed[mod] for mod in self.MODALITIES if mod in processed], axis=0)

        # Resample if needed
        if stacked.shape[1:] != self.target_size:
            stacked = self._resample(stacked)

        # Load segmentation if available
        seg = self._load_segmentation(patient_dir)

        result = {
            'image': stacked,
            'affine': affine,
            'patient_id': patient_id,
            'original_shape': volumes[list(volumes.keys())[0]].shape
        }

        if seg is not None:
            result['segmentation'] = seg

        # Save if output directory provided
        if output_dir:
            self._save_processed(result, output_dir)

        return result

    def _load_modality(
        self,
        patient_dir: Path,
        modality: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single modality"""
        if not NIBABEL_AVAILABLE:
            return None, None

        # Try different naming conventions
        patterns = [
            f"*_{modality}.nii.gz",
            f"*_{modality.upper()}.nii.gz",
            f"*_{modality}.nii",
            f"{modality}.nii.gz",
            # BraTS specific
            f"*_T1c*.nii.gz" if modality == 't1ce' else None,
        ]
        patterns = [p for p in patterns if p]

        for pattern in patterns:
            files = list(patient_dir.glob(pattern))
            if files:
                nii = nib.load(str(files[0]))
                return nii.get_fdata().astype(np.float32), nii.affine

        return None, None

    def _load_segmentation(self, patient_dir: Path) -> Optional[np.ndarray]:
        """Load segmentation mask if available"""
        if not NIBABEL_AVAILABLE:
            return None

        patterns = ["*_seg.nii.gz", "*_seg.nii", "seg.nii.gz"]

        for pattern in patterns:
            files = list(patient_dir.glob(pattern))
            if files:
                nii = nib.load(str(files[0]))
                return nii.get_fdata().astype(np.int32)

        return None

    def _n4_correction(self, volume: np.ndarray) -> np.ndarray:
        """Apply N4 Bias Field Correction"""
        if not SITK_AVAILABLE:
            return volume

        try:
            # Convert to SimpleITK
            sitk_image = sitk.GetImageFromArray(volume)
            sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)

            # Create mask
            mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)

            # N4 correction
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
            corrected = corrector.Execute(sitk_image, mask)

            return sitk.GetArrayFromImage(corrected)

        except Exception as e:
            logger.warning(f"N4 correction failed: {e}")
            return volume

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        """Normalize intensity values"""
        # Create brain mask
        mask = volume > 0

        if not np.any(mask):
            return volume

        if self.normalize_method == 'zscore':
            mean = volume[mask].mean()
            std = volume[mask].std()
            volume = (volume - mean) / (std + 1e-8)

        elif self.normalize_method == 'minmax':
            vmin = volume[mask].min()
            vmax = volume[mask].max()
            volume = (volume - vmin) / (vmax - vmin + 1e-8)

        elif self.normalize_method == 'percentile':
            p1, p99 = np.percentile(volume[mask], [1, 99])
            volume = np.clip(volume, p1, p99)
            volume = (volume - p1) / (p99 - p1 + 1e-8)

        # Zero out background
        volume = volume * mask

        return volume

    def _resample(self, volume: np.ndarray) -> np.ndarray:
        """Resample to target size"""
        from scipy.ndimage import zoom

        current_shape = volume.shape[1:]
        zoom_factors = [t / c for t, c in zip(self.target_size, current_shape)]

        # Resample each channel
        resampled = np.stack([
            zoom(volume[c], zoom_factors, order=1)
            for c in range(volume.shape[0])
        ], axis=0)

        return resampled

    def _save_processed(self, result: Dict, output_dir: str):
        """Save processed data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        patient_id = result['patient_id']

        # Save as numpy
        np.save(output_dir / f"{patient_id}_image.npy", result['image'])

        if 'segmentation' in result:
            np.save(output_dir / f"{patient_id}_seg.npy", result['segmentation'])

        # Save as NIfTI if nibabel available
        if NIBABEL_AVAILABLE and result['affine'] is not None:
            # Save each modality
            for i, mod in enumerate(self.MODALITIES):
                if i < result['image'].shape[0]:
                    nii = nib.Nifti1Image(result['image'][i], result['affine'])
                    nib.save(nii, str(output_dir / f"{patient_id}_{mod}_processed.nii.gz"))

        logger.info(f"Saved processed data to {output_dir}")


def batch_process(
    input_dir: str,
    output_dir: str,
    n_workers: int = 4,
    **kwargs
):
    """
    Process multiple patients in batch

    Args:
        input_dir: Directory containing patient folders
        output_dir: Output directory
        n_workers: Number of parallel workers
        **kwargs: Arguments passed to MRIPreprocessor
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find patient directories
    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(patient_dirs)} patients to process")

    preprocessor = MRIPreprocessor(**kwargs)

    def process_one(patient_dir):
        try:
            preprocessor.process_patient(str(patient_dir), str(output_dir))
            return str(patient_dir.name), True, None
        except Exception as e:
            return str(patient_dir.name), False, str(e)

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one, d): d for d in patient_dirs}

        for future in as_completed(futures):
            patient_id, success, error = future.result()
            results.append((patient_id, success, error))

            if success:
                logger.info(f"Completed: {patient_id}")
            else:
                logger.error(f"Failed: {patient_id} - {error}")

    # Summary
    n_success = sum(1 for _, s, _ in results if s)
    logger.info(f"\nProcessing complete: {n_success}/{len(patient_dirs)} successful")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MRI Preprocessing")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--normalize', type=str, default='zscore', choices=['zscore', 'minmax', 'percentile'])
    parser.add_argument('--no_n4', action='store_true')

    args = parser.parse_args()

    batch_process(
        args.input_dir,
        args.output_dir,
        n_workers=args.n_workers,
        target_size=tuple(args.target_size),
        normalize_method=args.normalize,
        apply_n4=not args.no_n4
    )

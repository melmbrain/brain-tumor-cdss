# Contributing to Brain Tumor CDSS

Thank you for your interest in contributing to the Brain Tumor Clinical Decision Support System!

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/melmbrain/brain-tumor-cdss/issues) page
- Check existing issues before creating a new one
- Provide detailed information:
  - Python version
  - PyTorch version
  - Error messages and stack traces
  - Steps to reproduce

### Submitting Pull Requests

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions
   - Update documentation if needed

4. **Test your changes**
   ```bash
   python -c "from models import *"
   python -m pytest tests/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add: brief description of changes"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8
- Use type hints for function parameters
- Maximum line length: 120 characters
- Use meaningful variable names

### Example

```python
def process_mri(
    mri_path: str,
    target_size: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """
    Process MRI volume for model input.

    Args:
        mri_path: Path to MRI file (.nii.gz)
        target_size: Target volume dimensions

    Returns:
        Preprocessed MRI volume as numpy array
    """
    ...
```

## Project Structure

```
brain-tumor-cdss/
├── models/          # Model architectures
├── training/        # Training scripts
├── preprocessing/   # Data preprocessing
├── inference/       # Inference pipeline
├── configs/         # Configuration files
├── docs/            # Documentation
└── tests/           # Unit tests
```

## Areas for Contribution

- [ ] Add more preprocessing options
- [ ] Implement additional model architectures
- [ ] Improve documentation
- [ ] Add unit tests
- [ ] Create visualization tools
- [ ] Optimize inference speed

## Questions?

Feel free to open an issue or contact the maintainers.

Thank you for contributing!

# Contributing

Thank you for your interest in contributing to Brain Tumor CDSS!

## How to Contribute

### Reporting Issues

1. Check existing [issues](https://github.com/melmbrain/brain-tumor-cdss/issues)
2. Use the appropriate issue template
3. Provide detailed information:
   - Environment (OS, Python version, PyTorch version)
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Format code: `black . && isort .`
6. Commit: `git commit -m "Add my feature"`
7. Push: `git push origin feature/my-feature`
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/brain-tumor-cdss.git
cd brain-tumor-cdss

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

## Code Style

We follow PEP 8 with these tools:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black .
isort .

# Check linting
flake8 .

# Type checking
mypy models/
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=models --cov-report=html

# Run specific test file
pytest tests/test_m1.py -v
```

## Documentation

Documentation uses MkDocs with Material theme:

```bash
# Install docs dependencies
pip install -e .[docs]

# Serve locally
mkdocs serve

# Build
mkdocs build
```

## Pull Request Guidelines

### Checklist

- [ ] Code follows project style
- [ ] Tests pass locally
- [ ] New features have tests
- [ ] Documentation updated
- [ ] Commit messages are clear

### Commit Message Format

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Review Process

1. Maintainers will review within 1 week
2. Address feedback promptly
3. Squash commits before merge

## Project Structure

```
brain-tumor-cdss/
├── models/           # Model implementations
│   ├── m1/          # MRI encoder
│   ├── mg/          # Gene VAE
│   └── mm/          # Multimodal fusion
├── preprocessing/    # Data preprocessing
├── training/         # Training scripts
├── inference/        # Inference pipeline
├── tests/            # Unit tests
├── notebooks/        # Demo notebooks
└── docs/             # Documentation
```

## Questions?

- Open a [Discussion](https://github.com/melmbrain/brain-tumor-cdss/discussions)
- Email: melmbrain@example.com

Thank you for contributing!

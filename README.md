# AI Crop Disease Identifier

A hierarchical, two-stage deep learning system for identifying crop types and classifying crop-specific diseases from leaf images.

## Project Overview

This prototype system uses a two-stage pipeline:
- **Stage 1**: Crop identification (14 crop types)
- **Stage 2**: Apple disease classification (4 disease classes) - triggered only for Apple crops

## Project Structure

```
crop-disease-identifier/
├── data/                          # Dataset directory
│   ├── raw/                       # Raw PlantVillage images
│   ├── processed/                 # Preprocessed images
│   └── splits/                    # Train/val/test splits
├── models/                        # Trained model checkpoints
│   ├── stage1/                    # Crop identification models
│   └── stage2/                    # Disease classification models
├── src/                           # Source code
│   ├── __init__.py
│   ├── data/                      # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py              # Dataset loading
│   │   └── preprocessing.py       # Image preprocessing and augmentation
│   ├── models/                    # Model architectures
│   │   ├── __init__.py
│   │   ├── stage1.py              # Crop identification model
│   │   └── stage2.py              # Disease classification model
│   ├── training/                  # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   └── evaluation.py          # Evaluation metrics
│   ├── inference/                 # Inference pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py            # Hierarchical inference pipeline
│   │   ├── validation.py          # Input validation gates
│   │   └── recommendations.py     # Treatment recommendations
│   ├── web/                       # Web application
│   │   ├── __init__.py
│   │   ├── app.py                 # Flask application
│   │   └── templates/             # HTML templates
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       ├── logging.py             # Logging setup
│       └── gpu.py                 # GPU utilities
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_data.py               # Data loading tests
│   ├── test_preprocessing.py      # Preprocessing tests
│   ├── test_models.py             # Model tests
│   ├── test_inference.py          # Inference pipeline tests
│   ├── test_validation.py         # Input validation tests
│   └── test_web.py                # Web application tests
├── logs/                          # Training and inference logs
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── .env.example                   # Environment variables template
└── setup.py                       # Package setup
```

## Environment Setup

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- cuDNN 8.6+ (for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd crop-disease-identifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify GPU acceleration:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Update `.env` with your settings (optional)

## Usage

### Data Preparation

```bash
python -m src.data.loader --download --prepare
```

### Training

```bash
# Train Stage 1 (Crop Identification)
python -m src.training.trainer --stage 1

# Train Stage 2 (Apple Disease Classification)
python -m src.training.trainer --stage 2
```

### Inference

```bash
# Run inference on a single image
python -m src.inference.pipeline --image path/to/image.jpg
```

### Web Application

```bash
python -m src.web.app
```

Visit `http://localhost:5000` in your browser.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

- [Architecture Documentation](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Evaluation Report](docs/evaluation.md)
- [Limitations and Future Work](docs/limitations.md)

## Requirements

See [Requirements Document](.kiro/specs/crop-disease-identifier/requirements.md)

## Design

See [Design Document](.kiro/specs/crop-disease-identifier/design.md)

## License

[Add license information]

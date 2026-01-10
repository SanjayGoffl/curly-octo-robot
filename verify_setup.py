"""Verification script for project setup."""

import sys
import torch
from pathlib import Path

print("=" * 70)
print("AI Crop Disease Identifier - Setup Verification")
print("=" * 70)

# Check Python version
print(f"\n✓ Python Version: {sys.version}")

# Check PyTorch
print(f"✓ PyTorch Version: {torch.__version__}")

# Check CUDA
cuda_available = torch.cuda.is_available()
print(f"✓ CUDA Available: {cuda_available}")

if cuda_available:
    print(f"  - CUDA Version: {torch.version.cuda}")
    print(f"  - cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"  - Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("  ⚠ WARNING: CUDA not available. CPU-only mode will be used.")

# Check project structure
print("\n✓ Project Structure:")
required_dirs = [
    "src",
    "src/data",
    "src/models",
    "src/training",
    "src/inference",
    "src/web",
    "src/utils",
    "tests",
    "data",
    "models",
    "logs"
]

for dir_name in required_dirs:
    path = Path(dir_name)
    if path.exists():
        print(f"  ✓ {dir_name}/")
    else:
        print(f"  ✗ {dir_name}/ (missing)")

# Check required files
print("\n✓ Required Files:")
required_files = [
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "README.md",
    "setup.py",
    "src/__init__.py",
    "src/utils/gpu.py",
    "src/utils/config.py",
    "src/utils/logging.py",
    "src/data/preprocessing.py",
    "src/data/loader.py",
    "src/models/stage1.py",
    "src/models/stage2.py",
    "src/training/trainer.py",
    "src/training/evaluation.py",
    "src/inference/pipeline.py",
    "src/inference/validation.py",
    "src/inference/recommendations.py",
    "src/web/app.py",
    "src/web/templates/index.html",
]

for file_name in required_files:
    path = Path(file_name)
    if path.exists():
        print(f"  ✓ {file_name}")
    else:
        print(f"  ✗ {file_name} (missing)")

# Test imports
print("\n✓ Testing Imports:")
try:
    from src.utils.gpu import check_gpu_availability, get_device
    print("  ✓ GPU utilities imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import GPU utilities: {e}")

try:
    from src.utils.config import config
    print("  ✓ Configuration imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import configuration: {e}")

try:
    from src.data.preprocessing import get_preprocessing_transforms
    print("  ✓ Preprocessing utilities imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import preprocessing utilities: {e}")

try:
    from src.models.stage1 import Stage1CropIdentifier
    print("  ✓ Stage 1 model imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import Stage 1 model: {e}")

try:
    from src.models.stage2 import Stage2DiseaseClassifier
    print("  ✓ Stage 2 model imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import Stage 2 model: {e}")

try:
    from src.inference.pipeline import InferencePipeline
    print("  ✓ Inference pipeline imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import inference pipeline: {e}")

try:
    from src.inference.recommendations import get_recommendations
    print("  ✓ Recommendations engine imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import recommendations engine: {e}")

print("\n" + "=" * 70)
print("Setup Verification Complete!")
print("=" * 70)

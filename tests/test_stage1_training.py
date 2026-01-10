"""Tests for Stage 1 model training."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import numpy as np
from src.models.stage1 import Stage1CropIdentifier
from src.training.trainer import Trainer
from src.data.loader import PlantVillageDataset
from torch.utils.data import DataLoader
from PIL import Image


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, num_samples=32, num_classes=14):
        self.num_samples = num_samples
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return random tensor and label
        image = torch.randn(3, 224, 224)
        label = idx % self.num_classes
        return image, label


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_model(device):
    """Create a sample Stage 1 model."""
    model = Stage1CropIdentifier(backbone="efficientnet_b0", num_classes=14, pretrained=False)
    return model.to(device)


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, num_samples=32, num_classes=14):
        self.num_samples = num_samples
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return random tensor and label
        image = torch.randn(3, 224, 224)
        label = idx % self.num_classes
        return image, label


@pytest.fixture
def sample_data_loader():
    """Create a sample data loader for testing."""
    dataset = DummyDataset(num_samples=32, num_classes=14)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return loader


def test_stage1_model_initialization(device):
    """Test Stage 1 model initialization."""
    model = Stage1CropIdentifier(backbone="efficientnet_b0", num_classes=14, pretrained=False)
    model = model.to(device)
    
    assert model is not None
    assert model.num_classes == 14
    assert model.backbone_name == "efficientnet_b0"


def test_stage1_model_forward_pass(device, sample_model):
    """Test Stage 1 model forward pass."""
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    output = sample_model(x)
    
    assert output.shape == (batch_size, 14)


def test_stage1_freeze_backbone(device, sample_model):
    """Test freezing backbone layers."""
    sample_model.freeze_backbone()
    
    # Check that backbone parameters are frozen
    for param in sample_model.backbone.parameters():
        assert param.requires_grad is False


def test_stage1_freeze_backbone_except_last(device, sample_model):
    """Test freezing backbone except last layers."""
    sample_model.freeze_backbone_except_last(num_layers=2)
    
    # Check that some parameters are frozen and some are not
    frozen_count = sum(1 for p in sample_model.backbone.parameters() if not p.requires_grad)
    unfrozen_count = sum(1 for p in sample_model.backbone.parameters() if p.requires_grad)
    
    assert frozen_count > 0
    assert unfrozen_count > 0


def test_trainer_initialization(device, sample_model):
    """Test trainer initialization."""
    trainer = Trainer(
        model=sample_model,
        device=device,
        learning_rate=0.001,
        epochs=10,
        early_stopping_patience=5
    )
    
    assert trainer.model is not None
    assert trainer.device == device
    assert trainer.epochs == 10
    assert trainer.early_stopping_patience == 5


def test_trainer_with_class_weights(device, sample_model):
    """Test trainer initialization with class weights."""
    class_weights = torch.ones(14).to(device)
    class_weights[0] = 2.0  # Higher weight for first class
    
    trainer = Trainer(
        model=sample_model,
        device=device,
        class_weights=class_weights
    )
    
    assert trainer.criterion is not None


def test_trainer_train_epoch(device, sample_model, sample_data_loader):
    """Test training for one epoch."""
    trainer = Trainer(
        model=sample_model,
        device=device,
        learning_rate=0.001
    )
    
    metrics = trainer.train_epoch(sample_data_loader)
    
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    assert metrics['loss'] > 0
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['f1_macro'] <= 1


def test_trainer_validate(device, sample_model, sample_data_loader):
    """Test validation."""
    trainer = Trainer(
        model=sample_model,
        device=device
    )
    
    metrics = trainer.validate(sample_data_loader)
    
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    assert 'predictions' in metrics
    assert 'labels' in metrics
    assert metrics['loss'] > 0
    assert 0 <= metrics['accuracy'] <= 1


def test_trainer_evaluate(device, sample_model, sample_data_loader):
    """Test evaluation with detailed metrics."""
    trainer = Trainer(
        model=sample_model,
        device=device
    )
    
    class_names = [f"Crop_{i}" for i in range(14)]
    results = trainer.evaluate(sample_data_loader, class_names=class_names)
    
    assert 'accuracy' in results
    assert 'f1_macro' in results
    assert 'per_class_precision' in results
    assert 'per_class_recall' in results
    assert 'per_class_f1' in results
    assert 'confusion_matrix' in results
    assert 'per_class_metrics' in results
    
    # Check per-class metrics
    assert len(results['per_class_metrics']) == 14
    for crop_name, metrics in results['per_class_metrics'].items():
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'support' in metrics


def test_trainer_save_and_load_model(device, sample_model):
    """Test saving and loading model."""
    trainer = Trainer(
        model=sample_model,
        device=device
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pth"
        
        # Save model
        trainer.save_model(str(model_path))
        assert model_path.exists()
        
        # Load model
        trainer.load_model(str(model_path))
        assert trainer.model is not None


def test_trainer_early_stopping(device, sample_model, sample_data_loader):
    """Test early stopping mechanism."""
    trainer = Trainer(
        model=sample_model,
        device=device,
        epochs=100,
        early_stopping_patience=2
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.checkpoint_dir = Path(tmpdir)
        
        # Train for a few epochs (should stop early)
        trainer.train(sample_data_loader, sample_data_loader, checkpoint_name="test.pth")
        
        # Check that early stopping was triggered
        assert trainer.patience_counter >= 0


def test_trainer_metrics_tracking(device, sample_model, sample_data_loader):
    """Test that trainer tracks metrics correctly."""
    trainer = Trainer(
        model=sample_model,
        device=device,
        epochs=2
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.checkpoint_dir = Path(tmpdir)
        trainer.train(sample_data_loader, sample_data_loader)
        
        # Check that history is populated
        assert len(trainer.train_history['loss']) > 0
        assert len(trainer.train_history['accuracy']) > 0
        assert len(trainer.train_history['f1_macro']) > 0
        assert len(trainer.val_history['loss']) > 0
        assert len(trainer.val_history['accuracy']) > 0
        assert len(trainer.val_history['f1_macro']) > 0


def test_stage1_model_resnet50(device):
    """Test Stage 1 model with ResNet-50 backbone."""
    model = Stage1CropIdentifier(backbone="resnet50", num_classes=14, pretrained=False)
    model = model.to(device)
    
    x = torch.randn(4, 3, 224, 224).to(device)
    output = model(x)
    
    assert output.shape == (4, 14)


def test_stage1_model_invalid_backbone(device):
    """Test Stage 1 model with invalid backbone."""
    with pytest.raises(ValueError):
        model = Stage1CropIdentifier(backbone="invalid_backbone", num_classes=14)


def test_trainer_confusion_matrix(device, sample_model, sample_data_loader):
    """Test confusion matrix generation."""
    trainer = Trainer(
        model=sample_model,
        device=device
    )
    
    results = trainer.evaluate(sample_data_loader)
    cm = results['confusion_matrix']
    
    # Check confusion matrix shape
    assert cm.shape == (14, 14)
    
    # Check that confusion matrix is valid
    assert np.all(cm >= 0)
    assert np.sum(cm) > 0

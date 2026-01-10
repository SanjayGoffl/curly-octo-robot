"""Tests for Stage 2 model training."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import numpy as np
from src.models.stage2 import Stage2DiseaseClassifier
from src.training.trainer import Trainer
from src.training.stage2_trainer import (
    filter_apple_images,
    create_oversampled_dataset,
    create_weighted_sampler,
    Stage2Dataset
)
from torch.utils.data import DataLoader
from PIL import Image


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, num_samples=32, num_classes=4):
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
    """Create a sample Stage 2 model."""
    model = Stage2DiseaseClassifier(backbone="efficientnet_b0", num_classes=4, pretrained=False)
    return model.to(device)


@pytest.fixture
def sample_data_loader():
    """Create a sample data loader for testing."""
    dataset = DummyDataset(num_samples=32, num_classes=4)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return loader


def test_stage2_model_initialization(device):
    """Test Stage 2 model initialization."""
    model = Stage2DiseaseClassifier(backbone="efficientnet_b0", num_classes=4, pretrained=False)
    model = model.to(device)
    
    assert model is not None
    assert model.num_classes == 4
    assert model.backbone_name == "efficientnet_b0"


def test_stage2_model_forward_pass(device, sample_model):
    """Test Stage 2 model forward pass."""
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    output = sample_model(x)
    
    assert output.shape == (batch_size, 4)


def test_stage2_freeze_backbone(device, sample_model):
    """Test freezing backbone layers."""
    sample_model.freeze_backbone()
    
    # Check that backbone parameters are frozen
    for param in sample_model.backbone.parameters():
        assert param.requires_grad is False


def test_stage2_freeze_backbone_except_last(device, sample_model):
    """Test freezing backbone except last layers."""
    sample_model.freeze_backbone_except_last(num_layers=2)
    
    # Check that some parameters are frozen and some are not
    frozen_count = sum(1 for p in sample_model.backbone.parameters() if not p.requires_grad)
    unfrozen_count = sum(1 for p in sample_model.backbone.parameters() if p.requires_grad)
    
    assert frozen_count > 0
    assert unfrozen_count > 0


def test_filter_apple_images():
    """Test filtering apple images from mixed dataset."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    labels = ["Apple Scab", "Tomato Blight", "Apple Healthy", "Potato Late Blight"]
    
    apple_paths, apple_diseases = filter_apple_images(image_paths, labels)
    
    assert len(apple_paths) == 2
    assert len(apple_diseases) == 2
    assert "Scab" in apple_diseases
    assert "Healthy" in apple_diseases


def test_create_oversampled_dataset():
    """Test oversampling minority classes."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]
    labels = ["Healthy", "Healthy", "Healthy", "Scab", "Cedar Rust"]
    
    oversampled_paths, oversampled_labels = create_oversampled_dataset(
        image_paths, labels, minority_classes={"Scab", "Cedar Rust"}
    )
    
    # Check that oversampling occurred
    assert len(oversampled_paths) >= len(image_paths)
    assert len(oversampled_labels) >= len(labels)
    
    # Check that minority classes are represented more
    from collections import Counter
    original_counts = Counter(labels)
    oversampled_counts = Counter(oversampled_labels)
    
    assert oversampled_counts["Scab"] >= original_counts["Scab"]
    assert oversampled_counts["Cedar Rust"] >= original_counts["Cedar Rust"]


def test_create_weighted_sampler():
    """Test weighted sampler creation."""
    labels = ["Healthy", "Healthy", "Healthy", "Scab", "Cedar Rust"]
    
    sampler = create_weighted_sampler(labels)
    
    assert sampler is not None
    assert len(sampler) == len(labels)


def test_trainer_with_class_weights(device, sample_model):
    """Test trainer initialization with class weights."""
    class_weights = torch.ones(4).to(device)
    class_weights[0] = 2.0  # Higher weight for first class
    
    trainer = Trainer(
        model=sample_model,
        device=device,
        class_weights=class_weights
    )
    
    assert trainer.criterion is not None


def test_trainer_validate_with_tracked_recall(device, sample_model, sample_data_loader):
    """Test validation with tracked class recall."""
    trainer = Trainer(
        model=sample_model,
        device=device
    )
    
    metrics = trainer.validate(sample_data_loader, track_class_recall=2)
    
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    assert 'tracked_class_recall' in metrics
    assert 0 <= metrics['tracked_class_recall'] <= 1


def test_trainer_train_with_tracked_recall(device, sample_model, sample_data_loader):
    """Test training with tracked class recall."""
    trainer = Trainer(
        model=sample_model,
        device=device,
        epochs=2
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.checkpoint_dir = Path(tmpdir)
        trainer.train(sample_data_loader, sample_data_loader, track_class_recall=2)
        
        # Check that training completed
        assert len(trainer.train_history['loss']) > 0


def test_trainer_evaluate_stage2(device, sample_model, sample_data_loader):
    """Test evaluation with detailed metrics for Stage 2."""
    trainer = Trainer(
        model=sample_model,
        device=device
    )
    
    class_names = ["Healthy", "Scab", "Black Rot", "Cedar Rust"]
    results = trainer.evaluate(sample_data_loader, class_names=class_names)
    
    assert 'accuracy' in results
    assert 'f1_macro' in results
    assert 'per_class_precision' in results
    assert 'per_class_recall' in results
    assert 'per_class_f1' in results
    assert 'confusion_matrix' in results
    assert 'per_class_metrics' in results
    
    # Check per-class metrics
    assert len(results['per_class_metrics']) == 4
    for disease_name, metrics in results['per_class_metrics'].items():
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'support' in metrics


def test_stage2_model_resnet50(device):
    """Test Stage 2 model with ResNet-50 backbone."""
    model = Stage2DiseaseClassifier(backbone="resnet50", num_classes=4, pretrained=False)
    model = model.to(device)
    
    x = torch.randn(4, 3, 224, 224).to(device)
    output = model(x)
    
    assert output.shape == (4, 4)


def test_stage2_model_invalid_backbone(device):
    """Test Stage 2 model with invalid backbone."""
    with pytest.raises(ValueError):
        model = Stage2DiseaseClassifier(backbone="invalid_backbone", num_classes=4)


def test_trainer_confusion_matrix_stage2(device, sample_model, sample_data_loader):
    """Test confusion matrix generation for Stage 2."""
    trainer = Trainer(
        model=sample_model,
        device=device
    )
    
    results = trainer.evaluate(sample_data_loader)
    cm = results['confusion_matrix']
    
    # Check confusion matrix shape (4x4 for 4 disease classes)
    assert cm.shape == (4, 4)
    
    # Check that confusion matrix is valid
    assert np.all(cm >= 0)
    assert np.sum(cm) > 0


def test_stage2_dataset_with_augmentation():
    """Test Stage 2 dataset with augmentation strategy."""
    from src.data.augmentation_manager import AugmentationStrategy
    from src.data.preprocessing import get_preprocessing_transforms
    
    # Create dummy image paths and labels
    image_paths = ["dummy1.jpg", "dummy2.jpg", "dummy3.jpg"]
    labels = ["Healthy", "Scab", "Cedar Rust"]
    
    # Create augmentation strategy
    augmentation_strategy = AugmentationStrategy(labels, image_size=224)
    
    # Create dataset
    dataset = Stage2Dataset(
        image_paths, labels,
        augmentation_strategy=augmentation_strategy,
        transform=get_preprocessing_transforms()
    )
    
    assert len(dataset) == 3
    assert dataset.augmentation_strategy is not None

"""Tests for image preprocessing."""

import pytest
import torch
from PIL import Image
import tempfile
import os
from src.data.preprocessing import (
    get_preprocessing_transforms,
    get_augmentation_transforms,
    preprocess_image,
    validate_image_format,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from src.data.augmentation_manager import (
    get_class_frequencies,
    identify_minority_classes,
    calculate_class_weights,
    get_sample_weights,
    AugmentationStrategy
)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (256, 256), color='red')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


def test_preprocessing_transforms_output_shape():
    """Test that preprocessing transforms produce correct output shape."""
    transforms = get_preprocessing_transforms(image_size=224)
    
    # Create a dummy image
    img = Image.new('RGB', (256, 256), color='blue')
    output = transforms(img)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)


def test_augmentation_transforms_output_shape():
    """Test that augmentation transforms produce correct output shape."""
    transforms = get_augmentation_transforms(image_size=224, strong=False)
    
    img = Image.new('RGB', (256, 256), color='green')
    output = transforms(img)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)


def test_strong_augmentation_transforms():
    """Test that strong augmentation transforms work."""
    transforms = get_augmentation_transforms(image_size=224, strong=True)
    
    img = Image.new('RGB', (256, 256), color='yellow')
    output = transforms(img)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)


def test_preprocess_image(sample_image):
    """Test preprocessing a single image."""
    output = preprocess_image(sample_image, image_size=224)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)


def test_validate_image_format_valid_jpg(sample_image):
    """Test validation of valid JPG image."""
    is_valid, error = validate_image_format(sample_image)
    assert is_valid is True
    assert error is None


def test_validate_image_format_invalid_file():
    """Test validation of invalid file."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"This is not an image")
        f.flush()
        
        is_valid, error = validate_image_format(f.name)
        assert is_valid is False
        assert error is not None
    
    os.unlink(f.name)


def test_validate_image_format_nonexistent_file():
    """Test validation of nonexistent file."""
    is_valid, error = validate_image_format('/nonexistent/path/image.jpg')
    assert is_valid is False
    assert error is not None


def test_get_class_frequencies():
    """Test class frequency calculation."""
    labels = ['Apple', 'Apple', 'Tomato', 'Tomato', 'Tomato', 'Pepper']
    frequencies = get_class_frequencies(labels)
    
    assert frequencies['Apple'] == 2
    assert frequencies['Tomato'] == 3
    assert frequencies['Pepper'] == 1


def test_identify_minority_classes():
    """Test minority class identification."""
    labels = ['A'] * 100 + ['B'] * 50 + ['C'] * 10 + ['D'] * 5
    minority = identify_minority_classes(labels, threshold_percentile=25)
    
    # Classes C and D should be minority
    assert 'C' in minority or 'D' in minority


def test_calculate_class_weights():
    """Test class weight calculation."""
    labels = ['Apple', 'Apple', 'Tomato', 'Tomato', 'Tomato', 'Pepper']
    weights = calculate_class_weights(labels)
    
    # Inverse frequency weighting: minority classes get higher weights
    assert weights['Apple'] > weights['Tomato']
    assert weights['Pepper'] > weights['Tomato']  # Pepper is minority


def test_get_sample_weights():
    """Test per-sample weight calculation."""
    labels = ['A', 'A', 'B', 'B', 'B']
    weights = get_sample_weights(labels)
    
    assert len(weights) == 5
    assert weights[0] == weights[1]  # Same class
    assert weights[2] == weights[3] == weights[4]  # Same class
    assert weights[0] > weights[2]  # A is minority


def test_augmentation_strategy():
    """Test augmentation strategy."""
    labels = ['Apple'] * 100 + ['Scab'] * 10 + ['Rust'] * 5
    strategy = AugmentationStrategy(labels)
    
    # Scab and Rust should be minority
    assert strategy.is_minority('Scab') or strategy.is_minority('Rust')
    
    # Get augmentation for each class
    aug_apple = strategy.get_augmentation('Apple')
    aug_scab = strategy.get_augmentation('Scab')
    
    assert aug_apple is not None
    assert aug_scab is not None


def test_augmentation_strategy_summary():
    """Test augmentation strategy summary."""
    labels = ['A'] * 50 + ['B'] * 30 + ['C'] * 10
    strategy = AugmentationStrategy(labels)
    
    summary = strategy.get_summary()
    
    assert 'total_classes' in summary
    assert 'minority_classes' in summary
    assert 'class_frequencies' in summary
    assert 'class_weights' in summary
    assert summary['total_classes'] == 3

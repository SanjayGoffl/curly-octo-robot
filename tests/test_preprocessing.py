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

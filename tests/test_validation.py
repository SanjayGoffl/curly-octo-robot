"""Tests for input validation."""

import pytest
import tempfile
import os
from PIL import Image
from src.inference.validation import (
    validate_file_type,
    validate_file_size,
    validate_file_integrity,
    validate_input_file
)


@pytest.fixture
def sample_jpg():
    """Create a sample JPG image."""
    img = Image.new('RGB', (100, 100), color='red')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_png():
    """Create a sample PNG image."""
    img = Image.new('RGB', (100, 100), color='blue')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_txt():
    """Create a sample text file."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"This is not an image")
        f.flush()
        yield f.name
    os.unlink(f.name)


def test_validate_file_type_jpg(sample_jpg):
    """Test validation of JPG file type."""
    is_valid, error = validate_file_type(sample_jpg)
    assert is_valid is True
    assert error is None


def test_validate_file_type_png(sample_png):
    """Test validation of PNG file type."""
    is_valid, error = validate_file_type(sample_png)
    assert is_valid is True
    assert error is None


def test_validate_file_type_invalid(sample_txt):
    """Test validation of invalid file type."""
    is_valid, error = validate_file_type(sample_txt)
    assert is_valid is False
    assert "Invalid file format" in error


def test_validate_file_size_valid(sample_jpg):
    """Test validation of valid file size."""
    is_valid, error = validate_file_size(sample_jpg, max_size_bytes=10485760)
    assert is_valid is True
    assert error is None


def test_validate_file_size_too_large(sample_jpg):
    """Test validation of file that exceeds size limit."""
    is_valid, error = validate_file_size(sample_jpg, max_size_bytes=1)
    assert is_valid is False
    assert "File too large" in error


def test_validate_file_integrity_valid(sample_jpg):
    """Test validation of file integrity."""
    is_valid, error = validate_file_integrity(sample_jpg)
    assert is_valid is True
    assert error is None


def test_validate_file_integrity_corrupted():
    """Test validation of corrupted file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(b"This is not a valid JPEG")
        f.flush()
        
        is_valid, error = validate_file_integrity(f.name)
        assert is_valid is False
        assert error is not None
    
    os.unlink(f.name)


def test_validate_input_file_valid(sample_jpg):
    """Test complete input file validation."""
    is_valid, error = validate_input_file(sample_jpg)
    assert is_valid is True
    assert error is None


def test_validate_input_file_invalid_type(sample_txt):
    """Test input file validation with invalid type."""
    is_valid, error = validate_input_file(sample_txt)
    assert is_valid is False
    assert error is not None

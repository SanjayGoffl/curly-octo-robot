"""Input validation gates."""

import os
from pathlib import Path


def validate_file_type(file_path, allowed_types=None):
    """
    Validate file type.
    
    Args:
        file_path (str): Path to file
        allowed_types (list): Allowed file extensions (default: jpg, png)
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if allowed_types is None:
        allowed_types = [".jpg", ".jpeg", ".png"]
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in allowed_types:
        return False, f"Invalid file format. Please upload JPG or PNG."
    
    return True, None


def validate_file_size(file_path, max_size_bytes=10485760):
    """
    Validate file size.
    
    Args:
        file_path (str): Path to file
        max_size_bytes (int): Maximum file size in bytes (default: 10 MB)
        
    Returns:
        tuple: (is_valid, error_message)
    """
    file_size = os.path.getsize(file_path)
    
    if file_size > max_size_bytes:
        max_size_mb = max_size_bytes / (1024 * 1024)
        return False, f"File too large. Maximum {max_size_mb:.1f} MB."
    
    return True, None


def validate_file_integrity(file_path):
    """
    Validate file integrity.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        from PIL import Image
        image = Image.open(file_path)
        image.verify()
        return True, None
    except Exception as e:
        return False, f"File corrupted. Please try again."


def validate_input_file(file_path, max_size_bytes=10485760):
    """
    Validate input file (type, size, integrity).
    
    Args:
        file_path (str): Path to file
        max_size_bytes (int): Maximum file size in bytes
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check file type
    is_valid, error = validate_file_type(file_path)
    if not is_valid:
        return False, error
    
    # Check file size
    is_valid, error = validate_file_size(file_path, max_size_bytes)
    if not is_valid:
        return False, error
    
    # Check file integrity
    is_valid, error = validate_file_integrity(file_path)
    if not is_valid:
        return False, error
    
    return True, None

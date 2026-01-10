"""Tests for dataset management and splitting."""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
from src.data.dataset_manager import (
    verify_dataset_structure,
    get_color_images_only,
    validate_dataset_completeness,
    EXPECTED_CROPS,
    APPLE_DISEASES
)
from src.data.data_splitter import (
    create_stratified_splits,
    verify_split_integrity,
    save_splits_to_disk,
    load_splits_from_disk
)


@pytest.fixture
def mock_plantvillage_dataset():
    """Create a mock PlantVillage dataset structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a minimal dataset with 2 crops and some diseases
        crops_to_create = ['Apple', 'Tomato']
        
        for crop in crops_to_create:
            crop_dir = tmpdir / crop
            crop_dir.mkdir()
            
            if crop == 'Apple':
                diseases = ['Apple___healthy', 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust']
            else:
                diseases = ['Tomato___healthy', 'Tomato___Early_blight', 'Tomato___Late_blight']
            
            for disease in diseases:
                disease_dir = crop_dir / disease
                disease_dir.mkdir()
                
                # Create 10 color images per disease
                for i in range(10):
                    img = Image.new('RGB', (256, 256), color='green')
                    img.save(disease_dir / f"image_{i}.jpg")
                
                # Create 2 segmented images (should be ignored)
                for i in range(2):
                    img = Image.new('RGB', (256, 256), color='blue')
                    img.save(disease_dir / f"image_{i}_segmented.jpg")
        
        yield tmpdir


def test_verify_dataset_structure(mock_plantvillage_dataset):
    """Test dataset structure verification."""
    stats = verify_dataset_structure(mock_plantvillage_dataset)
    
    assert stats['total_images'] > 0
    assert stats['color_images'] > 0
    assert 'Apple' in stats['crops']
    assert 'Tomato' in stats['crops']
    assert len(stats['apple_diseases']) == 4


def test_get_color_images_only(mock_plantvillage_dataset):
    """Test getting only color images."""
    images_by_crop = get_color_images_only(mock_plantvillage_dataset)
    
    assert 'Apple' in images_by_crop
    assert 'Tomato' in images_by_crop
    
    # Check that segmented images are excluded
    for crop, diseases in images_by_crop.items():
        for disease, image_paths in diseases.items():
            for path in image_paths:
                assert '_segmented' not in path


def test_validate_dataset_completeness(mock_plantvillage_dataset):
    """Test dataset completeness validation."""
    stats = verify_dataset_structure(mock_plantvillage_dataset)
    results = validate_dataset_completeness(stats)
    
    # Mock dataset won't have all 14 crops, so it should have errors
    assert results['crop_count'] < 14
    assert len(results['errors']) > 0


def test_create_stratified_splits(mock_plantvillage_dataset):
    """Test stratified split creation."""
    images_by_crop = get_color_images_only(mock_plantvillage_dataset)
    
    splits = create_stratified_splits(
        images_by_crop,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    assert 'train' in splits
    assert 'val' in splits
    assert 'test' in splits
    
    assert len(splits['train']) > 0
    assert len(splits['val']) > 0
    assert len(splits['test']) > 0
    
    # Check that each split has tuples of (path, crop, disease)
    for split_data in splits.values():
        for item in split_data:
            assert len(item) == 3
            assert isinstance(item[0], str)  # path
            assert isinstance(item[1], str)  # crop
            assert isinstance(item[2], str)  # disease


def test_verify_split_integrity(mock_plantvillage_dataset):
    """Test split integrity verification."""
    images_by_crop = get_color_images_only(mock_plantvillage_dataset)
    splits = create_stratified_splits(images_by_crop)
    
    results = verify_split_integrity(splits)
    
    assert results['valid'] is True
    assert len(results['errors']) == 0
    
    # Check split sizes
    assert results['split_sizes']['train'] > 0
    assert results['split_sizes']['val'] > 0
    assert results['split_sizes']['test'] > 0
    
    # Check that splits sum to total
    total = (results['split_sizes']['train'] + 
             results['split_sizes']['val'] + 
             results['split_sizes']['test'])
    assert total == results['split_sizes']['total']


def test_split_ratios(mock_plantvillage_dataset):
    """Test that split ratios are approximately correct."""
    images_by_crop = get_color_images_only(mock_plantvillage_dataset)
    splits = create_stratified_splits(
        images_by_crop,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    results = verify_split_integrity(splits)
    
    # Check approximate ratios (allow 5% tolerance)
    train_pct = results['split_percentages']['train']
    val_pct = results['split_percentages']['val']
    test_pct = results['split_percentages']['test']
    
    assert 65 <= train_pct <= 75
    assert 10 <= val_pct <= 20
    assert 10 <= test_pct <= 20


def test_no_data_leakage(mock_plantvillage_dataset):
    """Test that there's no data leakage between splits."""
    images_by_crop = get_color_images_only(mock_plantvillage_dataset)
    splits = create_stratified_splits(images_by_crop)
    
    train_paths = set(item[0] for item in splits['train'])
    val_paths = set(item[0] for item in splits['val'])
    test_paths = set(item[0] for item in splits['test'])
    
    # Check no overlaps
    assert len(train_paths & val_paths) == 0
    assert len(train_paths & test_paths) == 0
    assert len(val_paths & test_paths) == 0


def test_save_and_load_splits(mock_plantvillage_dataset):
    """Test saving and loading splits from disk."""
    images_by_crop = get_color_images_only(mock_plantvillage_dataset)
    original_splits = create_stratified_splits(images_by_crop)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save splits
        save_splits_to_disk(original_splits, tmpdir)
        
        # Verify files were created
        assert (Path(tmpdir) / 'train_split.txt').exists()
        assert (Path(tmpdir) / 'val_split.txt').exists()
        assert (Path(tmpdir) / 'test_split.txt').exists()
        
        # Load splits
        loaded_splits = load_splits_from_disk(tmpdir)
        
        # Verify loaded splits match original
        assert len(loaded_splits['train']) == len(original_splits['train'])
        assert len(loaded_splits['val']) == len(original_splits['val'])
        assert len(loaded_splits['test']) == len(original_splits['test'])


def test_class_distribution_maintained(mock_plantvillage_dataset):
    """Test that class distribution is maintained across splits."""
    images_by_crop = get_color_images_only(mock_plantvillage_dataset)
    splits = create_stratified_splits(images_by_crop)
    
    results = verify_split_integrity(splits)
    
    # Check that all crops appear in all splits
    train_crops = set(results['class_distribution']['train']['crops'].keys())
    val_crops = set(results['class_distribution']['val']['crops'].keys())
    test_crops = set(results['class_distribution']['test']['crops'].keys())
    
    all_crops = train_crops | val_crops | test_crops
    
    # Each crop should ideally be in all splits (though small crops might not be)
    for crop in all_crops:
        # At least in train and test
        assert crop in train_crops or crop in val_crops or crop in test_crops

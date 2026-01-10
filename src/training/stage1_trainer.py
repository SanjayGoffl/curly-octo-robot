"""Stage 1 model training script."""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from src.models.stage1 import Stage1CropIdentifier
from src.training.trainer import Trainer
from src.data.loader import load_plantvillage_dataset, create_data_loaders
from src.data.augmentation_manager import calculate_class_weights, AugmentationStrategy
from src.data.dataset_manager import verify_dataset_structure, validate_dataset_completeness


def prepare_stage1_training(
    dataset_root,
    batch_size=32,
    num_workers=4,
    backbone="efficientnet_b0",
    learning_rate=0.001,
    epochs=50,
    early_stopping_patience=10,
    checkpoint_dir="./models/stage1",
    device=None
):
    """
    Prepare Stage 1 training with data loading and model initialization.
    
    Args:
        dataset_root (str): Root path to PlantVillage dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        backbone (str): Backbone architecture (efficientnet_b0 or resnet50)
        learning_rate (float): Learning rate
        epochs (int): Number of epochs
        early_stopping_patience (int): Early stopping patience
        checkpoint_dir (str): Directory to save checkpoints
        device (torch.device): Device to use (auto-detect if None)
        
    Returns:
        dict: Training configuration with model, loaders, trainer, and metadata
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Verify dataset structure
    print("\nVerifying dataset structure...")
    stats = verify_dataset_structure(dataset_root)
    validation = validate_dataset_completeness(stats)
    
    if not validation['valid']:
        print("Dataset validation errors:")
        for error in validation['errors']:
            print(f"  ERROR: {error}")
        raise ValueError("Dataset validation failed")
    
    print(f"Dataset verified: {stats['total_images']} color images")
    print(f"Crops found: {validation['crop_count']}")
    print(f"Apple diseases found: {validation['apple_disease_count']}")
    
    # Load dataset with stratified splitting
    print("\nLoading dataset with stratified splitting...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        load_plantvillage_dataset(dataset_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    print(f"Train set: {len(train_paths)} images")
    print(f"Val set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    
    # Create label to index mapping
    unique_crops = sorted(set(train_labels))
    crop_to_idx = {crop: idx for idx, crop in enumerate(unique_crops)}
    idx_to_crop = {idx: crop for crop, idx in crop_to_idx.items()}
    
    print(f"\nCrop classes ({len(unique_crops)}):")
    for crop, idx in crop_to_idx.items():
        print(f"  {idx}: {crop}")
    
    # Convert labels to indices
    train_labels_idx = [crop_to_idx[crop] for crop in train_labels]
    val_labels_idx = [crop_to_idx[crop] for crop in val_labels]
    test_labels_idx = [crop_to_idx[crop] for crop in test_labels]
    
    # Calculate class weights for imbalance handling
    print("\nCalculating class weights for imbalance handling...")
    class_weights_dict = calculate_class_weights(train_labels_idx)
    
    # Convert to tensor for loss function
    class_weights_tensor = torch.zeros(len(unique_crops))
    for idx, weight in class_weights_dict.items():
        class_weights_tensor[idx] = weight
    class_weights_tensor = class_weights_tensor.to(device)
    
    print("Class weights (inverse frequency):")
    for crop, idx in crop_to_idx.items():
        print(f"  {crop}: {class_weights_tensor[idx]:.4f}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_paths, train_labels_idx,
        val_paths, val_labels_idx,
        test_paths, test_labels_idx,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=True
    )
    
    # Initialize model
    print(f"\nInitializing Stage 1 model with {backbone} backbone...")
    model = Stage1CropIdentifier(
        backbone=backbone,
        num_classes=len(unique_crops),
        pretrained=True
    )
    
    # Freeze backbone layers for transfer learning
    model.freeze_backbone_except_last(num_layers=2)
    print("Backbone layers frozen (except last 2 layers)")
    
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize trainer
    print(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
        class_weights=class_weights_tensor
    )
    
    return {
        'model': model,
        'trainer': trainer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'device': device,
        'crop_to_idx': crop_to_idx,
        'idx_to_crop': idx_to_crop,
        'class_weights': class_weights_dict,
        'stats': stats,
        'checkpoint_dir': checkpoint_dir
    }


def train_stage1(config):
    """
    Train Stage 1 model.
    
    Args:
        config (dict): Training configuration from prepare_stage1_training
        
    Returns:
        dict: Training results including metrics and model path
    """
    print("\n" + "="*80)
    print("STAGE 1 TRAINING")
    print("="*80)
    
    trainer = config['trainer']
    train_loader = config['train_loader']
    val_loader = config['val_loader']
    checkpoint_dir = config['checkpoint_dir']
    
    # Train model
    trainer.train(train_loader, val_loader, checkpoint_name="stage1_best.pth")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("STAGE 1 EVALUATION")
    print("="*80)
    
    test_loader = config['test_loader']
    idx_to_crop = config['idx_to_crop']
    class_names = [idx_to_crop[i] for i in range(len(idx_to_crop))]
    
    eval_results = trainer.evaluate(test_loader, class_names=class_names)
    
    # Print evaluation results
    print(f"\nTest Set Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Test Set Macro F1-Score: {eval_results['f1_macro']:.4f}")
    
    print("\nPer-Class Metrics:")
    print(f"{'Crop':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 66)
    
    for crop, metrics in eval_results['per_class_metrics'].items():
        print(f"{crop:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} {int(metrics['support']):<10}")
    
    # Save results
    results = {
        'accuracy': eval_results['accuracy'],
        'f1_macro': eval_results['f1_macro'],
        'per_class_metrics': eval_results['per_class_metrics'],
        'confusion_matrix': eval_results['confusion_matrix'],
        'training_history': {
            'train': trainer.train_history,
            'val': trainer.val_history
        },
        'model_path': str(Path(checkpoint_dir) / "stage1_best.pth"),
        'class_names': class_names
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    dataset_root = "./data/raw/plantvillage"
    
    # Prepare training
    config = prepare_stage1_training(
        dataset_root=dataset_root,
        batch_size=32,
        num_workers=4,
        backbone="efficientnet_b0",
        learning_rate=0.001,
        epochs=50,
        early_stopping_patience=10,
        checkpoint_dir="./models/stage1"
    )
    
    # Train model
    results = train_stage1(config)
    
    print("\nTraining complete!")
    print(f"Model saved to: {results['model_path']}")

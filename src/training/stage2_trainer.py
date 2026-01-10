"""Stage 2 model training script with class imbalance handling."""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models.stage2 import Stage2DiseaseClassifier
from src.training.trainer import Trainer
from src.data.loader import load_plantvillage_dataset, create_data_loaders, PlantVillageDataset
from src.data.augmentation_manager import calculate_class_weights, AugmentationStrategy
from src.data.dataset_manager import verify_dataset_structure, validate_dataset_completeness
from src.data.preprocessing import get_preprocessing_transforms, get_augmentation_transforms


def filter_apple_images(image_paths, labels):
    """
    Filter dataset to only include Apple images.
    
    Args:
        image_paths (list): List of image file paths
        labels (list): List of crop labels
        
    Returns:
        tuple: (filtered_paths, filtered_labels, disease_labels)
    """
    apple_paths = []
    apple_disease_labels = []
    
    for path, label in zip(image_paths, labels):
        if label.startswith("Apple"):
            apple_paths.append(path)
            # Extract disease from label (e.g., "Apple Scab" -> "Scab")
            disease = label.replace("Apple ", "").replace("Apple", "Healthy")
            apple_disease_labels.append(disease)
    
    return apple_paths, apple_disease_labels


class Stage2Dataset(torch.utils.data.Dataset):
    """Stage 2 dataset with per-sample augmentation strategy."""
    
    def __init__(self, image_paths, labels, augmentation_strategy=None, transform=None):
        """
        Initialize Stage 2 dataset.
        
        Args:
            image_paths (list): List of image file paths
            labels (list): List of disease labels
            augmentation_strategy (AugmentationStrategy): Strategy for per-class augmentation
            transform (callable): Optional base transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.augmentation_strategy = augmentation_strategy
        self.base_transform = transform or get_preprocessing_transforms()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply augmentation strategy if available
        if self.augmentation_strategy:
            augmentation = self.augmentation_strategy.get_augmentation(label)
            image = augmentation(image)
        else:
            image = self.base_transform(image)
        
        return image, label


def create_weighted_sampler(labels, device=None):
    """
    Create weighted random sampler for class imbalance handling.
    
    Args:
        labels (list): List of class labels
        device (torch.device): Device (for compatibility, not used)
        
    Returns:
        WeightedRandomSampler: Sampler for balanced batch sampling
    """
    class_weights = calculate_class_weights(labels)
    
    # Create per-sample weights
    sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.float)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    
    return sampler


def create_oversampled_dataset(image_paths, labels, minority_classes=None):
    """
    Create oversampled dataset by duplicating minority class samples.
    
    Args:
        image_paths (list): List of image file paths
        labels (list): List of disease labels
        minority_classes (set): Set of minority class labels to oversample
        
    Returns:
        tuple: (oversampled_paths, oversampled_labels)
    """
    if minority_classes is None:
        # Identify minority classes (bottom 25% by frequency)
        frequencies = Counter(labels)
        sorted_freqs = sorted(frequencies.values())
        threshold_idx = max(0, int(len(sorted_freqs) * 0.25))
        threshold = sorted_freqs[threshold_idx]
        minority_classes = {label for label, freq in frequencies.items() if freq <= threshold}
    
    # Find max frequency
    frequencies = Counter(labels)
    max_freq = max(frequencies.values())
    
    # Oversample minority classes to match max frequency
    oversampled_paths = list(image_paths)
    oversampled_labels = list(labels)
    
    for class_label in minority_classes:
        class_indices = [i for i, label in enumerate(labels) if label == class_label]
        current_freq = len(class_indices)
        
        if current_freq < max_freq:
            # Calculate how many samples to add
            samples_to_add = max_freq - current_freq
            
            # Randomly select indices to duplicate
            np.random.seed(42)
            indices_to_duplicate = np.random.choice(class_indices, size=samples_to_add, replace=True)
            
            # Add duplicated samples
            for idx in indices_to_duplicate:
                oversampled_paths.append(image_paths[idx])
                oversampled_labels.append(labels[idx])
    
    return oversampled_paths, oversampled_labels


def prepare_stage2_training(
    dataset_root,
    batch_size=32,
    num_workers=4,
    backbone="efficientnet_b0",
    learning_rate=0.001,
    epochs=50,
    early_stopping_patience=10,
    checkpoint_dir="./models/stage2",
    device=None,
    use_oversampling=True,
    use_weighted_loss=True
):
    """
    Prepare Stage 2 training with data loading and model initialization.
    
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
        use_oversampling (bool): Whether to use oversampling for minority classes
        use_weighted_loss (bool): Whether to use weighted loss
        
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
    
    # Load dataset with stratified splitting
    print("\nLoading dataset with stratified splitting...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        load_plantvillage_dataset(dataset_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Filter to Apple images only
    print("\nFiltering to Apple images only...")
    train_paths, train_disease_labels = filter_apple_images(train_paths, train_labels)
    val_paths, val_disease_labels = filter_apple_images(val_paths, val_labels)
    test_paths, test_disease_labels = filter_apple_images(test_paths, test_labels)
    
    print(f"Train set: {len(train_paths)} apple images")
    print(f"Val set: {len(val_paths)} apple images")
    print(f"Test set: {len(test_paths)} apple images")
    
    # Create label to index mapping
    disease_classes = ["Healthy", "Scab", "Black Rot", "Cedar Rust"]
    disease_to_idx = {disease: idx for idx, disease in enumerate(disease_classes)}
    idx_to_disease = {idx: disease for disease, idx in disease_to_idx.items()}
    
    print(f"\nDisease classes ({len(disease_classes)}):")
    for disease, idx in disease_to_idx.items():
        print(f"  {idx}: {disease}")
    
    # Convert labels to indices
    train_labels_idx = [disease_to_idx[disease] for disease in train_disease_labels]
    val_labels_idx = [disease_to_idx[disease] for disease in val_disease_labels]
    test_labels_idx = [disease_to_idx[disease] for disease in test_disease_labels]
    
    # Print class distribution
    print("\nClass distribution (before imbalance handling):")
    train_freq = Counter(train_disease_labels)
    for disease in disease_classes:
        count = train_freq.get(disease, 0)
        pct = 100 * count / len(train_disease_labels) if train_disease_labels else 0
        print(f"  {disease}: {count} ({pct:.1f}%)")
    
    # Identify minority classes (bottom 25% by frequency)
    frequencies = Counter(train_disease_labels)
    sorted_freqs = sorted(frequencies.values())
    threshold_idx = max(0, int(len(sorted_freqs) * 0.25))
    threshold = sorted_freqs[threshold_idx]
    minority_classes = {label for label, freq in frequencies.items() if freq <= threshold}
    
    print(f"\nMinority classes: {minority_classes}")
    
    # Apply oversampling if enabled
    if use_oversampling:
        print("\nApplying oversampling to minority classes...")
        train_paths, train_disease_labels = create_oversampled_dataset(
            train_paths, train_disease_labels, minority_classes
        )
        train_labels_idx = [disease_to_idx[disease] for disease in train_disease_labels]
        
        print(f"Train set after oversampling: {len(train_paths)} apple images")
        
        # Print class distribution after oversampling
        print("\nClass distribution (after oversampling):")
        train_freq = Counter(train_disease_labels)
        for disease in disease_classes:
            count = train_freq.get(disease, 0)
            pct = 100 * count / len(train_disease_labels) if train_disease_labels else 0
            print(f"  {disease}: {count} ({pct:.1f}%)")
    
    # Calculate class weights for weighted loss
    print("\nCalculating class weights for weighted loss...")
    class_weights_dict = calculate_class_weights(train_labels_idx)
    
    # Convert to tensor for loss function
    class_weights_tensor = torch.zeros(len(disease_classes))
    for idx, weight in class_weights_dict.items():
        class_weights_tensor[idx] = weight
    class_weights_tensor = class_weights_tensor.to(device)
    
    print("Class weights (inverse frequency):")
    for disease, idx in disease_to_idx.items():
        print(f"  {disease}: {class_weights_tensor[idx]:.4f}")
    
    # Create augmentation strategy for stronger augmentation on minority classes
    print("\nCreating augmentation strategy...")
    augmentation_strategy = AugmentationStrategy(
        train_disease_labels,
        image_size=224,
        threshold_percentile=25
    )
    
    # Create datasets with augmentation strategy
    print("\nCreating datasets with per-class augmentation...")
    train_dataset = Stage2Dataset(
        train_paths, train_disease_labels,
        augmentation_strategy=augmentation_strategy
    )
    
    val_dataset = Stage2Dataset(
        val_paths, val_disease_labels,
        transform=get_preprocessing_transforms()
    )
    
    test_dataset = Stage2Dataset(
        test_paths, test_disease_labels,
        transform=get_preprocessing_transforms()
    )
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Initialize model
    print(f"\nInitializing Stage 2 model with {backbone} backbone...")
    model = Stage2DiseaseClassifier(
        backbone=backbone,
        num_classes=len(disease_classes),
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
    class_weights_for_loss = class_weights_tensor if use_weighted_loss else None
    
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
        class_weights=class_weights_for_loss
    )
    
    return {
        'model': model,
        'trainer': trainer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'device': device,
        'disease_to_idx': disease_to_idx,
        'idx_to_disease': idx_to_disease,
        'class_weights': class_weights_dict,
        'minority_classes': minority_classes,
        'augmentation_strategy': augmentation_strategy,
        'stats': stats,
        'checkpoint_dir': checkpoint_dir
    }


def train_stage2(config):
    """
    Train Stage 2 model.
    
    Args:
        config (dict): Training configuration from prepare_stage2_training
        
    Returns:
        dict: Training results including metrics and model path
    """
    print("\n" + "="*80)
    print("STAGE 2 TRAINING")
    print("="*80)
    
    trainer = config['trainer']
    train_loader = config['train_loader']
    val_loader = config['val_loader']
    checkpoint_dir = config['checkpoint_dir']
    disease_to_idx = config['disease_to_idx']
    
    # Get Cedar Rust index for tracking
    cedar_rust_idx = disease_to_idx.get("Cedar Rust", None)
    
    # Train model with Cedar Rust recall tracking
    trainer.train(train_loader, val_loader, checkpoint_name="stage2_best.pth", 
                  track_class_recall=cedar_rust_idx)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("STAGE 2 EVALUATION")
    print("="*80)
    
    test_loader = config['test_loader']
    idx_to_disease = config['idx_to_disease']
    class_names = [idx_to_disease[i] for i in range(len(idx_to_disease))]
    
    eval_results = trainer.evaluate(test_loader, class_names=class_names)
    
    # Print evaluation results
    print(f"\nTest Set Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Test Set Macro F1-Score: {eval_results['f1_macro']:.4f}")
    
    print("\nPer-Class Metrics:")
    print(f"{'Disease':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 66)
    
    cedar_rust_recall = None
    for disease, metrics in eval_results['per_class_metrics'].items():
        print(f"{disease:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} {int(metrics['support']):<10}")
        
        if disease == "Cedar Rust":
            cedar_rust_recall = metrics['recall']
    
    print(f"\nCedar Rust Recall (target ≥0.70): {cedar_rust_recall:.4f}")
    
    # Save results
    results = {
        'accuracy': eval_results['accuracy'],
        'f1_macro': eval_results['f1_macro'],
        'cedar_rust_recall': cedar_rust_recall,
        'per_class_metrics': eval_results['per_class_metrics'],
        'confusion_matrix': eval_results['confusion_matrix'],
        'training_history': {
            'train': trainer.train_history,
            'val': trainer.val_history
        },
        'model_path': str(Path(checkpoint_dir) / "stage2_best.pth"),
        'class_names': class_names
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    dataset_root = "./data/raw/plantvillage"
    
    # Prepare training
    config = prepare_stage2_training(
        dataset_root=dataset_root,
        batch_size=32,
        num_workers=4,
        backbone="efficientnet_b0",
        learning_rate=0.001,
        epochs=50,
        early_stopping_patience=10,
        checkpoint_dir="./models/stage2",
        use_oversampling=True,
        use_weighted_loss=True
    )
    
    # Train model
    results = train_stage2(config)
    
    print("\nTraining complete!")
    print(f"Model saved to: {results['model_path']}")

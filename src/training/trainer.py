"""Training loop and utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np


class Trainer:
    """Trainer class for model training."""
    
    def __init__(
        self,
        model,
        device,
        learning_rate=0.001,
        epochs=50,
        early_stopping_patience=10,
        checkpoint_dir="./models",
        class_weights=None
    ):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Model to train
            device (torch.device): Device to use
            learning_rate (float): Learning rate
            epochs (int): Number of epochs
            early_stopping_patience (int): Early stopping patience
            checkpoint_dir (str): Directory to save checkpoints
            class_weights (torch.Tensor): Optional class weights for imbalanced data
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.best_model_state = None
        
        # Metrics tracking
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'f1_macro': []
        }
        self.val_history = {
            'loss': [],
            'accuracy': [],
            'f1_macro': []
        }
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            dict: Training metrics (loss, accuracy, f1_macro)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate macro F1-score
        _, _, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro
        }
    
    def validate(self, val_loader, track_class_recall=None):
        """
        Validate model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            track_class_recall (int): Optional class index to track recall for
            
        Returns:
            dict: Validation metrics (loss, accuracy, f1_macro)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Collect predictions for metrics
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate macro F1-score
        _, _, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        # Track specific class recall if requested
        if track_class_recall is not None:
            _, recall, _, _ = precision_recall_fscore_support(
                all_labels, all_preds, average=None, zero_division=0
            )
            results['tracked_class_recall'] = recall[track_class_recall]
        
        return results
    
    def train(self, train_loader, val_loader, checkpoint_name="model.pth", track_class_recall=None):
        """
        Train model with early stopping.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            checkpoint_name (str): Name for checkpoint file
            track_class_recall (int): Optional class index to track recall for
        """
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader, track_class_recall=track_class_recall)
            
            # Store history
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            self.train_history['f1_macro'].append(train_metrics['f1_macro'])
            
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['accuracy'].append(val_metrics['accuracy'])
            self.val_history['f1_macro'].append(val_metrics['f1_macro'])
            
            # Print training progress
            log_msg = (f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.4f}, "
                      f"Train F1: {train_metrics['f1_macro']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Val F1: {val_metrics['f1_macro']:.4f}")
            
            # Add tracked class recall to log if available
            if 'tracked_class_recall' in val_metrics:
                log_msg += f", Tracked Recall: {val_metrics['tracked_class_recall']:.4f}"
            
            print(log_msg)
            
            # Early stopping based on macro F1-score
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                torch.save(self.best_model_state, checkpoint_path)
                print(f"  -> Checkpoint saved to {checkpoint_path} (F1: {self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Best model restored (Val F1: {self.best_val_f1:.4f})")
    
    def save_model(self, path):
        """
        Save model.
        
        Args:
            path (str): Path to save model
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model.
        
        Args:
            path (str): Path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
    
    def evaluate(self, test_loader, class_names=None):
        """
        Evaluate model on test set with detailed metrics.
        
        Args:
            test_loader (DataLoader): Test data loader
            class_names (list): Optional list of class names for reporting
            
        Returns:
            dict: Evaluation metrics including per-class precision, recall, F1
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = np.mean(all_preds == all_labels)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Macro F1-score
        _, _, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        # Add class names if provided
        if class_names is not None:
            results['class_names'] = class_names
            results['per_class_metrics'] = {
                class_names[i]: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': support[i]
                }
                for i in range(len(class_names))
            }
        
        return results

"""Training loop and utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


class Trainer:
    """Trainer class for model training."""
    
    def __init__(
        self,
        model,
        device,
        learning_rate=0.001,
        epochs=50,
        early_stopping_patience=10,
        checkpoint_dir="./models"
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
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
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
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Validate model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, checkpoint_name="model.pth"):
        """
        Train model with early stopping.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            checkpoint_name (str): Name for checkpoint file
        """
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                torch.save(self.best_model_state, checkpoint_path)
                print(f"  -> Checkpoint saved to {checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Best model restored")
    
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

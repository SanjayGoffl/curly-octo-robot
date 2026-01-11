"""Model versioning and metadata management."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class ModelVersion:
    """Model version information."""
    
    def __init__(
        self,
        version: str,
        stage: str,
        backbone: str,
        dataset_version: str,
        training_date: str,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize model version.
        
        Args:
            version (str): Version string (e.g., "1.0.0")
            stage (str): Stage name ("stage1" or "stage2")
            backbone (str): Backbone architecture (e.g., "efficientnet_b0")
            dataset_version (str): Dataset version used for training
            training_date (str): Training date (ISO format)
            metrics (dict, optional): Training metrics
            notes (str, optional): Additional notes
        """
        self.version = version
        self.stage = stage
        self.backbone = backbone
        self.dataset_version = dataset_version
        self.training_date = training_date
        self.metrics = metrics or {}
        self.notes = notes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "stage": self.stage,
            "backbone": self.backbone,
            "dataset_version": self.dataset_version,
            "training_date": self.training_date,
            "metrics": self.metrics,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            stage=data["stage"],
            backbone=data["backbone"],
            dataset_version=data["dataset_version"],
            training_date=data["training_date"],
            metrics=data.get("metrics"),
            notes=data.get("notes")
        )


class ModelVersionManager:
    """Manages model versions and metadata."""
    
    def __init__(self, models_dir: str = "./models"):
        """
        Initialize model version manager.
        
        Args:
            models_dir (str): Directory containing models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("crop_disease_identifier.versioning")
        
        # Version metadata files
        self.stage1_version_file = self.models_dir / "stage1" / "version.json"
        self.stage2_version_file = self.models_dir / "stage2" / "version.json"
        
        # Ensure directories exist
        (self.models_dir / "stage1").mkdir(parents=True, exist_ok=True)
        (self.models_dir / "stage2").mkdir(parents=True, exist_ok=True)
    
    def save_model_version(
        self,
        stage: str,
        version: str,
        backbone: str,
        dataset_version: str,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None
    ) -> None:
        """
        Save model version metadata.
        
        Args:
            stage (str): Stage name ("stage1" or "stage2")
            version (str): Version string
            backbone (str): Backbone architecture
            dataset_version (str): Dataset version
            metrics (dict, optional): Training metrics
            notes (str, optional): Additional notes
        """
        model_version = ModelVersion(
            version=version,
            stage=stage,
            backbone=backbone,
            dataset_version=dataset_version,
            training_date=datetime.now().isoformat(),
            metrics=metrics,
            notes=notes
        )
        
        # Determine version file path
        if stage == "stage1":
            version_file = self.stage1_version_file
        elif stage == "stage2":
            version_file = self.stage2_version_file
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        # Write version metadata
        try:
            with open(version_file, "w") as f:
                json.dump(model_version.to_dict(), f, indent=2)
            self.logger.info(f"Saved {stage} model version: {version}")
        except Exception as e:
            self.logger.error(f"Failed to save {stage} model version: {str(e)}")
    
    def load_model_version(self, stage: str) -> Optional[ModelVersion]:
        """
        Load model version metadata.
        
        Args:
            stage (str): Stage name ("stage1" or "stage2")
            
        Returns:
            ModelVersion: Model version information or None if not found
        """
        # Determine version file path
        if stage == "stage1":
            version_file = self.stage1_version_file
        elif stage == "stage2":
            version_file = self.stage2_version_file
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        if not version_file.exists():
            self.logger.warning(f"Version file not found for {stage}: {version_file}")
            return None
        
        try:
            with open(version_file, "r") as f:
                data = json.load(f)
            return ModelVersion.from_dict(data)
        except Exception as e:
            self.logger.error(f"Failed to load {stage} model version: {str(e)}")
            return None
    
    def get_current_version(self, stage: str) -> str:
        """
        Get current model version string.
        
        Args:
            stage (str): Stage name ("stage1" or "stage2")
            
        Returns:
            str: Version string or "unknown" if not found
        """
        model_version = self.load_model_version(stage)
        if model_version:
            return model_version.version
        return "unknown"
    
    def get_dataset_version(self, stage: str) -> str:
        """
        Get dataset version used for training.
        
        Args:
            stage (str): Stage name ("stage1" or "stage2")
            
        Returns:
            str: Dataset version or "unknown" if not found
        """
        model_version = self.load_model_version(stage)
        if model_version:
            return model_version.dataset_version
        return "unknown"
    
    def get_model_info(self, stage: str) -> Dict[str, Any]:
        """
        Get complete model information.
        
        Args:
            stage (str): Stage name ("stage1" or "stage2")
            
        Returns:
            dict: Model information
        """
        model_version = self.load_model_version(stage)
        if model_version:
            return model_version.to_dict()
        return {
            "version": "unknown",
            "stage": stage,
            "backbone": "unknown",
            "dataset_version": "unknown",
            "training_date": None,
            "metrics": {},
            "notes": None
        }


# Global version manager instance
_version_manager: Optional[ModelVersionManager] = None


def get_version_manager(models_dir: str = "./models") -> ModelVersionManager:
    """
    Get or create version manager instance.
    
    Args:
        models_dir (str): Directory containing models
        
    Returns:
        ModelVersionManager: Version manager instance
    """
    global _version_manager
    if _version_manager is None:
        _version_manager = ModelVersionManager(models_dir)
    return _version_manager

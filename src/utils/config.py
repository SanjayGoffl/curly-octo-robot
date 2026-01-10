"""Configuration management."""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration class for the application."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Load environment variables from .env file
        load_dotenv()
        
        # GPU Configuration
        self.use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        self.cuda_device = int(os.getenv("CUDA_DEVICE", "0"))
        
        # Model Configuration
        self.stage1_backbone = os.getenv("STAGE1_BACKBONE", "efficientnet_b0")
        self.stage2_backbone = os.getenv("STAGE2_BACKBONE", "efficientnet_b0")
        self.stage1_threshold = float(os.getenv("STAGE1_THRESHOLD", "0.7"))
        self.stage2_threshold = float(os.getenv("STAGE2_THRESHOLD", "0.6"))
        
        # Training Configuration
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
        self.epochs = int(os.getenv("EPOCHS", "50"))
        self.early_stopping_patience = int(os.getenv("EARLY_STOPPING_PATIENCE", "10"))
        
        # Data Configuration
        self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
        self.train_split = float(os.getenv("TRAIN_SPLIT", "0.7"))
        self.val_split = float(os.getenv("VAL_SPLIT", "0.15"))
        self.test_split = float(os.getenv("TEST_SPLIT", "0.15"))
        
        # Logging Configuration
        self.log_dir = Path(os.getenv("LOG_DIR", "./logs"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Web Application Configuration
        self.flask_env = os.getenv("FLASK_ENV", "development")
        self.flask_debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
        self.max_upload_size = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "splits").mkdir(parents=True, exist_ok=True)
        
        models_dir = Path("./models")
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "stage1").mkdir(parents=True, exist_ok=True)
        (models_dir / "stage2").mkdir(parents=True, exist_ok=True)
    
    def __repr__(self):
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  use_gpu={self.use_gpu},\n"
            f"  cuda_device={self.cuda_device},\n"
            f"  stage1_backbone={self.stage1_backbone},\n"
            f"  stage2_backbone={self.stage2_backbone},\n"
            f"  batch_size={self.batch_size},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  epochs={self.epochs}\n"
            f")"
        )


# Global configuration instance
config = Config()

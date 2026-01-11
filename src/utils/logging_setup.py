"""Logging setup and initialization."""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from src.utils.logging import setup_logging, get_logger
from src.utils.prediction_logger import get_prediction_logger
from src.utils.model_versioning import get_version_manager


def initialize_logging(log_dir: str = "./logs", log_level: str = "INFO") -> None:
    """
    Initialize all logging infrastructure.
    
    Args:
        log_dir (str): Directory to store logs
        log_level (str): Logging level
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Set up main logger with rotation
    setup_logging(log_dir, log_level)
    
    # Get logger
    logger = get_logger("setup")
    logger.info(f"Logging initialized in {log_dir}")
    logger.info(f"Log level: {log_level}")
    
    # Initialize prediction logger
    prediction_logger = get_prediction_logger(log_dir)
    logger.info("Prediction logger initialized")
    
    # Initialize version manager
    version_manager = get_version_manager()
    logger.info("Model version manager initialized")
    
    # Log system information
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("Logging infrastructure ready")


def get_log_stats(log_dir: str = "./logs") -> dict:
    """
    Get logging statistics.
    
    Args:
        log_dir (str): Directory containing logs
        
    Returns:
        dict: Logging statistics
    """
    log_path = Path(log_dir)
    
    stats = {
        "log_directory": str(log_path),
        "exists": log_path.exists(),
        "log_files": [],
        "total_size_mb": 0.0
    }
    
    if log_path.exists():
        for log_file in log_path.glob("*.log*"):
            file_size_mb = log_file.stat().st_size / (1024 * 1024)
            stats["log_files"].append({
                "name": log_file.name,
                "size_mb": round(file_size_mb, 2),
                "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
            })
            stats["total_size_mb"] += file_size_mb
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    
    return stats


if __name__ == "__main__":
    # Initialize logging
    initialize_logging()
    
    # Get logger
    logger = get_logger("main")
    logger.info("Logging setup complete")
    
    # Print stats
    stats = get_log_stats()
    logger.info(f"Log statistics: {stats}")

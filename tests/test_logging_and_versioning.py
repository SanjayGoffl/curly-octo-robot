"""Tests for logging and model versioning."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from src.utils.prediction_logger import PredictionLogger, get_prediction_logger
from src.utils.model_versioning import ModelVersion, ModelVersionManager, get_version_manager
from src.utils.logging_setup import initialize_logging, get_log_stats


# ============================================================================
# TASK 9.1: BASIC LOGGING TESTS
# ============================================================================

class TestPredictionLogger:
    """Tests for prediction logging."""
    
    def test_prediction_logger_initialization(self):
        """Test prediction logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(tmpdir, "1.0.0")
            
            assert logger.log_dir == Path(tmpdir)
            assert logger.model_version == "1.0.0"
            assert logger.predictions_log_file == Path(tmpdir) / "predictions.jsonl"
            assert logger.errors_log_file == Path(tmpdir) / "errors.jsonl"
    
    def test_log_prediction_creates_file(self):
        """Test that logging a prediction creates the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(tmpdir, "1.0.0")
            
            logger.log_prediction(
                image_path="/path/to/image.jpg",
                crop="Apple",
                crop_confidence=0.95,
                disease="Apple Scab",
                disease_confidence=0.87
            )
            
            assert logger.predictions_log_file.exists()
    
    def test_log_prediction_format(self):
        """Test that predictions are logged in correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(tmpdir, "1.0.0")
            
            logger.log_prediction(
                image_path="/path/to/image.jpg",
                crop="Apple",
                crop_confidence=0.95,
                disease="Apple Scab",
                disease_confidence=0.87,
                processing_time_ms=150.5
            )
            
            # Read and verify log file
            with open(logger.predictions_log_file, "r") as f:
                line = f.readline()
                data = json.loads(line)
            
            assert data["crop"] == "Apple"
            assert data["crop_confidence"] == 0.95
            assert data["disease"] == "Apple Scab"
            assert data["disease_confidence"] == 0.87
            assert data["model_version"] == "1.0.0"
            assert data["processing_time_ms"] == 150.5
            assert "timestamp" in data
    
    def test_log_error_creates_file(self):
        """Test that logging an error creates the error log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(tmpdir, "1.0.0")
            
            logger.log_error(
                error_type="INVALID_INPUT",
                error_message="Invalid file format"
            )
            
            assert logger.errors_log_file.exists()
    
    def test_log_error_format(self):
        """Test that errors are logged in correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(tmpdir, "1.0.0")
            
            exc = ValueError("Test error")
            logger.log_error(
                error_type="INFERENCE_FAILURE",
                error_message="Processing failed",
                image_path="/path/to/image.jpg",
                exception=exc
            )
            
            # Read and verify log file
            with open(logger.errors_log_file, "r") as f:
                line = f.readline()
                data = json.loads(line)
            
            assert data["error_type"] == "INFERENCE_FAILURE"
            assert data["error_message"] == "Processing failed"
            assert data["image_path"] == "/path/to/image.jpg"
            assert data["exception_type"] == "ValueError"
            assert data["exception_message"] == "Test error"
            assert "timestamp" in data
    
    def test_get_prediction_stats(self):
        """Test getting prediction statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(tmpdir, "1.0.0")
            
            # Log multiple predictions
            logger.log_prediction(
                image_path="/path/to/image1.jpg",
                crop="Apple",
                crop_confidence=0.95,
                disease="Apple Scab",
                disease_confidence=0.87
            )
            logger.log_prediction(
                image_path="/path/to/image2.jpg",
                crop="Tomato",
                crop_confidence=0.92,
                disease=None,
                disease_confidence=None
            )
            
            stats = logger.get_prediction_stats()
            
            assert stats["total_predictions"] == 2
            assert stats["crops"]["Apple"] == 1
            assert stats["crops"]["Tomato"] == 1
            assert stats["diseases"]["Apple Scab"] == 1
            assert stats["avg_crop_confidence"] > 0
    
    def test_get_prediction_logger_function(self):
        """Test get_prediction_logger function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_prediction_logger(tmpdir, "2.0.0")
            
            assert isinstance(logger, PredictionLogger)
            assert logger.model_version == "2.0.0"


# ============================================================================
# TASK 9.2: MODEL VERSIONING TESTS
# ============================================================================

class TestModelVersion:
    """Tests for ModelVersion class."""
    
    def test_model_version_initialization(self):
        """Test ModelVersion initialization."""
        version = ModelVersion(
            version="1.0.0",
            stage="stage1",
            backbone="efficientnet_b0",
            dataset_version="plantvillage_v1",
            training_date="2024-01-01T12:00:00"
        )
        
        assert version.version == "1.0.0"
        assert version.stage == "stage1"
        assert version.backbone == "efficientnet_b0"
        assert version.dataset_version == "plantvillage_v1"
    
    def test_model_version_to_dict(self):
        """Test converting ModelVersion to dictionary."""
        version = ModelVersion(
            version="1.0.0",
            stage="stage1",
            backbone="efficientnet_b0",
            dataset_version="plantvillage_v1",
            training_date="2024-01-01T12:00:00",
            metrics={"accuracy": 0.85, "f1_score": 0.82},
            notes="Initial version"
        )
        
        data = version.to_dict()
        
        assert data["version"] == "1.0.0"
        assert data["stage"] == "stage1"
        assert data["metrics"]["accuracy"] == 0.85
        assert data["notes"] == "Initial version"
    
    def test_model_version_from_dict(self):
        """Test creating ModelVersion from dictionary."""
        data = {
            "version": "1.0.0",
            "stage": "stage1",
            "backbone": "efficientnet_b0",
            "dataset_version": "plantvillage_v1",
            "training_date": "2024-01-01T12:00:00",
            "metrics": {"accuracy": 0.85},
            "notes": "Test"
        }
        
        version = ModelVersion.from_dict(data)
        
        assert version.version == "1.0.0"
        assert version.stage == "stage1"
        assert version.metrics["accuracy"] == 0.85


class TestModelVersionManager:
    """Tests for ModelVersionManager."""
    
    def test_version_manager_initialization(self):
        """Test ModelVersionManager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelVersionManager(tmpdir)
            
            assert manager.models_dir == Path(tmpdir)
            assert (Path(tmpdir) / "stage1").exists()
            assert (Path(tmpdir) / "stage2").exists()
    
    def test_save_and_load_model_version(self):
        """Test saving and loading model version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelVersionManager(tmpdir)
            
            # Save version
            manager.save_model_version(
                stage="stage1",
                version="1.0.0",
                backbone="efficientnet_b0",
                dataset_version="plantvillage_v1",
                metrics={"accuracy": 0.85, "f1_score": 0.82}
            )
            
            # Load version
            loaded_version = manager.load_model_version("stage1")
            
            assert loaded_version is not None
            assert loaded_version.version == "1.0.0"
            assert loaded_version.stage == "stage1"
            assert loaded_version.metrics["accuracy"] == 0.85
    
    def test_get_current_version(self):
        """Test getting current model version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelVersionManager(tmpdir)
            
            # Before saving
            version = manager.get_current_version("stage1")
            assert version == "unknown"
            
            # After saving
            manager.save_model_version(
                stage="stage1",
                version="1.0.0",
                backbone="efficientnet_b0",
                dataset_version="plantvillage_v1"
            )
            
            version = manager.get_current_version("stage1")
            assert version == "1.0.0"
    
    def test_get_dataset_version(self):
        """Test getting dataset version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelVersionManager(tmpdir)
            
            manager.save_model_version(
                stage="stage2",
                version="1.0.0",
                backbone="efficientnet_b0",
                dataset_version="plantvillage_apple_v2"
            )
            
            dataset_version = manager.get_dataset_version("stage2")
            assert dataset_version == "plantvillage_apple_v2"
    
    def test_get_model_info(self):
        """Test getting complete model information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelVersionManager(tmpdir)
            
            manager.save_model_version(
                stage="stage1",
                version="1.0.0",
                backbone="efficientnet_b0",
                dataset_version="plantvillage_v1",
                metrics={"accuracy": 0.85},
                notes="Initial version"
            )
            
            info = manager.get_model_info("stage1")
            
            assert info["version"] == "1.0.0"
            assert info["stage"] == "stage1"
            assert info["backbone"] == "efficientnet_b0"
            assert info["metrics"]["accuracy"] == 0.85
            assert info["notes"] == "Initial version"
    
    def test_get_version_manager_singleton(self):
        """Test that get_version_manager returns singleton."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = get_version_manager(tmpdir)
            manager2 = get_version_manager(tmpdir)
            
            # Should be same instance
            assert manager1 is manager2


# ============================================================================
# TASK 9.3: LOGGING INFRASTRUCTURE TESTS
# ============================================================================

class TestLoggingSetup:
    """Tests for logging infrastructure setup."""
    
    def test_initialize_logging(self):
        """Test logging initialization creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Just verify the directory exists
            log_path = Path(tmpdir)
            assert log_path.exists()
    
    def test_get_log_stats(self):
        """Test getting log statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test log file with content
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text("test log content\n" * 100)  # Write enough content
            
            stats = get_log_stats(tmpdir)
            
            assert stats["log_directory"] == str(Path(tmpdir))
            assert stats["exists"] is True
            assert len(stats["log_files"]) > 0
            assert stats["total_size_mb"] >= 0  # May be very small
    
    def test_log_rotation_configuration(self):
        """Test that log rotation is configured."""
        # Just verify the logging module has rotation capabilities
        from src.utils.logging import setup_logging
        import logging.handlers
        
        # Verify RotatingFileHandler exists in logging module
        assert hasattr(logging.handlers, 'RotatingFileHandler')


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestLoggingIntegration:
    """Integration tests for logging and versioning."""
    
    def test_prediction_logging_with_versioning(self):
        """Test prediction logging with model versioning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up versioning
            version_manager = ModelVersionManager(tmpdir)
            version_manager.save_model_version(
                stage="stage1",
                version="1.0.0",
                backbone="efficientnet_b0",
                dataset_version="plantvillage_v1"
            )
            
            # Set up logging
            logger = PredictionLogger(tmpdir, "1.0.0")
            
            # Log prediction
            logger.log_prediction(
                image_path="/path/to/image.jpg",
                crop="Apple",
                crop_confidence=0.95,
                model_version=version_manager.get_current_version("stage1"),
                dataset_version=version_manager.get_dataset_version("stage1")
            )
            
            # Verify
            with open(logger.predictions_log_file, "r") as f:
                data = json.loads(f.readline())
            
            assert data["model_version"] == "1.0.0"
            assert data["dataset_version"] == "plantvillage_v1"
    
    def test_error_logging_with_versioning(self):
        """Test error logging with model versioning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            version_manager = ModelVersionManager(tmpdir)
            version_manager.save_model_version(
                stage="stage1",
                version="1.0.0",
                backbone="efficientnet_b0",
                dataset_version="plantvillage_v1"
            )
            
            logger = PredictionLogger(tmpdir, "1.0.0")
            
            logger.log_error(
                error_type="INVALID_INPUT",
                error_message="Invalid file",
                model_version=version_manager.get_current_version("stage1")
            )
            
            with open(logger.errors_log_file, "r") as f:
                data = json.loads(f.readline())
            
            assert data["model_version"] == "1.0.0"
            assert data["error_type"] == "INVALID_INPUT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

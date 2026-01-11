"""Prediction logging utilities."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class PredictionLogger:
    """Logger for prediction results and errors."""
    
    def __init__(self, log_dir: str = "./logs", model_version: str = "1.0.0"):
        """
        Initialize prediction logger.
        
        Args:
            log_dir (str): Directory to store prediction logs
            model_version (str): Current model version
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version
        
        # Get logger
        self.logger = logging.getLogger("crop_disease_identifier.predictions")
        
        # Create predictions log file
        self.predictions_log_file = self.log_dir / "predictions.jsonl"
        self.errors_log_file = self.log_dir / "errors.jsonl"
    
    def log_prediction(
        self,
        image_path: str,
        crop: Optional[str],
        crop_confidence: Optional[float],
        disease: Optional[str] = None,
        disease_confidence: Optional[float] = None,
        model_version: Optional[str] = None,
        dataset_version: Optional[str] = None,
        processing_time_ms: Optional[float] = None
    ) -> None:
        """
        Log a prediction result.
        
        Args:
            image_path (str): Path to input image
            crop (str): Predicted crop name
            crop_confidence (float): Crop prediction confidence
            disease (str, optional): Predicted disease name
            disease_confidence (float, optional): Disease prediction confidence
            model_version (str, optional): Model version used
            dataset_version (str, optional): Dataset version used for training
            processing_time_ms (float, optional): Processing time in milliseconds
        """
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "crop": crop,
            "crop_confidence": round(crop_confidence, 4) if crop_confidence is not None else None,
            "disease": disease,
            "disease_confidence": round(disease_confidence, 4) if disease_confidence is not None else None,
            "model_version": model_version or self.model_version,
            "dataset_version": dataset_version,
            "processing_time_ms": round(processing_time_ms, 2) if processing_time_ms is not None else None
        }
        
        # Write to JSONL file
        try:
            with open(self.predictions_log_file, "a") as f:
                f.write(json.dumps(prediction_data) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write prediction log: {str(e)}")
        
        # Also log to standard logger
        disease_conf_str = f"{disease_confidence:.4f}" if disease_confidence is not None else "N/A"
        self.logger.info(
            f"Prediction: crop={crop} (conf={crop_confidence:.4f}), "
            f"disease={disease} (conf={disease_conf_str})"
        )
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        image_path: Optional[str] = None,
        model_version: Optional[str] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """
        Log an error.
        
        Args:
            error_type (str): Type of error
            error_message (str): Error message
            image_path (str, optional): Path to input image
            model_version (str, optional): Model version
            exception (Exception, optional): Exception object
        """
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "image_path": str(image_path) if image_path else None,
            "model_version": model_version or self.model_version,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_message": str(exception) if exception else None
        }
        
        # Write to JSONL file
        try:
            with open(self.errors_log_file, "a") as f:
                f.write(json.dumps(error_data) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write error log: {str(e)}")
        
        # Also log to standard logger
        self.logger.error(
            f"Error: type={error_type}, message={error_message}",
            exc_info=exception
        )
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """
        Get statistics from prediction logs.
        
        Returns:
            dict: Statistics including total predictions, crops, diseases, etc.
        """
        stats = {
            "total_predictions": 0,
            "crops": {},
            "diseases": {},
            "avg_crop_confidence": 0.0,
            "avg_disease_confidence": 0.0,
            "total_errors": 0
        }
        
        if not self.predictions_log_file.exists():
            return stats
        
        crop_confidences = []
        disease_confidences = []
        
        try:
            with open(self.predictions_log_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        stats["total_predictions"] += 1
                        
                        # Count crops
                        crop = data.get("crop")
                        if crop:
                            stats["crops"][crop] = stats["crops"].get(crop, 0) + 1
                        
                        # Count diseases
                        disease = data.get("disease")
                        if disease:
                            stats["diseases"][disease] = stats["diseases"].get(disease, 0) + 1
                        
                        # Collect confidences
                        if data.get("crop_confidence") is not None:
                            crop_confidences.append(data["crop_confidence"])
                        if data.get("disease_confidence") is not None:
                            disease_confidences.append(data["disease_confidence"])
        except Exception as e:
            self.logger.error(f"Failed to read prediction logs: {str(e)}")
        
        # Calculate averages
        if crop_confidences:
            stats["avg_crop_confidence"] = round(sum(crop_confidences) / len(crop_confidences), 4)
        if disease_confidences:
            stats["avg_disease_confidence"] = round(sum(disease_confidences) / len(disease_confidences), 4)
        
        # Count errors
        if self.errors_log_file.exists():
            try:
                with open(self.errors_log_file, "r") as f:
                    stats["total_errors"] = sum(1 for line in f if line.strip())
            except Exception as e:
                self.logger.error(f"Failed to count errors: {str(e)}")
        
        return stats


def get_prediction_logger(log_dir: str = "./logs", model_version: str = "1.0.0") -> PredictionLogger:
    """
    Get a prediction logger instance.
    
    Args:
        log_dir (str): Directory to store logs
        model_version (str): Model version
        
    Returns:
        PredictionLogger: Logger instance
    """
    return PredictionLogger(log_dir, model_version)

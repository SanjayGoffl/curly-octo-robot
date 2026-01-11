"""Hierarchical inference pipeline."""

import torch
import torch.nn.functional as F
import time
from pathlib import Path
from src.data.preprocessing import preprocess_image
from src.inference.validation import validate_input_file
from src.inference.recommendations import get_recommendations
from src.utils.prediction_logger import get_prediction_logger
from src.utils.model_versioning import get_version_manager


class InferencePipeline:
    """Hierarchical two-stage inference pipeline."""
    
    def __init__(
        self,
        stage1_model,
        stage2_model,
        device,
        stage1_threshold=0.7,
        stage2_threshold=0.6,
        stage1_classes=None,
        stage2_classes=None,
        log_dir="./logs",
        models_dir="./models"
    ):
        """
        Initialize inference pipeline.
        
        Args:
            stage1_model (nn.Module): Stage 1 crop identification model
            stage2_model (nn.Module): Stage 2 disease classification model
            device (torch.device): Device to use
            stage1_threshold (float): Confidence threshold for Stage 1
            stage2_threshold (float): Confidence threshold for Stage 2
            stage1_classes (list): Stage 1 class names
            stage2_classes (list): Stage 2 class names
            log_dir (str): Directory for logs
            models_dir (str): Directory containing models
        """
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.device = device
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        
        # Default class names
        self.stage1_classes = stage1_classes or [f"Crop {i}" for i in range(14)]
        self.stage2_classes = stage2_classes or [
            "Apple Healthy",
            "Apple Scab",
            "Apple Black Rot",
            "Apple Cedar Rust"
        ]
        
        # Initialize logging and versioning
        self.version_manager = get_version_manager(models_dir)
        stage1_version = self.version_manager.get_current_version("stage1")
        self.prediction_logger = get_prediction_logger(log_dir, stage1_version)
        
        # Set models to eval mode
        self.stage1_model.eval()
        self.stage2_model.eval()
    
    def predict(self, image_path):
        """
        Run hierarchical inference pipeline.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            dict: Prediction result with crop, disease, confidence, and recommendations
        """
        start_time = time.time()
        
        # Validate input file
        is_valid, error_msg = validate_input_file(image_path)
        if not is_valid:
            self.prediction_logger.log_error(
                error_type="INVALID_INPUT",
                error_message=error_msg,
                image_path=image_path
            )
            return {
                "success": False,
                "error": error_msg,
                "crop": None,
                "disease": None,
                "crop_confidence": None,
                "disease_confidence": None,
                "recommendations": None
            }
        
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Stage 1: Crop identification
            with torch.no_grad():
                stage1_logits = self.stage1_model(image_tensor)
                stage1_probs = F.softmax(stage1_logits, dim=1)
                stage1_confidence, stage1_pred = torch.max(stage1_probs, dim=1)
            
            stage1_confidence = stage1_confidence.item()
            stage1_pred_idx = stage1_pred.item()
            crop_name = self.stage1_classes[stage1_pred_idx]
            
            # Check Stage 1 confidence threshold
            if stage1_confidence < self.stage1_threshold:
                processing_time = (time.time() - start_time) * 1000
                self.prediction_logger.log_prediction(
                    image_path=image_path,
                    crop="Unknown crop",
                    crop_confidence=stage1_confidence,
                    disease=None,
                    disease_confidence=None,
                    model_version=self.version_manager.get_current_version("stage1"),
                    dataset_version=self.version_manager.get_dataset_version("stage1"),
                    processing_time_ms=processing_time
                )
                return {
                    "success": True,
                    "error": None,
                    "crop": "Unknown crop",
                    "disease": None,
                    "crop_confidence": stage1_confidence,
                    "disease_confidence": None,
                    "recommendations": "Try taking a clearer photo of a single leaf or submit for expert review."
                }
            
            # Check if crop is Apple
            if crop_name != "Apple":
                processing_time = (time.time() - start_time) * 1000
                self.prediction_logger.log_prediction(
                    image_path=image_path,
                    crop=crop_name,
                    crop_confidence=stage1_confidence,
                    disease=None,
                    disease_confidence=None,
                    model_version=self.version_manager.get_current_version("stage1"),
                    dataset_version=self.version_manager.get_dataset_version("stage1"),
                    processing_time_ms=processing_time
                )
                return {
                    "success": True,
                    "error": None,
                    "crop": crop_name,
                    "disease": None,
                    "crop_confidence": stage1_confidence,
                    "disease_confidence": None,
                    "recommendations": None
                }
            
            # Stage 2: Apple disease classification
            with torch.no_grad():
                stage2_logits = self.stage2_model(image_tensor)
                stage2_probs = F.softmax(stage2_logits, dim=1)
                stage2_confidence, stage2_pred = torch.max(stage2_probs, dim=1)
            
            stage2_confidence = stage2_confidence.item()
            stage2_pred_idx = stage2_pred.item()
            disease_name = self.stage2_classes[stage2_pred_idx]
            
            # Check Stage 2 confidence threshold
            if stage2_confidence < self.stage2_threshold:
                processing_time = (time.time() - start_time) * 1000
                self.prediction_logger.log_prediction(
                    image_path=image_path,
                    crop=crop_name,
                    crop_confidence=stage1_confidence,
                    disease="Unknown disease",
                    disease_confidence=stage2_confidence,
                    model_version=self.version_manager.get_current_version("stage2"),
                    dataset_version=self.version_manager.get_dataset_version("stage2"),
                    processing_time_ms=processing_time
                )
                return {
                    "success": True,
                    "error": None,
                    "crop": crop_name,
                    "disease": "Unknown disease",
                    "crop_confidence": stage1_confidence,
                    "disease_confidence": stage2_confidence,
                    "recommendations": "Prediction confidence low. Please retake photo or consult an expert."
                }
            
            # Get recommendations
            recommendations = get_recommendations(disease_name)
            if recommendations:
                recommendations_text = (
                    f"Treatment: {recommendations['treatment']}\n"
                    f"Prevention: {recommendations['prevention']}"
                )
            else:
                recommendations_text = "Please consult an agronomist."
            
            # Log successful prediction
            processing_time = (time.time() - start_time) * 1000
            self.prediction_logger.log_prediction(
                image_path=image_path,
                crop=crop_name,
                crop_confidence=stage1_confidence,
                disease=disease_name,
                disease_confidence=stage2_confidence,
                model_version=self.version_manager.get_current_version("stage2"),
                dataset_version=self.version_manager.get_dataset_version("stage2"),
                processing_time_ms=processing_time
            )
            
            return {
                "success": True,
                "error": None,
                "crop": crop_name,
                "disease": disease_name,
                "crop_confidence": stage1_confidence,
                "disease_confidence": stage2_confidence,
                "recommendations": recommendations_text
            }
        
        except Exception as e:
            self.prediction_logger.log_error(
                error_type="INFERENCE_FAILURE",
                error_message=str(e),
                image_path=image_path,
                exception=e
            )
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "crop": None,
                "disease": None,
                "crop_confidence": None,
                "disease_confidence": None,
                "recommendations": None
            }

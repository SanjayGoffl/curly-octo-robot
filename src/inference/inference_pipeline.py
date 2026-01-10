"""
End-to-end inference pipeline for crop disease identification.

This module orchestrates the complete inference flow:
1. Input validation (file type, size, integrity)
2. Basic leaf detection
3. Stage 1: Crop identification
4. Stage 2: Apple disease classification (if Apple detected)
5. Treatment recommendations
6. Result formatting
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

from src.models.stage1 import Stage1CropIdentifier
from src.models.stage2 import Stage2DiseaseClassifier
from src.data.preprocessing import get_preprocessing_transforms, validate_image_format
from src.inference.recommendations import get_recommendations


class PredictionStatus(Enum):
    """Status codes for predictions."""
    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    NON_LEAF = "non_leaf"
    UNKNOWN_CROP = "unknown_crop"
    UNKNOWN_DISEASE = "unknown_disease"
    LOW_CONFIDENCE = "low_confidence"
    SYSTEM_ERROR = "system_error"


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    status: PredictionStatus
    crop_name: Optional[str] = None
    crop_confidence: Optional[float] = None
    disease_name: Optional[str] = None
    disease_confidence: Optional[float] = None
    treatment: Optional[str] = None
    prevention: Optional[str] = None
    warnings: List[str] = None
    error_message: Optional[str] = None
    model_version: str = "1.0"
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON serialization."""
        return {
            'status': self.status.value,
            'crop': self.crop_name,
            'crop_confidence': round(self.crop_confidence, 4) if self.crop_confidence else None,
            'disease': self.disease_name,
            'disease_confidence': round(self.disease_confidence, 4) if self.disease_confidence else None,
            'treatment': self.treatment,
            'prevention': self.prevention,
            'warnings': self.warnings,
            'error': self.error_message,
            'model_version': self.model_version
        }


class InferencePipeline:
    """
    End-to-end inference pipeline for crop disease identification.
    
    Orchestrates:
    - Input validation
    - Leaf detection
    - Stage 1 crop identification
    - Stage 2 apple disease classification
    - Treatment recommendations
    """
    
    # Confidence thresholds (configurable)
    CROP_CONFIDENCE_THRESHOLD = 0.7
    DISEASE_CONFIDENCE_THRESHOLD = 0.6
    
    # Model paths
    STAGE1_MODEL_PATH = "./models/stage1/stage1_best.pth"
    STAGE2_MODEL_PATH = "./models/stage2/stage2_best.pth"
    
    # Class mappings
    CROP_CLASSES = [
        "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Peach", "Pepper",
        "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato", "Wheat"
    ]
    
    DISEASE_CLASSES = ["Healthy", "Scab", "Black Rot", "Cedar Rust"]
    
    def __init__(
        self,
        stage1_model_path: str = None,
        stage2_model_path: str = None,
        device: torch.device = None,
        crop_threshold: float = 0.7,
        disease_threshold: float = 0.6
    ):
        """
        Initialize inference pipeline.
        
        Args:
            stage1_model_path (str): Path to Stage 1 model checkpoint
            stage2_model_path (str): Path to Stage 2 model checkpoint
            device (torch.device): Device to use (auto-detect if None)
            crop_threshold (float): Confidence threshold for crop prediction
            disease_threshold (float): Confidence threshold for disease prediction
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.crop_threshold = crop_threshold
        self.disease_threshold = disease_threshold
        
        # Load models
        self.stage1_model = self._load_stage1_model(stage1_model_path)
        self.stage2_model = self._load_stage2_model(stage2_model_path)
        
        # Preprocessing
        self.preprocess = get_preprocessing_transforms()
        
        print(f"InferencePipeline initialized on {self.device}")
        print(f"Crop threshold: {self.crop_threshold}")
        print(f"Disease threshold: {self.disease_threshold}")
    
    def _load_stage1_model(self, model_path: str = None) -> Stage1CropIdentifier:
        """Load Stage 1 model."""
        if model_path is None:
            model_path = self.STAGE1_MODEL_PATH
        
        try:
            model = Stage1CropIdentifier(
                backbone="efficientnet_b0",
                num_classes=len(self.CROP_CLASSES),
                pretrained=False
            )
            
            if Path(model_path).exists():
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Stage 1 model loaded from {model_path}")
            else:
                print(f"Warning: Stage 1 model not found at {model_path}, using untrained model")
            
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load Stage 1 model: {str(e)}")
    
    def _load_stage2_model(self, model_path: str = None) -> Stage2DiseaseClassifier:
        """Load Stage 2 model."""
        if model_path is None:
            model_path = self.STAGE2_MODEL_PATH
        
        try:
            model = Stage2DiseaseClassifier(
                backbone="efficientnet_b0",
                num_classes=len(self.DISEASE_CLASSES),
                pretrained=False
            )
            
            if Path(model_path).exists():
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Stage 2 model loaded from {model_path}")
            else:
                print(f"Warning: Stage 2 model not found at {model_path}, using untrained model")
            
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load Stage 2 model: {str(e)}")
    
    def validate_input(self, image_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate input image file.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check file exists
        if not Path(image_path).exists():
            return False, "File not found"
        
        # Check file size (max 10 MB)
        file_size_mb = Path(image_path).stat().st_size / (1024 * 1024)
        if file_size_mb > 10:
            return False, "File too large. Maximum 10 MB."
        
        # Validate image format and integrity
        is_valid, error_msg = validate_image_format(image_path)
        if not is_valid:
            return False, error_msg
        
        return True, None
    
    def detect_leaf(self, image: Image.Image) -> Tuple[bool, float, Optional[str]]:
        """
        Detect if image contains a valid leaf.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            tuple: (is_leaf, confidence, warning)
        """
        # Simple heuristic: check green pixel ratio
        # A leaf image should have significant green content
        
        image_array = np.array(image)
        
        # Calculate green channel prominence
        if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
            r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
            
            # Green pixels: G > R and G > B
            green_mask = (g > r) & (g > b)
            green_ratio = np.sum(green_mask) / (image_array.shape[0] * image_array.shape[1])
            
            # Thresholds
            if green_ratio > 0.3:
                return True, green_ratio, None
            elif green_ratio > 0.15:
                return True, green_ratio, "Image quality may affect accuracy"
            else:
                return False, green_ratio, None
        
        # Fallback: assume it's a leaf if we can't determine
        return True, 0.5, None
    
    def predict_stage1(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Run Stage 1 crop identification.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            tuple: (crop_name, confidence)
        """
        with torch.no_grad():
            outputs = self.stage1_model(image_tensor.unsqueeze(0).to(self.device))
            probabilities = torch.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)
            
            crop_name = self.CROP_CLASSES[class_idx.item()]
            confidence = confidence.item()
            
            return crop_name, confidence
    
    def predict_stage2(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Run Stage 2 apple disease classification.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            tuple: (disease_name, confidence)
        """
        with torch.no_grad():
            outputs = self.stage2_model(image_tensor.unsqueeze(0).to(self.device))
            probabilities = torch.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)
            
            disease_name = self.DISEASE_CLASSES[class_idx.item()]
            confidence = confidence.item()
            
            return disease_name, confidence
    
    def predict(self, image_path: str) -> PredictionResult:
        """
        Run complete inference pipeline on an image.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            PredictionResult: Complete prediction result
        """
        try:
            # Step 1: Validate input
            is_valid, error_msg = self.validate_input(image_path)
            if not is_valid:
                return PredictionResult(
                    status=PredictionStatus.INVALID_INPUT,
                    error_message=error_msg
                )
            
            # Step 2: Load and preprocess image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                return PredictionResult(
                    status=PredictionStatus.INVALID_INPUT,
                    error_message=f"Failed to load image: {str(e)}"
                )
            
            # Step 3: Detect leaf
            is_leaf, leaf_confidence, leaf_warning = self.detect_leaf(image)
            if not is_leaf:
                return PredictionResult(
                    status=PredictionStatus.NON_LEAF,
                    error_message="Invalid input — please upload a clear leaf image"
                )
            
            warnings = []
            if leaf_warning:
                warnings.append(leaf_warning)
            
            # Step 4: Preprocess image
            image_tensor = self.preprocess(image)
            
            # Step 5: Stage 1 - Crop identification
            crop_name, crop_confidence = self.predict_stage1(image_tensor)
            
            # Step 6: Check crop confidence threshold
            if crop_confidence < self.crop_threshold:
                return PredictionResult(
                    status=PredictionStatus.UNKNOWN_CROP,
                    crop_name=crop_name,
                    crop_confidence=crop_confidence,
                    error_message="Unknown crop / Not a supported crop. Try a clearer photo or submit for expert review.",
                    warnings=warnings
                )
            
            # Step 7: If Apple, run Stage 2
            if crop_name == "Apple":
                disease_name, disease_confidence = self.predict_stage2(image_tensor)
                
                # Check disease confidence threshold
                if disease_confidence < self.disease_threshold:
                    return PredictionResult(
                        status=PredictionStatus.UNKNOWN_DISEASE,
                        crop_name=crop_name,
                        crop_confidence=crop_confidence,
                        disease_name=disease_name,
                        disease_confidence=disease_confidence,
                        error_message="Unknown disease — consider expert review.",
                        warnings=warnings
                    )
                
                # Get treatment recommendations
                treatment, prevention = get_recommendations(disease_name)
                
                return PredictionResult(
                    status=PredictionStatus.SUCCESS,
                    crop_name=crop_name,
                    crop_confidence=crop_confidence,
                    disease_name=disease_name,
                    disease_confidence=disease_confidence,
                    treatment=treatment,
                    prevention=prevention,
                    warnings=warnings
                )
            else:
                # Non-Apple crop - return crop prediction only
                return PredictionResult(
                    status=PredictionStatus.SUCCESS,
                    crop_name=crop_name,
                    crop_confidence=crop_confidence,
                    warnings=warnings
                )
        
        except Exception as e:
            return PredictionResult(
                status=PredictionStatus.SYSTEM_ERROR,
                error_message=f"System error: {str(e)}"
            )
    
    def predict_batch(self, image_paths: List[str]) -> List[PredictionResult]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            list: List of PredictionResult objects
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results

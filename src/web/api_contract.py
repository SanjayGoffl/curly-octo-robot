"""
API Contract Specification for Crop Disease Identifier Web Service.

This module defines the exact request/response schemas and API endpoints
for the web application.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class APIStatus(Enum):
    """API response status codes."""
    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    NON_LEAF = "non_leaf"
    UNKNOWN_CROP = "unknown_crop"
    UNKNOWN_DISEASE = "unknown_disease"
    LOW_CONFIDENCE = "low_confidence"
    SYSTEM_ERROR = "system_error"


@dataclass
class PredictionResponse:
    """
    API Response Schema for Prediction Endpoint.
    
    Returned by: POST /api/predict
    """
    status: str  # APIStatus.value
    crop: Optional[str] = None
    crop_confidence: Optional[float] = None
    disease: Optional[str] = None
    disease_confidence: Optional[float] = None
    treatment: Optional[str] = None
    prevention: Optional[str] = None
    warnings: List[str] = None
    error: Optional[str] = None
    model_version: str = "1.0"
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResponse':
        """Create from dictionary."""
        return cls(**data)


class APIContract:
    """
    Complete API Contract Specification.
    
    Defines all endpoints, request/response schemas, and error handling.
    """
    
    # ============================================================================
    # ENDPOINT 1: Health Check
    # ============================================================================
    
    @staticmethod
    def health_check_spec() -> Dict[str, Any]:
        """
        GET /api/health
        
        Purpose: Check if API is running and models are loaded
        
        Response (200 OK):
        {
            "status": "healthy",
            "models_loaded": true,
            "device": "cuda",
            "version": "1.0"
        }
        
        Response (503 Service Unavailable):
        {
            "status": "unhealthy",
            "error": "Models not loaded"
        }
        """
        return {
            "endpoint": "GET /api/health",
            "description": "Health check - verify API and models are ready",
            "response_200": {
                "status": "healthy",
                "models_loaded": True,
                "device": "cuda",
                "version": "1.0"
            },
            "response_503": {
                "status": "unhealthy",
                "error": "Models not loaded"
            }
        }
    
    # ============================================================================
    # ENDPOINT 2: Single Image Prediction
    # ============================================================================
    
    @staticmethod
    def predict_spec() -> Dict[str, Any]:
        """
        POST /api/predict
        
        Purpose: Predict crop and disease from a single leaf image
        
        Request:
        - Content-Type: multipart/form-data
        - Field: "image" (file, required)
        
        Response (200 OK) - Success:
        {
            "status": "success",
            "crop": "Apple",
            "crop_confidence": 0.9234,
            "disease": "Cedar Rust",
            "disease_confidence": 0.8756,
            "treatment": "Apply fungicide spray...",
            "prevention": "Ensure good air circulation...",
            "warnings": [],
            "model_version": "1.0"
        }
        
        Response (200 OK) - Unknown Crop:
        {
            "status": "unknown_crop",
            "crop": "Unknown",
            "crop_confidence": 0.4521,
            "error": "Unknown crop / Not a supported crop. Try a clearer photo or submit for expert review.",
            "warnings": [],
            "model_version": "1.0"
        }
        
        Response (200 OK) - Non-Leaf:
        {
            "status": "non_leaf",
            "error": "Invalid input — please upload a clear leaf image",
            "warnings": [],
            "model_version": "1.0"
        }
        
        Response (400 Bad Request) - Invalid Input:
        {
            "status": "invalid_input",
            "error": "File too large. Maximum 10 MB.",
            "model_version": "1.0"
        }
        
        Response (500 Internal Server Error):
        {
            "status": "system_error",
            "error": "System error: [error details]",
            "model_version": "1.0"
        }
        """
        return {
            "endpoint": "POST /api/predict",
            "description": "Predict crop and disease from leaf image",
            "request": {
                "content_type": "multipart/form-data",
                "fields": {
                    "image": {
                        "type": "file",
                        "required": True,
                        "formats": ["JPG", "PNG"],
                        "max_size_mb": 10
                    }
                }
            },
            "response_200_success": {
                "status": "success",
                "crop": "Apple",
                "crop_confidence": 0.9234,
                "disease": "Cedar Rust",
                "disease_confidence": 0.8756,
                "treatment": "Apply fungicide spray...",
                "prevention": "Ensure good air circulation...",
                "warnings": [],
                "model_version": "1.0"
            },
            "response_200_unknown_crop": {
                "status": "unknown_crop",
                "crop": "Unknown",
                "crop_confidence": 0.4521,
                "error": "Unknown crop / Not a supported crop.",
                "warnings": [],
                "model_version": "1.0"
            },
            "response_200_non_leaf": {
                "status": "non_leaf",
                "error": "Invalid input — please upload a clear leaf image",
                "warnings": [],
                "model_version": "1.0"
            },
            "response_400": {
                "status": "invalid_input",
                "error": "File too large. Maximum 10 MB.",
                "model_version": "1.0"
            },
            "response_500": {
                "status": "system_error",
                "error": "System error: [error details]",
                "model_version": "1.0"
            }
        }
    
    # ============================================================================
    # ENDPOINT 3: Batch Prediction
    # ============================================================================
    
    @staticmethod
    def predict_batch_spec() -> Dict[str, Any]:
        """
        POST /api/predict-batch
        
        Purpose: Predict crop and disease from multiple leaf images
        
        Request:
        - Content-Type: multipart/form-data
        - Field: "images" (multiple files, required)
        
        Response (200 OK):
        {
            "status": "success",
            "results": [
                {
                    "status": "success",
                    "crop": "Apple",
                    "crop_confidence": 0.9234,
                    "disease": "Cedar Rust",
                    "disease_confidence": 0.8756,
                    "treatment": "...",
                    "prevention": "...",
                    "warnings": []
                },
                {
                    "status": "unknown_crop",
                    "crop": "Unknown",
                    "crop_confidence": 0.4521,
                    "error": "Unknown crop...",
                    "warnings": []
                }
            ],
            "total": 2,
            "successful": 1,
            "failed": 1,
            "model_version": "1.0"
        }
        """
        return {
            "endpoint": "POST /api/predict-batch",
            "description": "Predict crop and disease from multiple leaf images",
            "request": {
                "content_type": "multipart/form-data",
                "fields": {
                    "images": {
                        "type": "file[]",
                        "required": True,
                        "formats": ["JPG", "PNG"],
                        "max_size_mb": 10,
                        "max_files": 10
                    }
                }
            },
            "response_200": {
                "status": "success",
                "results": [
                    {
                        "status": "success",
                        "crop": "Apple",
                        "crop_confidence": 0.9234,
                        "disease": "Cedar Rust",
                        "disease_confidence": 0.8756,
                        "treatment": "...",
                        "prevention": "...",
                        "warnings": []
                    }
                ],
                "total": 1,
                "successful": 1,
                "failed": 0,
                "model_version": "1.0"
            }
        }
    
    # ============================================================================
    # ENDPOINT 4: Model Info
    # ============================================================================
    
    @staticmethod
    def model_info_spec() -> Dict[str, Any]:
        """
        GET /api/model-info
        
        Purpose: Get information about loaded models
        
        Response (200 OK):
        {
            "version": "1.0",
            "stage1": {
                "name": "Stage 1: Crop Identification",
                "backbone": "EfficientNet-B0",
                "num_classes": 14,
                "classes": ["Apple", "Blueberry", ...],
                "confidence_threshold": 0.7
            },
            "stage2": {
                "name": "Stage 2: Apple Disease Classification",
                "backbone": "EfficientNet-B0",
                "num_classes": 4,
                "classes": ["Healthy", "Scab", "Black Rot", "Cedar Rust"],
                "confidence_threshold": 0.6
            },
            "device": "cuda",
            "input_size": [224, 224],
            "supported_formats": ["JPG", "PNG"],
            "max_file_size_mb": 10
        }
        """
        return {
            "endpoint": "GET /api/model-info",
            "description": "Get information about loaded models",
            "response_200": {
                "version": "1.0",
                "stage1": {
                    "name": "Stage 1: Crop Identification",
                    "backbone": "EfficientNet-B0",
                    "num_classes": 14,
                    "classes": ["Apple", "Blueberry", "Cherry", "Corn", "Grape", "Peach", "Pepper",
                               "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato", "Wheat"],
                    "confidence_threshold": 0.7
                },
                "stage2": {
                    "name": "Stage 2: Apple Disease Classification",
                    "backbone": "EfficientNet-B0",
                    "num_classes": 4,
                    "classes": ["Healthy", "Scab", "Black Rot", "Cedar Rust"],
                    "confidence_threshold": 0.6
                },
                "device": "cuda",
                "input_size": [224, 224],
                "supported_formats": ["JPG", "PNG"],
                "max_file_size_mb": 10
            }
        }
    
    # ============================================================================
    # ERROR HANDLING SPECIFICATION
    # ============================================================================
    
    @staticmethod
    def error_handling_spec() -> Dict[str, Any]:
        """
        Error Handling Specification
        
        All errors follow this format:
        {
            "status": "error_type",
            "error": "Human-readable error message",
            "model_version": "1.0"
        }
        
        Error Types:
        - invalid_input: File format, size, or integrity issues
        - non_leaf: Image does not contain a valid leaf
        - unknown_crop: Crop not recognized or confidence too low
        - unknown_disease: Disease not recognized or confidence too low
        - system_error: Internal server error
        
        HTTP Status Codes:
        - 200 OK: Prediction completed (check status field for result)
        - 400 Bad Request: Invalid input file
        - 500 Internal Server Error: System error
        """
        return {
            "error_format": {
                "status": "error_type",
                "error": "Human-readable message",
                "model_version": "1.0"
            },
            "error_types": {
                "invalid_input": "File format, size, or integrity issues",
                "non_leaf": "Image does not contain a valid leaf",
                "unknown_crop": "Crop not recognized or confidence too low",
                "unknown_disease": "Disease not recognized or confidence too low",
                "system_error": "Internal server error"
            },
            "http_status_codes": {
                "200": "Prediction completed (check status field)",
                "400": "Invalid input file",
                "500": "System error"
            }
        }
    
    # ============================================================================
    # CONFIDENCE THRESHOLDS (LOCKED)
    # ============================================================================
    
    @staticmethod
    def confidence_thresholds_spec() -> Dict[str, Any]:
        """
        Confidence Threshold Specification (LOCKED - DO NOT CHANGE)
        
        These thresholds are fixed and define the decision boundaries
        for the inference pipeline.
        """
        return {
            "stage1_crop_threshold": {
                "value": 0.7,
                "description": "Minimum confidence for crop prediction",
                "below_threshold": "Return 'unknown_crop' status",
                "locked": True
            },
            "stage2_disease_threshold": {
                "value": 0.6,
                "description": "Minimum confidence for disease prediction",
                "below_threshold": "Return 'unknown_disease' status",
                "locked": True
            },
            "leaf_detection_threshold": {
                "value": 0.15,
                "description": "Minimum green pixel ratio for leaf detection",
                "below_threshold": "Return 'non_leaf' status",
                "locked": True
            }
        }
    
    # ============================================================================
    # ROUTING RULES (LOCKED)
    # ============================================================================
    
    @staticmethod
    def routing_rules_spec() -> Dict[str, Any]:
        """
        Hierarchical Routing Rules (LOCKED - DO NOT CHANGE)
        
        These rules define the exact flow through the inference pipeline.
        """
        return {
            "rule_1_input_validation": {
                "description": "Validate file type, size, and integrity",
                "conditions": [
                    "File must be JPG or PNG",
                    "File size must be <= 10 MB",
                    "File must not be corrupted"
                ],
                "on_failure": "Return invalid_input status",
                "locked": True
            },
            "rule_2_leaf_detection": {
                "description": "Detect if image contains a valid leaf",
                "conditions": [
                    "Green pixel ratio >= 0.15",
                    "Image must have recognizable leaf structure"
                ],
                "on_failure": "Return non_leaf status",
                "on_borderline": "Proceed with warning",
                "locked": True
            },
            "rule_3_stage1_execution": {
                "description": "Always run Stage 1 crop identification",
                "conditions": [
                    "Input validation passed",
                    "Leaf detection passed"
                ],
                "output": "crop_name, crop_confidence",
                "locked": True
            },
            "rule_4_stage1_threshold": {
                "description": "Check Stage 1 confidence threshold",
                "conditions": [
                    "crop_confidence >= 0.7"
                ],
                "on_success": "Proceed to rule 5",
                "on_failure": "Return unknown_crop status",
                "locked": True
            },
            "rule_5_apple_routing": {
                "description": "Route to Stage 2 only if Apple detected",
                "conditions": [
                    "crop_name == 'Apple'",
                    "crop_confidence >= 0.7"
                ],
                "on_true": "Execute Stage 2",
                "on_false": "Return crop prediction only",
                "locked": True
            },
            "rule_6_stage2_execution": {
                "description": "Run Stage 2 apple disease classification",
                "conditions": [
                    "crop_name == 'Apple'",
                    "crop_confidence >= 0.7"
                ],
                "output": "disease_name, disease_confidence",
                "locked": True
            },
            "rule_7_stage2_threshold": {
                "description": "Check Stage 2 confidence threshold",
                "conditions": [
                    "disease_confidence >= 0.6"
                ],
                "on_success": "Proceed to rule 8",
                "on_failure": "Return unknown_disease status",
                "locked": True
            },
            "rule_8_recommendations": {
                "description": "Fetch treatment recommendations",
                "conditions": [
                    "disease_name in ['Healthy', 'Scab', 'Black Rot', 'Cedar Rust']"
                ],
                "output": "treatment, prevention",
                "locked": True
            },
            "rule_9_success_response": {
                "description": "Return successful prediction",
                "output": {
                    "status": "success",
                    "crop": "crop_name",
                    "crop_confidence": "float",
                    "disease": "disease_name",
                    "disease_confidence": "float",
                    "treatment": "string",
                    "prevention": "string",
                    "warnings": "list"
                },
                "locked": True
            }
        }

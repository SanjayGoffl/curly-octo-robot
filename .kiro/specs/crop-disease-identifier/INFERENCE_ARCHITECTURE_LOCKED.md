# Inference Architecture - LOCKED ✅

**Status**: LOCKED - DO NOT MODIFY WITHOUT EXPLICIT APPROVAL  
**Date Locked**: January 10, 2026  
**Version**: 1.0  

---

## Executive Summary

The inference architecture is now **LOCKED** and defines the exact flow for all predictions. This document serves as the single source of truth for:
- Request/response schemas
- Confidence thresholds
- Hierarchical routing rules
- Error handling
- API endpoints

---

## 🔐 LOCKED COMPONENTS

### 1. Confidence Thresholds (IMMUTABLE)

```
Stage 1 Crop Threshold:     0.7 (70%)
Stage 2 Disease Threshold:  0.6 (60%)
Leaf Detection Threshold:   0.15 (15% green pixels)
```

**Why These Values?**
- Stage 1: 0.7 balances false positives vs false negatives for crop identification
- Stage 2: 0.6 is more lenient for disease detection (better to flag uncertain cases)
- Leaf Detection: 0.15 catches most real leaves while rejecting obvious non-leaves

**Change Protocol**: Any threshold change requires:
1. Explicit user approval
2. Retraining evaluation
3. Real-world test set validation
4. Documentation update

---

### 2. Hierarchical Routing Rules (IMMUTABLE)

```
INPUT
  ↓
[Rule 1] Input Validation Gate
  - File type: JPG/PNG only
  - File size: ≤ 10 MB
  - File integrity: Not corrupted
  ↓ (FAIL) → Return: invalid_input
  ↓ (PASS)
[Rule 2] Leaf Detection
  - Green pixel ratio ≥ 0.15
  - Image contains recognizable leaf structure
  ↓ (FAIL) → Return: non_leaf
  ↓ (PASS, borderline) → Add warning, continue
  ↓ (PASS)
[Rule 3] Stage 1 Execution
  - Always run crop identification
  - Output: crop_name, crop_confidence
  ↓
[Rule 4] Stage 1 Threshold Check
  - crop_confidence ≥ 0.7?
  ↓ (FAIL) → Return: unknown_crop
  ↓ (PASS)
[Rule 5] Apple Routing Decision
  - crop_name == "Apple"?
  ↓ (NO) → Return: crop prediction only
  ↓ (YES)
[Rule 6] Stage 2 Execution
  - Run apple disease classification
  - Output: disease_name, disease_confidence
  ↓
[Rule 7] Stage 2 Threshold Check
  - disease_confidence ≥ 0.6?
  ↓ (FAIL) → Return: unknown_disease
  ↓ (PASS)
[Rule 8] Treatment Recommendations
  - Lookup hardcoded recommendations
  - Output: treatment, prevention
  ↓
[Rule 9] Success Response
  - Return: crop, disease, confidence, treatment, prevention
  ↓
OUTPUT
```

**Key Invariants**:
- Stage 2 ONLY executes if crop_name == "Apple" AND crop_confidence ≥ 0.7
- Non-Apple crops return crop prediction only (no disease)
- All errors are caught and returned with appropriate status codes
- No exceptions propagate to user (all wrapped in system_error)

---

### 3. API Endpoints (LOCKED)

#### Endpoint 1: Health Check
```
GET /api/health

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
```

#### Endpoint 2: Single Image Prediction
```
POST /api/predict
Content-Type: multipart/form-data
Field: "image" (file, required)

Response (200 OK - Success):
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

Response (200 OK - Unknown Crop):
{
  "status": "unknown_crop",
  "crop": "Unknown",
  "crop_confidence": 0.4521,
  "error": "Unknown crop / Not a supported crop...",
  "warnings": [],
  "model_version": "1.0"
}

Response (200 OK - Non-Leaf):
{
  "status": "non_leaf",
  "error": "Invalid input — please upload a clear leaf image",
  "warnings": [],
  "model_version": "1.0"
}

Response (400 Bad Request):
{
  "status": "invalid_input",
  "error": "File too large. Maximum 10 MB.",
  "model_version": "1.0"
}

Response (500 Internal Server Error):
{
  "status": "system_error",
  "error": "System error: [details]",
  "model_version": "1.0"
}
```

#### Endpoint 3: Batch Prediction
```
POST /api/predict-batch
Content-Type: multipart/form-data
Field: "images" (multiple files, max 10)

Response (200 OK):
{
  "status": "success",
  "results": [
    { /* PredictionResponse */ },
    { /* PredictionResponse */ }
  ],
  "total": 2,
  "successful": 1,
  "failed": 1,
  "model_version": "1.0"
}
```

#### Endpoint 4: Model Info
```
GET /api/model-info

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
```

#### Endpoint 5: API Contract Documentation
```
GET /api/contract

Response (200 OK):
{
  "health_check": { /* spec */ },
  "predict": { /* spec */ },
  "predict_batch": { /* spec */ },
  "model_info": { /* spec */ },
  "error_handling": { /* spec */ },
  "confidence_thresholds": { /* spec */ },
  "routing_rules": { /* spec */ }
}
```

---

### 4. Error Handling (LOCKED)

All errors follow this format:
```json
{
  "status": "error_type",
  "error": "Human-readable message",
  "model_version": "1.0"
}
```

**Error Types**:
- `invalid_input`: File format, size, or integrity issues
- `non_leaf`: Image does not contain a valid leaf
- `unknown_crop`: Crop not recognized or confidence < 0.7
- `unknown_disease`: Disease not recognized or confidence < 0.6
- `system_error`: Internal server error

**HTTP Status Codes**:
- `200 OK`: Prediction completed (check status field)
- `400 Bad Request`: Invalid input file
- `413 Payload Too Large`: File exceeds 10 MB
- `500 Internal Server Error`: System error
- `503 Service Unavailable`: Models not loaded

---

### 5. Class Mappings (LOCKED)

**Stage 1 - Crop Classes (14 total)**:
```
0: Apple
1: Blueberry
2: Cherry
3: Corn
4: Grape
5: Peach
6: Pepper
7: Potato
8: Raspberry
9: Soybean
10: Squash
11: Strawberry
12: Tomato
13: Wheat
```

**Stage 2 - Apple Disease Classes (4 total)**:
```
0: Healthy
1: Scab
2: Black Rot
3: Cedar Rust
```

---

### 6. Input Constraints (LOCKED)

- **File Formats**: JPG, PNG only
- **File Size**: Maximum 10 MB
- **Image Dimensions**: Resized to 224×224 internally
- **Color Space**: RGB (converted from any format)
- **Batch Size**: Maximum 10 images per batch request

---

### 7. Model Architecture (LOCKED)

**Stage 1**:
- Backbone: EfficientNet-B0 (pretrained)
- Input: 224×224 RGB images
- Output: 14-class softmax probabilities
- Frozen: Base layers (except last 2)
- Loss: CrossEntropyLoss with class weights

**Stage 2**:
- Backbone: EfficientNet-B0 (pretrained)
- Input: 224×224 RGB images
- Output: 4-class softmax probabilities
- Frozen: Base layers (except last 2)
- Loss: CrossEntropyLoss with class weights

---

## 📋 Implementation Files

**Core Inference**:
- `src/inference/inference_pipeline.py` - Main inference orchestrator
- `src/inference/recommendations.py` - Treatment recommendations lookup

**Web API**:
- `src/web/app.py` - Flask application with locked endpoints
- `src/web/api_contract.py` - API contract specification

**Models**:
- `src/models/stage1.py` - Stage 1 model class
- `src/models/stage2.py` - Stage 2 model class

---

## 🚀 Task 5 Implementation Plan

Task 5 will implement:

1. **5.1 Implement inference function for Stage 1**
   - Load trained Stage 1 model
   - Preprocess input image
   - Run prediction and extract confidence score

2. **5.2 Implement inference function for Stage 2**
   - Load trained Stage 2 model
   - Preprocess input image
   - Run prediction and extract confidence score

3. **5.3 Implement confidence threshold logic**
   - Stage 1: T_crop = 0.7 (configurable)
   - Stage 2: T_disease = 0.6 (configurable)
   - Return "Unknown" if confidence below threshold

4. **5.4 Implement hierarchical pipeline orchestration**
   - Run Stage 1 on input image
   - If Stage 1 predicts Apple with confidence ≥ 0.7: run Stage 2
   - Otherwise: return Stage 1 result only

5. **5.5 Write property test for confidence threshold consistency**
   - Generate predictions with various confidence scores
   - Verify predictions below threshold return "Unknown"

6. **5.6 Write property test for Apple-only Stage 2 execution**
   - Generate Stage 1 predictions for all 14 crops
   - Verify Stage 2 executes only for Apple with confidence ≥ 0.7

7. **5.7 Implement input validation gate**
   - Check file type (JPG/PNG only)
   - Check file size (max 10 MB)
   - Validate file integrity

8. **5.8 Implement basic leaf detection**
   - Implement heuristic (green pixel ratio) OR lightweight binary classifier
   - Reject obvious non-leaf images
   - Target: ≥90% rejection accuracy on small non-leaf test set

9. **5.9 Implement borderline handling**
   - If leaf detection is borderline: proceed with inference but mark as low-confidence
   - Display warning: "Image quality may affect accuracy"

10. **5.10 Implement Flask web application**
    - Create REST API endpoints
    - Implement image upload endpoint
    - Implement inference endpoint
    - Create frontend HTML/CSS/JavaScript

---

## ✅ Verification Checklist

Before proceeding to Task 5, verify:

- [ ] Inference pipeline loads both Stage 1 and Stage 2 models
- [ ] Confidence thresholds are correctly applied
- [ ] Hierarchical routing works (Stage 2 only for Apple)
- [ ] All error cases return appropriate status codes
- [ ] API endpoints match contract specification
- [ ] Batch prediction processes multiple images correctly
- [ ] File validation rejects invalid inputs
- [ ] Leaf detection works with heuristic
- [ ] Treatment recommendations are returned for all diseases
- [ ] Model info endpoint returns correct class mappings

---

## 🔒 Lock Status

**LOCKED**: This architecture is now frozen. Any changes require:
1. Explicit user approval
2. Documentation of rationale
3. Impact analysis on Task 5
4. Re-validation of all tests

**Last Updated**: January 10, 2026  
**Locked By**: Kiro AI Agent  
**Status**: ✅ READY FOR TASK 5 IMPLEMENTATION

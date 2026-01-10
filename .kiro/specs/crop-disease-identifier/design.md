# Design Document: AI Crop Disease Identifier (Prototype Scope)

## Overview

The AI Crop Disease Identifier is a hierarchical, two-stage deep learning system designed to identify crop types and classify crop-specific diseases from leaf images. The system architecture prioritizes:

1. **Modularity**: Independent Stage 1 (crop identification) and Stage 2 (disease classification) models
2. **Robustness**: Real-world generalization through augmentation
3. **Scalability**: Easy addition of new crops and crop-specific disease models
4. **Explainability**: Clear decision logic with confidence scores and fallback mechanisms
5. **Prototype-Realistic**: Achievable within prototype scope while maintaining production-aware design

The system is designed as a **prototype with production-aware architecture**, not a full production deployment. This design focuses on core functionality and avoids production-level complexity (calibration, extensive monitoring, complex validation gates).

## Architecture

### System Pipeline

```
User Upload
    ↓
[Input Validation Gate]
  - File type check (JPG/PNG)
  - File size limit (10 MB)
  - Image integrity check
    ↓
[Basic Leaf Detection]
  - Leaf vs non-leaf classification (heuristic or lightweight classifier)
  - Reject if not a valid leaf image
  - Proceed with warning if borderline
    ↓
[Stage 1: Crop Identification Model]
  - Input: Preprocessed leaf image
  - Output: Crop name + confidence score
  - Classes: 14 crop types
    ↓
[Confidence Threshold Check: T_crop = 0.7]
  - If confidence < 0.7 → Return "Unknown crop"
  - If confidence ≥ 0.7 → Continue to Stage 2 (if Apple)
    ↓
[Stage 2: Apple Disease Classification Model]
  - Triggered ONLY if Stage 1 predicts "Apple" with confidence ≥ 0.7
  - Input: Preprocessed leaf image
  - Output: Disease name + confidence score
  - Classes: 4 apple disease classes
    ↓
[Confidence Threshold Check: T_disease = 0.6]
  - If confidence < 0.6 → Return "Unknown disease"
  - If confidence ≥ 0.6 → Return disease + recommendations
    ↓
[Treatment Recommendation Lookup]
  - Hardcoded recommendations for 4 Apple diseases
  - Return treatment and preventive measures
    ↓
User Result Display
```

### Stage 1: Crop Identification Model

**Purpose**: Classify input leaf image into one of 14 crop types

**Architecture**:
- Transfer learning with proven CNN backbone (EfficientNet-B0 or ResNet-50)
- Input: 224×224 RGB images (normalized)
- Output: 14-class softmax probabilities
- Confidence threshold: T_crop = 0.7

**Training Strategy**:
- Freeze base layers initially
- Fine-tune last 2-3 layers on PlantVillage color images
- GPU acceleration (CUDA/cuDNN)
- Early stopping on validation macro F1-score
- Track per-class precision, recall, macro F1
- Apply balanced sampling or class weights to offset crop imbalance (Tomato dominance)

**Data**:
- Source: PlantVillage color images only
- Stratified split: 70% train, 15% val, 15% test
- All 14 crops represented
- Balanced sampling or class weights applied

### Stage 2: Apple Disease Classification Model

**Purpose**: Classify apple leaf images into 4 disease classes

**Architecture**:
- Transfer learning with same CNN backbone as Stage 1 (EfficientNet-B0 or ResNet-50)
- Input: 224×224 RGB images (normalized)
- Output: 4-class softmax probabilities
- Confidence threshold: T_disease = 0.6

**Training Strategy**:
- Freeze base layers initially
- Fine-tune last 2-3 layers on PlantVillage apple images
- GPU acceleration
- Weighted loss function (inverse frequency weighting)
- Oversample minority classes (Scab, Black Rot, Cedar Rust)
- Stronger augmentation for minority classes
- Early stopping on validation macro F1-score
- Special tracking of Cedar Rust recall (target ≥ 0.70, prototype goal)

**Data**:
- Source: PlantVillage color images, Apple class only
- Stratified split: 70% train, 15% val, 15% test
- 4 classes: Apple Healthy, Apple Scab, Apple Black Rot, Apple Cedar Rust
- Class imbalance handling:
  - Weighted loss: weight = 1 / class_frequency
  - Oversampling: minority classes (Scab, Black Rot, Cedar Rust)
  - NO oversampling of Healthy class
  - Stronger augmentation for minority classes

### Input Validation and Pre-Processing Gates

**Gate 1: File Validation**
- Accept only JPG, PNG formats
- File size limit: 10 MB
- Validate file integrity (not corrupted)

**Gate 2: Basic Leaf Detection**
- Heuristic (e.g., green pixel ratio) or lightweight binary classifier
- Reject obvious non-leaf images (soil, tools, hands, animals)
- Target: ≥90% rejection accuracy on small curated non-leaf test set (50-100 images)
- Output: "Invalid input — please upload a clear leaf image"
- If borderline: proceed with inference but mark as low-confidence

### Confidence Thresholds (No Calibration in Prototype)

**Thresholds**:
- Stage 1: T_crop = 0.7 (if confidence < 0.7 → "Unknown crop")
- Stage 2: T_disease = 0.6 (if confidence < 0.6 → "Unknown disease")
- Thresholds are configurable without retraining
- **Note**: Confidence scores are uncalibrated in prototype. Future enhancement: temperature scaling or isotonic regression.

### Fallback and Error Handling

**Unknown Crop (Stage 1 confidence < 0.7)**:
- Return: "Unknown crop / Not a supported crop"
- Confidence score displayed
- Suggested actions: "Try taking a clearer photo" or "Submit for expert review"

**Non-Apple Crop (Stage 1 predicts non-Apple)**:
- Return: "Not an apple leaf"
- Confidence score displayed
- No Stage 2 execution

**Unknown Disease (Stage 2 confidence < 0.6)**:
- Return: "Unknown disease — consider expert review"
- Confidence score displayed

**Non-Leaf Image**:
- Return: "Invalid input — please upload a clear leaf image"
- Reject and ask for retry

**Low-Quality Image (Borderline Leaf Detection)**:
- Proceed with inference but mark as low-confidence
- Display warning: "Image quality may affect accuracy"

## Components and Interfaces

### Component 1: Image Preprocessing

**Input**: Raw uploaded image (JPG/PNG)

**Output**: Normalized 224×224 RGB tensor

**Operations**:
- Resize to 224×224
- Normalize to ImageNet mean/std
- Optional: augmentation (rotation, brightness, contrast, crop)

### Component 2: Stage 1 Model

**Input**: Preprocessed image tensor

**Output**: 14-class probabilities + confidence score

**Interface**:
```
stage1_model.predict(image_tensor) → {
  crop_name: str,
  confidence: float,
  probabilities: dict[crop_name → float]
}
```

### Component 3: Stage 2 Model

**Input**: Preprocessed image tensor (only if Stage 1 predicts Apple)

**Output**: 4-class probabilities + confidence score

**Interface**:
```
stage2_model.predict(image_tensor) → {
  disease_name: str,
  confidence: float,
  probabilities: dict[disease_name → float]
}
```

### Component 4: Treatment Recommendation Engine

**Input**: Disease name

**Output**: Treatment and preventive recommendations

**Interface**:
```
recommendation_engine.get_recommendations(disease_name) → {
  treatment: str,
  prevention: str
}
```

**Implementation**: Hardcoded dictionary for 4 apple diseases

### Component 5: Leaf Detection Module

**Input**: Raw image

**Output**: Leaf detection result (valid/invalid/borderline) + quality metrics

**Interface**:
```
leaf_detector.validate(image) → {
  is_leaf: bool,
  confidence: float,
  warnings: list[str]
}
```

## Data Models

### Input Data Model

```
UploadedImage {
  file_path: str,
  file_size: int,
  file_type: str (jpg|png),
  upload_timestamp: datetime,
  user_id: str (optional)
}
```

### Prediction Result Model

```
PredictionResult {
  crop_name: str,
  crop_confidence: float,
  disease_name: str (optional, only if Apple),
  disease_confidence: float (optional),
  treatment: str (optional),
  prevention: str (optional),
  warnings: list[str],
  model_version: str,
  inference_timestamp: datetime,
  processing_time_ms: float
}
```

### Training Data Model

```
TrainingImage {
  file_path: str,
  crop_label: str,
  disease_label: str,
  split: str (train|val|test),
  augmented: bool
}
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Hierarchical Pipeline Enforcement

**For any** leaf image, if Stage 1 predicts a non-Apple crop with confidence ≥ T_crop, then Stage 2 SHALL NOT execute and the result SHALL contain only the crop prediction.

**Validates: Requirements 1.1, 1.2**

### Property 2: Confidence Threshold Consistency

**For any** prediction, if the confidence score is below the defined threshold (T_crop = 0.7 for Stage 1, T_disease = 0.6 for Stage 2), the system SHALL return an "Unknown" result rather than a specific prediction.

**Validates: Requirements 12.2, 12.4**

### Property 3: Apple-Only Stage 2 Execution

**For any** leaf image, Stage 2 SHALL execute if and only if Stage 1 predicts "Apple" with confidence ≥ T_crop (0.7).

**Validates: Requirements 1.3, 3.1**

### Property 4: Input Validation Gate Enforcement

**For any** uploaded file, if the file is not a valid image (non-JPG/PNG, corrupted, oversized), the system SHALL reject it with an appropriate error message before executing Stage 1.

**Validates: Requirements 11.1, 11.2, 11.6**

### Property 5: Class Imbalance Handling in Stage 2

**For any** training run of Stage 2, the minority classes (Cedar Rust, Black Rot, Scab) SHALL receive stronger augmentation and oversampling compared to the Healthy class, resulting in improved recall for minority classes.

**Validates: Requirements 3.4, 3.5, 3.6**

### Property 6: Real-World Generalization

**For any** real-world leaf image (Google Images, mobile photos) with varied lighting and backgrounds, the model is designed to maintain reasonable accuracy (target: ≥70% for supported crops on representative real-world test set) despite training on controlled backgrounds.

**Validates: Requirements 5.6, 5.7, 15.2**

### Property 7: Treatment Recommendation Consistency

**For any** predicted disease, the system SHALL return consistent, conservative treatment recommendations that do not make prescriptive medical claims and include a "consult agronomist" fallback for severe cases.

**Validates: Requirements 13.2, 13.3, 13.4**

### Property 8: Leaf Detection Accuracy

**For any** non-leaf image (soil, tools, hands, animals), the leaf detection module is designed to correctly identify it as invalid with ≥90% accuracy on small curated non-leaf test set (50-100 images).

**Validates: Requirements 11.4, 11.5**

### Property 9: Cedar Rust Recall Priority

**For any** test set of apple images, the Stage 2 model is designed to target Cedar Rust recall ≥ 0.70, prioritizing detection of this minority disease (prototype goal subject to data availability).

**Validates: Requirements 3.9, 8.2**

### Property 10: Unknown Crop Handling

**For any** image of an unsupported crop, Stage 1 is designed to return "Unknown crop" with confidence < 0.7 at least 70% of the time (prototype validation goal).

**Validates: Requirements 12.3, 12.4**

## Error Handling

### Error Categories

**1. Input Errors**
- Invalid file type → "Invalid file format. Please upload JPG or PNG."
- File too large → "File too large. Maximum 10 MB."
- Corrupted file → "File corrupted. Please try again."
- Non-leaf image → "Invalid input — please upload a clear leaf image."

**2. Prediction Errors**
- Unknown crop → "Unknown crop / Not a supported crop. Try a clearer photo or submit for expert review."
- Unknown disease → "Unknown disease — consider expert review."
- Low confidence → "Prediction confidence low. Please retake photo or consult an expert."

**3. System Errors**
- Model loading failure → "System error. Please try again later."
- GPU unavailable → "Processing error. Please try again."
- Timeout → "Processing took too long. Please try again."

### Error Recovery

- All errors are logged with timestamp and error type
- User receives clear, actionable error messages
- System suggests next steps (retake photo, submit for review, contact support)
- No sensitive information exposed in error messages

## Testing Strategy

### Unit Testing

**Test Coverage**:
- Image preprocessing (resizing, normalization, augmentation)
- Threshold logic (T_crop = 0.7, T_disease = 0.6)
- Treatment recommendation lookup
- Error handling and fallback logic
- Input validation (file type, size, integrity)
- Leaf detection accuracy

**Test Examples**:
- Preprocessing: verify 224×224 output, correct normalization
- Thresholds: verify correct behavior at boundary values
- Recommendations: verify all diseases have recommendations
- Errors: verify correct error messages for each error type

### Property-Based Testing

**Property Tests** (using Hypothesis or similar):

1. **Hierarchical Pipeline Enforcement** (Property 1)
   - Generate random crop predictions with various confidence scores
   - Verify Stage 2 only executes when Stage 1 predicts Apple with confidence ≥ 0.7

2. **Confidence Threshold Consistency** (Property 2)
   - Generate predictions with confidence scores across full range
   - Verify predictions below threshold return "Unknown"

3. **Apple-Only Stage 2 Execution** (Property 3)
   - Generate Stage 1 predictions for all 14 crops
   - Verify Stage 2 executes only for Apple with confidence ≥ 0.7

4. **Input Validation Gate Enforcement** (Property 4)
   - Generate various invalid inputs (non-images, corrupted files, wrong formats)
   - Verify all are rejected before Stage 1

5. **Class Imbalance Handling** (Property 5)
   - Train Stage 2 with and without imbalance handling
   - Verify minority class recall improves with oversampling/augmentation

6. **Real-World Generalization** (Property 6)
   - Test on real-world image set (50-100 Google Images, mobile photos)
   - Verify accuracy ≥ 70% for supported crops

7. **Treatment Recommendation Consistency** (Property 7)
   - Generate all possible disease predictions
   - Verify each returns consistent, conservative recommendations

8. **Leaf Detection Accuracy** (Property 8)
   - Generate non-leaf images (soil, tools, hands, animals)
   - Verify ≥90% rejection accuracy on small test set

9. **Cedar Rust Recall Priority** (Property 9)
   - Test Stage 2 on balanced apple test set
   - Verify Cedar Rust recall ≥ 0.70 (prototype goal)

10. **Unknown Crop Handling** (Property 10)
    - Test Stage 1 on unsupported crops
    - Verify ≥70% return "Unknown crop" with confidence < 0.7

### Integration Testing

- End-to-end pipeline: upload image → validation → Stage 1 → Stage 2 → result
- Verify correct behavior for all crop types
- Verify correct behavior for all apple diseases
- Verify error handling for invalid inputs

### Real-World Robustness Testing

- Test on representative real-world image set (50-100 Google Images, mobile photos)
- Test on images with occlusion (partial leaves)
- Test on images with extreme lighting
- Test on compressed/low-quality images
- Compare results: controlled (PlantVillage) vs real-world

### Evaluation Metrics

**Stage 1 Metrics**:
- Top-1 accuracy (overall and per-class)
- Per-class precision, recall, F1-score
- Macro F1-score (primary metric)
- Confusion matrix

**Stage 2 Metrics**:
- Top-1 accuracy (overall and per-class)
- Per-class precision, recall, F1-score
- Macro F1-score (primary metric)
- Cedar Rust recall (special emphasis)
- Confusion matrix

**System Metrics**:
- Inference latency (target: <1s per image on GPU)
- Leaf detection accuracy (target: ≥90%)

## Deployment Architecture

### Web Application Stack

**Frontend**:
- Simple HTML/CSS/JavaScript interface
- Single image upload form
- Result display with crop, disease, confidence, recommendations
- Error message display

**Backend**:
- REST API (Flask, FastAPI, or Django)
- Image upload endpoint
- Inference endpoint (Stage 1 + Stage 2)
- Result formatting and recommendation lookup

**Model Serving**:
- GPU-accelerated inference (CUDA/cuDNN)
- Model caching in memory

### Inference Flow

1. User uploads image via web form
2. Backend receives image, validates file
3. Backend runs input validation gates (file type, basic leaf detection)
4. Backend runs Stage 1 model
5. If Stage 1 predicts Apple with confidence ≥ 0.7:
   - Backend runs Stage 2 model
6. Backend formats result with recommendations
7. Backend returns result to frontend
8. Frontend displays result to user

### Logging and Monitoring

- Log all predictions with timestamp, model version, confidence scores
- Track per-class recall for both models
- Monitor inference latency
- Store logs for analysis and debugging

## Scalability Considerations

### Adding New Crops

1. Collect training data for new crop
2. Retrain Stage 1 model with new crop class
3. No changes to Stage 2 or inference pipeline
4. Deploy updated Stage 1 model

### Adding New Crop-Specific Disease Models

1. Collect training data for new crop's diseases
2. Train new Stage 2 model (same architecture as Apple model)
3. Add conditional logic: if Stage 1 predicts new crop → run new Stage 2 model
4. No changes to Stage 1 or other Stage 2 models
5. Deploy new Stage 2 model

### Example: Adding Tomato Disease Classification

```
if stage1_prediction == "Tomato" and confidence >= 0.7:
    tomato_disease_result = stage2_tomato_model.predict(image)
    return format_result(stage1_prediction, tomato_disease_result)
```

This modular design allows independent development and deployment of new models.

## Future Enhancements (Out of Prototype Scope)

- Confidence calibration (temperature scaling, isotonic regression)
- Multiple leaf detection and per-leaf analysis
- Advanced data drift monitoring and automated alerts
- Perceptual hashing for data deduplication
- EXIF sanitization and formal privacy policy
- Extensive real-world dataset curation (700+ images)
- User feedback collection and continuous learning pipeline

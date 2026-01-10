# Implementation Plan: AI Crop Disease Identifier

## Overview

This implementation plan breaks down the AI Crop Disease Identifier into discrete, testable tasks. The plan follows a logical progression: data preparation → model training → evaluation → web application → documentation. Each task builds on previous tasks with no orphaned code. Tasks marked with `*` are optional and can be skipped for faster MVP, but are recommended for comprehensive validation.

## Tasks

- [x] 1. Project Setup and Environment Configuration
  - Set up Python environment with required dependencies (PyTorch/TensorFlow, Flask, etc.)
  - Configure GPU acceleration (CUDA/cuDNN verification)
  - Create project directory structure
  - Set up version control (Git) and documentation
  - _Requirements: 10.1, 10.2_

- [x] 2. Data Loading and Preprocessing Pipeline
  - [x] 2.1 Download and organize PlantVillage dataset (color images only)
    - Verify dataset structure and image counts
    - Confirm 14 crop types and 4 apple disease classes
    - _Requirements: 4.1, 4.2_

  - [x] 2.2 Implement stratified data splitting (70/15/15)
    - Create train/val/test splits maintaining class distribution
    - Ensure no data leakage between splits
    - Verify split sizes and class balance
    - _Requirements: 4.2, 4.3, 4.5_

  - [x] 2.3 Implement image preprocessing and augmentation
    - Resize images to 224×224
    - Normalize to ImageNet mean/std
    - Implement augmentation: rotation, brightness, contrast, random crop
    - Apply stronger augmentation to minority classes in Stage 2
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 2.4 Write unit tests for data pipeline
    - Test stratified split correctness
    - Test preprocessing output shapes and normalization
    - Test augmentation application
    - _Requirements: 4.2, 4.5_

- [ ] 3. Stage 1: Crop Identification Model Training
  - [ ] 3.1 Implement Stage 1 model architecture
    - Load pre-trained CNN backbone (EfficientNet-B0 or ResNet-50)
    - Freeze base layers
    - Add classification head for 14 crops
    - _Requirements: 2.3, 2.5, 2.6_

  - [ ] 3.2 Implement balanced sampling or class weighting for Stage 1
    - Apply inverse frequency weighting to offset crop imbalance
    - OR implement balanced batch sampling
    - Track per-crop recall during training
    - _Requirements: 2.7_

  - [ ] 3.3 Implement Stage 1 training loop
    - GPU-accelerated training with early stopping
    - Track loss, accuracy, per-class precision/recall, macro F1-score
    - Save best checkpoint based on validation macro F1-score
    - _Requirements: 10.1, 10.3, 10.5, 10.6, 10.7_

  - [ ] 3.4 Evaluate Stage 1 on validation and test sets
    - Generate confusion matrix
    - Report per-class precision, recall, F1-score
    - Report macro F1-score (primary metric)
    - Document per-crop accuracy
    - _Requirements: 8.1, 8.2, 8.6_

  - [ ]* 3.5 Write property test for Stage 1 hierarchical enforcement
    - **Property 1: Hierarchical Pipeline Enforcement**
    - **Validates: Requirements 1.1, 1.2**
    - Generate random crop predictions and verify Stage 2 only executes for Apple
    - _Requirements: 1.1, 1.2_

- [ ] 4. Stage 2: Apple Disease Classification Model Training
  - [ ] 4.1 Implement Stage 2 model architecture
    - Load same pre-trained CNN backbone as Stage 1
    - Freeze base layers
    - Add classification head for 4 apple diseases
    - _Requirements: 3.1, 3.7_

  - [ ] 4.2 Implement class imbalance handling for Stage 2
    - Implement weighted loss (inverse frequency weighting)
    - Implement oversampling for minority classes (Scab, Black Rot, Cedar Rust)
    - Ensure Healthy class is NOT oversampled
    - Apply stronger augmentation to minority classes
    - _Requirements: 3.3, 3.4, 3.5, 3.6_

  - [ ] 4.3 Implement Stage 2 training loop
    - GPU-accelerated training with early stopping
    - Track loss, accuracy, per-class precision/recall, macro F1-score
    - Special tracking of Cedar Rust recall
    - Save best checkpoint based on validation macro F1-score
    - _Requirements: 10.1, 10.3, 10.5, 10.6, 10.7_

  - [ ] 4.4 Evaluate Stage 2 on validation and test sets
    - Generate confusion matrix
    - Report per-class precision, recall, F1-score
    - Report macro F1-score (primary metric)
    - Report Cedar Rust recall (target ≥0.70)
    - _Requirements: 8.1, 8.2, 8.6, 8.8_

  - [ ]* 4.5 Write property test for class imbalance handling
    - **Property 5: Class Imbalance Handling in Stage 2**
    - **Validates: Requirements 3.4, 3.5, 3.6**
    - Verify minority class recall improves with oversampling/augmentation
    - _Requirements: 3.4, 3.5, 3.6_

  - [ ]* 4.6 Write property test for Cedar Rust recall priority
    - **Property 9: Cedar Rust Recall Priority**
    - **Validates: Requirements 3.9, 8.2**
    - Test Stage 2 on balanced apple test set
    - Verify Cedar Rust recall ≥0.70 (prototype goal)
    - _Requirements: 3.9, 8.2_

- [ ] 5. Inference Pipeline and Confidence Thresholds
  - [ ] 5.1 Implement inference function for Stage 1
    - Load trained Stage 1 model
    - Preprocess input image
    - Run prediction and extract confidence score
    - _Requirements: 6.1, 2.1, 2.2_

  - [ ] 5.2 Implement inference function for Stage 2
    - Load trained Stage 2 model
    - Preprocess input image
    - Run prediction and extract confidence score
    - _Requirements: 6.3, 3.1, 3.2_

  - [ ] 5.3 Implement confidence threshold logic
    - Stage 1: T_crop = 0.7 (configurable)
    - Stage 2: T_disease = 0.6 (configurable)
    - Return "Unknown" if confidence below threshold
    - _Requirements: 12.2, 12.4_

  - [ ] 5.4 Implement hierarchical pipeline orchestration
    - Run Stage 1 on input image
    - If Stage 1 predicts Apple with confidence ≥ 0.7: run Stage 2
    - Otherwise: return Stage 1 result only
    - _Requirements: 1.1, 1.2, 1.3_

  - [ ]* 5.5 Write property test for confidence threshold consistency
    - **Property 2: Confidence Threshold Consistency**
    - **Validates: Requirements 12.2, 12.4**
    - Generate predictions with various confidence scores
    - Verify predictions below threshold return "Unknown"
    - _Requirements: 12.2, 12.4_

  - [ ]* 5.6 Write property test for Apple-only Stage 2 execution
    - **Property 3: Apple-Only Stage 2 Execution**
    - **Validates: Requirements 1.3, 3.1**
    - Generate Stage 1 predictions for all 14 crops
    - Verify Stage 2 executes only for Apple with confidence ≥ 0.7
    - _Requirements: 1.3, 3.1_

- [ ] 6. Input Validation and Leaf Detection
  - [ ] 6.1 Implement file validation gate
    - Check file type (JPG/PNG only)
    - Check file size (max 10 MB)
    - Validate file integrity
    - _Requirements: 11.1, 11.2, 11.6_

  - [ ] 6.2 Implement basic leaf detection
    - Implement heuristic (green pixel ratio) OR lightweight binary classifier
    - Reject obvious non-leaf images
    - Target: ≥90% rejection accuracy on small non-leaf test set
    - _Requirements: 11.4, 11.5_

  - [ ] 6.3 Implement borderline handling
    - If leaf detection is borderline: proceed with inference but mark as low-confidence
    - Display warning: "Image quality may affect accuracy"
    - _Requirements: 11.7_

  - [ ]* 6.4 Write property test for input validation gate enforcement
    - **Property 4: Input Validation Gate Enforcement**
    - **Validates: Requirements 11.1, 11.2, 11.6**
    - Generate various invalid inputs (non-images, corrupted files, wrong formats)
    - Verify all are rejected before Stage 1
    - _Requirements: 11.1, 11.2, 11.6_

  - [ ]* 6.5 Write property test for leaf detection accuracy
    - **Property 8: Leaf Detection Accuracy**
    - **Validates: Requirements 11.4, 11.5**
    - Generate non-leaf images (soil, tools, hands, animals)
    - Verify ≥90% rejection accuracy on small test set
    - _Requirements: 11.4, 11.5_

- [ ] 7. Treatment Recommendations Engine
  - [ ] 7.1 Create hardcoded treatment recommendation database
    - Map 4 apple diseases to treatment/prevention strings
    - Include conservative guidance without prescriptive medical claims
    - Include "consult agronomist" fallback for severe cases
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ] 7.2 Implement recommendation lookup function
    - Input: disease name
    - Output: treatment and prevention strings
    - _Requirements: 13.1, 13.2_

  - [ ]* 7.3 Write property test for treatment recommendation consistency
    - **Property 7: Treatment Recommendation Consistency**
    - **Validates: Requirements 13.2, 13.3, 13.4**
    - Generate all possible disease predictions
    - Verify each returns consistent, conservative recommendations
    - _Requirements: 13.2, 13.3, 13.4_

- [ ] 8. Error Handling and Fallback Logic
  - [ ] 8.1 Implement error handling for all error categories
    - Input errors (invalid file type, file too large, corrupted)
    - Prediction errors (unknown crop, unknown disease, low confidence)
    - System errors (model loading failure, GPU unavailable, timeout)
    - _Requirements: 6.6, 11.1, 11.2, 12.1, 12.4_

  - [ ] 8.2 Implement user-friendly error messages
    - Clear, actionable error messages
    - Suggested next steps (retake photo, submit for review)
    - No sensitive information exposed
    - _Requirements: 6.6, 11.1, 11.2, 12.1, 12.4_

  - [ ] 8.3 Implement fallback logic for edge cases
    - Unknown crop: suggest retaking photo or expert review
    - Unknown disease: suggest expert review
    - Low confidence: suggest retaking photo or consulting expert
    - _Requirements: 12.1, 12.4_

- [ ] 9. Logging and Model Versioning
  - [ ] 9.1 Implement basic logging
    - Log all predictions with timestamp, model version, crop/disease, confidence scores
    - Log errors with timestamp and error type
    - _Requirements: 14.1, 14.2_

  - [ ] 9.2 Implement model versioning
    - Store model version in predictions
    - Log dataset version used for training
    - _Requirements: 14.3, 14.4_

  - [ ] 9.3 Set up logging infrastructure
    - Create logs directory
    - Configure log file rotation
    - _Requirements: 14.1, 14.2_

- [ ] 10. Web Application Interface
  - [ ] 10.1 Set up Flask/FastAPI backend
    - Create REST API endpoints
    - Implement image upload endpoint
    - Implement inference endpoint
    - _Requirements: 7.1, 7.2_

  - [ ] 10.2 Implement image upload and preprocessing
    - Accept single image upload
    - Validate file type and size
    - Preprocess image for inference
    - _Requirements: 7.1, 11.1, 11.2_

  - [ ] 10.3 Implement inference endpoint
    - Call Stage 1 model
    - Call Stage 2 model if applicable
    - Format result with recommendations
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ] 10.4 Create frontend HTML/CSS/JavaScript
    - Single image upload form
    - Result display (crop, disease, confidence, recommendations)
    - Error message display
    - Visual feedback during processing
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.6_

  - [ ] 10.5 Implement result formatting and display
    - Format predictions as user-friendly text
    - Display confidence as percentage
    - Display treatment and prevention recommendations
    - _Requirements: 6.4, 6.5, 7.2_

  - [ ]* 10.6 Write integration tests for web application
    - Test image upload endpoint
    - Test inference endpoint
    - Test error handling
    - Test end-to-end pipeline (upload → result)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 11. Real-World Robustness Testing
  - [ ] 11.1 Curate representative real-world test set
    - Collect 50-100 Google Images and/or mobile camera photos
    - Include 5-10 images per crop (70 total)
    - Include 20-30 images per apple disease class
    - Include images with occlusion, varied lighting, compression
    - _Requirements: 15.1, 15.7_

  - [ ] 11.2 Evaluate on real-world test set
    - Run inference on all real-world images
    - Calculate accuracy for supported crops
    - Compare with controlled (PlantVillage) test results
    - Document performance differences
    - _Requirements: 15.2, 15.6_

  - [ ]* 11.3 Write property test for real-world generalization
    - **Property 6: Real-World Generalization**
    - **Validates: Requirements 5.6, 5.7, 15.2**
    - Test on real-world image set (50-100 images)
    - Verify accuracy ≥70% for supported crops
    - _Requirements: 5.6, 5.7, 15.2_

  - [ ]* 11.4 Write property test for unknown crop handling
    - **Property 10: Unknown Crop Handling**
    - **Validates: Requirements 12.3, 12.4**
    - Test Stage 1 on unsupported crops
    - Verify ≥70% return "Unknown crop" with confidence < 0.7
    - _Requirements: 12.3, 12.4_

- [ ] 12. Evaluation and Reporting
  - [ ] 12.1 Generate confusion matrices for Stage 1 and Stage 2
    - Create confusion matrix visualizations
    - Save as images for documentation
    - _Requirements: 8.1_

  - [ ] 12.2 Generate per-class metrics report
    - Per-class precision, recall, F1-score for both stages
    - Macro F1-score (primary metric)
    - Cedar Rust recall (special emphasis)
    - _Requirements: 8.2, 8.6_

  - [ ] 12.3 Create evaluation report
    - Explain why accuracy is misleading due to class imbalance
    - Compare controlled vs real-world test results
    - Document model limitations and failure modes
    - _Requirements: 8.3, 8.4, 8.5, 8.7_

  - [ ] 12.4 Verify all success criteria
    - Stage 1 accuracy ≥85% on PlantVillage test set
    - Stage 2 macro F1-score ≥75% on Apple test set
    - End-to-end pipeline latency <2s on GPU
    - Real-world accuracy ≥70% on 50-100 images
    - All 10 properties verified through testing
    - Web interface functional with error handling
    - _Requirements: Success Criteria_

- [ ] 13. Documentation and Presentation
  - [ ] 13.1 Create architecture documentation
    - System pipeline diagram (text-based or visual)
    - Component descriptions
    - Data flow explanation
    - _Requirements: 9.1, 9.4_

  - [ ] 13.2 Create testing documentation
    - Unit test coverage summary
    - Property-based test descriptions
    - Integration test results
    - Real-world test results
    - _Requirements: 8.1, 8.2, 8.6_

  - [ ] 13.3 Create limitations and future work document
    - Document model limitations
    - Explain failure modes
    - List future enhancements (calibration, multiple leaf detection, etc.)
    - _Requirements: 8.7, 9.5_

  - [ ] 13.4 Create README with setup and usage instructions
    - Environment setup (Python, GPU, dependencies)
    - Dataset download and preparation
    - Model training instructions
    - Web application usage
    - Evaluation and testing instructions
    - _Requirements: 9.1, 9.4_

  - [ ] 13.5 Create presentation materials
    - System architecture explanation
    - Key results and metrics
    - Real-world robustness demonstration
    - Scalability explanation
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 14. Final Checkpoint - Ensure All Tests Pass
  - Ensure all unit tests pass
  - Ensure all property-based tests pass
  - Ensure all integration tests pass
  - Verify all success criteria met
  - Ask the user if questions arise
  - _Requirements: Success Criteria_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP, but are recommended for comprehensive validation
- Each task references specific requirements for traceability
- Property-based tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- All code should be modular and testable
- GPU acceleration is required for model training
- Real-world test set should be curated carefully for meaningful evaluation

# Requirements Document: AI Crop Disease Identifier (Prototype Scope)

## Introduction

The AI Crop Disease Identifier is a hierarchical, two-stage AI system designed to identify crop types from leaf images and classify diseases specific to each crop. This is a **prototype system with production-aware design**, not a production deployment. The system prioritizes real-world robustness by generalizing beyond controlled training data to handle Google images, mobile photos, and varied lighting conditions. The architecture uses a two-stage pipeline: Stage 1 identifies the crop type, and Stage 2 (triggered only for Apple crops) classifies specific apple diseases. Input validation is a pre-processing gate, not an additional stage. This hierarchical approach prevents cross-crop disease confusion and enables scalable addition of new crops and disease models.

## Glossary

- **Crop**: A type of plant (e.g., Apple, Tomato, Potato) identified from a leaf image
- **Disease**: A specific condition affecting a crop (e.g., Apple Scab, Cedar Rust)
- **Stage 1 Model**: Crop identification model that classifies input images into one of 14 crop types
- **Stage 2 Model**: Apple-specific disease classification model triggered only when Stage 1 predicts Apple
- **Confidence Score**: Probability output from the model indicating prediction certainty (0-1 range)
- **Class Imbalance**: Unequal distribution of samples across classes (e.g., Cedar Rust is minority in Apple classes)
- **Weighted Loss**: Loss function that assigns higher penalties to minority class misclassifications
- **Stratified Split**: Data partitioning that maintains class distribution across train/val/test sets
- **Transfer Learning**: Using pre-trained CNN backbone and fine-tuning for specific task
- **Real-World Robustness**: Model's ability to generalize to uncontrolled images (Google photos, mobile captures)
- **Augmentation**: Synthetic data transformations (rotation, lighting, scale) to improve generalization
- **Macro F1-Score**: Average F1-score across all classes, treating each class equally regardless of frequency

## Requirements

### Requirement 1: Hierarchical Two-Stage Classification Architecture

**User Story:** As a system architect, I want a hierarchical, multi-stage AI pipeline, so that crop identification precedes disease classification, improving robustness and preventing cross-crop disease confusion.

#### Acceptance Criteria

1. THE System SHALL implement exactly two sequential stages: Stage 1 (Crop Identification) and Stage 2 (Apple Disease Classification)
2. WHEN Stage 1 predicts a non-Apple crop, THE System SHALL NOT execute Stage 2 and SHALL return only the crop prediction with confidence
3. WHEN Stage 1 predicts Apple with confidence ≥ T_crop (configured default: 0.7), THE System SHALL execute Stage 2 to classify the specific apple disease
4. THE System SHALL maintain separate, independently trainable models for Stage 1 and Stage 2
5. THE System architecture SHALL allow addition of new crop-specific Stage 2 models without retraining Stage 1

### Requirement 2: Stage 1 – Crop Identification Model

**User Story:** As a farmer, I want the system to first identify what crop the leaf belongs to, so that I can be confident the disease classification is relevant to my crop.

#### Acceptance Criteria

1. THE Stage 1 Model SHALL classify input leaf images into one of 14 crop types
2. WHEN a leaf image is provided, THE Stage 1 Model SHALL output a crop name and confidence score
3. THE Stage 1 Model SHALL use transfer learning with a proven CNN backbone (EfficientNet-B0 or ResNet-50)
4. THE Stage 1 Model SHALL be trained on color images from the PlantVillage dataset
5. THE Stage 1 Model SHALL use GPU acceleration during training
6. THE Stage 1 Model SHALL freeze base layers initially and perform light fine-tuning
7. THE Stage 1 Model SHALL apply balanced sampling or class weights to offset crop imbalance
8. THE Stage 1 Model SHALL track per-class precision, recall, and macro F1-score during evaluation

### Requirement 3: Stage 2 – Apple Disease Classification Model

**User Story:** As a farmer with apple trees, I want detailed disease classification for apple leaves, so that I can identify the specific disease and apply targeted treatment.

#### Acceptance Criteria

1. THE Stage 2 Model SHALL classify apple leaf images into exactly 4 classes: Apple Healthy, Apple Scab, Apple Black Rot, Apple Cedar Rust
2. WHEN Stage 1 predicts Apple, THE Stage 2 Model SHALL execute and output disease name and confidence score
3. THE Stage 2 Model SHALL use weighted loss function to handle class imbalance (inverse frequency weighting)
4. THE Stage 2 Model SHALL oversample only minority apple classes (Scab, Black Rot, Cedar Rust) during training
5. THE Stage 2 Model SHALL NOT oversample the Healthy class
6. THE Stage 2 Model SHALL apply stronger augmentation to minority classes
7. THE Stage 2 Model SHALL use transfer learning with a proven CNN backbone
8. THE Stage 2 Model SHALL track per-class precision, recall, and macro F1-score, with special attention to Cedar Rust recall
9. THE Stage 2 Model is designed to target Cedar Rust recall ≥ 0.70 (prototype goal subject to data availability)

### Requirement 4: Data Handling and Stratified Splitting

**User Story:** As a data scientist, I want proper data stratification and class-balanced test sets, so that evaluation metrics accurately reflect model performance across all classes.

#### Acceptance Criteria

1. THE System SHALL use only color images from the PlantVillage dataset for training
2. THE System SHALL perform manual stratified split during data loading: 70% training, 15% validation, 15% test
3. THE test set SHALL be class-balanced, not random
4. THE System SHALL ignore grayscale and segmented image versions during training
5. THE System SHALL maintain class distribution proportions in training and validation sets
6. WHEN loading data, THE System SHALL ensure no data leakage between train/val/test splits

### Requirement 5: Real-World Robustness Through Augmentation

**User Story:** As a farmer using mobile devices and Google images, I want the model to work with my real-world photos, not just controlled lab images, so that the system is practical for field use.

#### Acceptance Criteria

1. THE System SHALL apply augmentation to simulate lighting variations (brightness, contrast adjustments)
2. THE System SHALL apply augmentation to simulate partial leaves (random cropping)
3. THE System SHALL apply augmentation to simulate rotation and scale variations
4. THE System SHALL apply stronger augmentation to minority classes to improve generalization
5. WHEN a non-apple leaf image is provided, THE System SHALL reject it with high confidence
6. THE System is designed to generalize to Google images and mobile camera photos despite training on controlled backgrounds
7. THE System is designed to maintain reasonable performance on real-world images with varied lighting and backgrounds

### Requirement 6: Inference Logic and User Interaction

**User Story:** As a farmer, I want to upload a single leaf image and receive clear, actionable results, so that I can quickly identify and treat crop diseases.

#### Acceptance Criteria

1. WHEN a user uploads a single leaf image, THE System SHALL execute Stage 1 crop identification
2. IF Stage 1 predicts crop ≠ Apple, THE System SHALL return "Not an apple leaf" with confidence score
3. IF Stage 1 predicts Apple, THE System SHALL execute Stage 2 and return disease name, confidence score, and treatment recommendations
4. THE System SHALL display prediction confidence as a percentage or probability score
5. THE System SHALL provide basic treatment and preventive recommendations based on the predicted disease
6. THE System SHALL reject invalid inputs (non-image files, corrupted images) with clear error messages

### Requirement 7: Web Application Interface

**User Story:** As a non-technical farmer, I want a simple, intuitive web interface, so that I can easily upload images and understand results without technical knowledge.

#### Acceptance Criteria

1. THE Web Application SHALL provide a single image upload interface
2. THE Web Application SHALL display results clearly showing: Crop, Disease, Confidence, and Recommendations
3. THE Web Application SHALL be designed for non-technical users with clear, actionable language
4. THE Web Application SHALL NOT support batch processing (single image per upload)
5. THE Web Application SHALL provide visual feedback during image processing
6. THE Web Application SHALL handle errors gracefully with user-friendly messages

### Requirement 8: Evaluation and Reporting

**User Story:** As a project stakeholder, I want comprehensive evaluation metrics and analysis, so that I can understand model performance, limitations, and real-world applicability.

#### Acceptance Criteria

1. THE System SHALL generate confusion matrices for both Stage 1 and Stage 2 models
2. THE System SHALL report per-class recall for all classes, with special emphasis on Cedar Rust recall
3. THE System SHALL report macro F1-score as the primary evaluation metric (not accuracy alone)
4. THE System SHALL explain why accuracy is misleading due to class imbalance
5. THE System SHALL compare controlled test results (PlantVillage) vs real-world sample results
6. THE System SHALL provide per-class precision and recall for detailed performance analysis
7. THE System SHALL document model limitations and failure modes

### Requirement 9: Explainability and Scalability

**User Story:** As a project presenter, I want to explain the system architecture clearly, so that judges and stakeholders understand the design rationale and scalability benefits.

#### Acceptance Criteria

1. THE System architecture SHALL be explainable as: "A hierarchical AI pipeline where crop identification precedes crop-specific disease classification, improving robustness, accuracy, and real-world usability"
2. THE System design SHALL allow easy addition of new crops without retraining Stage 1
3. THE System design SHALL allow easy addition of new crop-specific Stage 2 models without retraining existing models
4. THE System SHALL document the hierarchical design rationale and scalability approach
5. THE System SHALL be presented as an AI system architecture, not a single flat classifier

### Requirement 10: Model Training and Optimization

**User Story:** As a machine learning engineer, I want efficient, GPU-accelerated training with proper regularization, so that the model trains quickly and generalizes well to real-world data.

#### Acceptance Criteria

1. THE System SHALL use GPU acceleration for all model training
2. THE System SHALL NOT train models from scratch; transfer learning is mandatory
3. THE System SHALL freeze base layers initially and perform light fine-tuning
4. THE System SHALL use weighted loss (inverse frequency) to handle class imbalance in Stage 2
5. THE System SHALL track training metrics (loss, accuracy, F1-score) during training
6. THE System SHALL implement early stopping to prevent overfitting
7. THE System SHALL save the best model checkpoint based on validation macro F1-score

### Requirement 11: Input Validation and Basic Quality Checks

**User Story:** As a farmer, I want the system to reject obviously invalid inputs, so that I don't waste time on non-leaf images.

#### Acceptance Criteria

1. WHEN a non-image file is uploaded, THE System SHALL return "Invalid file format. Please upload JPG or PNG."
2. WHEN a file exceeds size limit (10 MB), THE System SHALL return "File too large. Maximum 10 MB."
3. THE System SHALL implement basic leaf detection (heuristic or lightweight classifier) to validate input
4. WHEN a non-leaf image is detected, THE System SHALL return "Invalid input — please upload a clear leaf image"
5. THE System is designed to target ≥90% rejection accuracy on small curated non-leaf test set (50-100 images)
6. THE System SHALL reject invalid file types (only accept JPG, PNG)
7. IF leaf detection is borderline, THE System SHALL proceed with inference but mark result as low-confidence

### Requirement 12: Unknown Crop Handling

**User Story:** As a farmer, I want the system to recognize when an image is not one of the 14 supported crops, so that I don't receive incorrect disease predictions for unsupported plants.

#### Acceptance Criteria

1. WHEN Stage 1 predicts a crop with confidence < 0.7, THE System SHALL return "Unknown crop / Not a supported crop" with confidence score
2. THE System is designed to target ≥85% accuracy on supported crops (PlantVillage test set)
3. THE System is designed to target ≥70% rejection of obvious non-crops (prototype validation)
4. WHEN an unknown crop is detected, THE System SHALL suggest user actions: "Try taking a clearer photo of a single leaf" or "Submit for expert review"
5. THE System SHALL document the confidence threshold (T_crop = 0.7) and allow configuration

### Requirement 13: Treatment Recommendations

**User Story:** As a farmer, I want practical treatment guidance for identified diseases, so that I can take immediate action.

#### Acceptance Criteria

1. WHEN a disease is predicted, THE System SHALL return basic treatment and preventive recommendations
2. THE System SHALL provide conservative treatment guidance without prescriptive medical claims
3. THE System SHALL include a "consult agronomist" fallback for severe cases
4. THE System is designed to provide hardcoded recommendations for the 4 apple disease classes
5. THE System SHALL display recommendations in clear, farmer-friendly language

### Requirement 14: Basic Logging and Model Versioning

**User Story:** As a system operator, I want basic logging and version tracking, so that I can debug issues and track model changes.

#### Acceptance Criteria

1. THE System SHALL log all predictions with: timestamp, model version, crop prediction, disease prediction, and confidence scores
2. THE System SHALL maintain model version information for reproducibility
3. THE System SHALL log dataset version used for training each model
4. THE System is designed to support future enhancements: data drift monitoring, automated alerts, and detailed audit logs

### Requirement 15: Real-World Robustness Testing

**User Story:** As a project stakeholder, I want the system tested on real-world images, so that I can be confident it works beyond controlled lab conditions.

#### Acceptance Criteria

1. THE System SHALL be evaluated on a representative real-world test set (50-100 Google Images and/or mobile camera photos)
2. THE System is designed to target ≥70% top-1 accuracy on real-world images for supported crops (prototype goal — to be validated on representative real-world test set)
3. THE System SHALL be tested on images with occlusion (partial leaves)
4. THE System SHALL be tested on images with varied lighting (brightness/contrast variations)
5. THE System SHALL be tested on compressed and low-quality images
6. THE System SHALL provide evaluation reports comparing controlled (PlantVillage) vs real-world test results
7. THE System is designed to use representative real-world test set composition: 5-10 images per crop (70 total), Apple balanced (20-30 per class), Google/mobile robustness (50-100 curated images)


## Prototype Success Criteria

This prototype is considered successful if:

1. Stage 1 achieves ≥85% accuracy on PlantVillage test set
2. Stage 2 achieves ≥75% macro F1-score on Apple test set
3. End-to-end pipeline works (upload → result in <2s on GPU)
4. Basic real-world testing shows ≥70% accuracy on 50-100 real-world images
5. All 10 correctness properties verified through testing
6. Web interface functional with clear error handling
7. Documentation complete (architecture, testing, limitations)

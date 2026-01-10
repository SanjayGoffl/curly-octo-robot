# Task 5 Quick Start Guide

**Task**: Inference Pipeline and Confidence Thresholds  
**Status**: Ready to Start  
**Locked Architecture**: ✅ YES  

---

## What's Already Done ✅

1. **Inference Pipeline Architecture** - LOCKED
   - `src/inference/inference_pipeline.py` - Complete orchestrator
   - Handles all routing rules
   - Implements all error cases

2. **API Contract** - LOCKED
   - `src/web/api_contract.py` - Full specification
   - 5 endpoints defined
   - Request/response schemas locked

3. **Flask Web Application** - LOCKED
   - `src/web/app.py` - All endpoints implemented
   - Error handlers configured
   - Batch processing ready

4. **Confidence Thresholds** - LOCKED
   - Stage 1: 0.7
   - Stage 2: 0.6
   - Leaf Detection: 0.15

---

## What Task 5 Needs to Implement

### Sub-Task 5.1: Stage 1 Inference Function
**File**: `src/inference/inference_pipeline.py` (already has `predict_stage1()`)

```python
def predict_stage1(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
    """Run Stage 1 crop identification."""
    # Already implemented ✅
```

**Status**: ✅ DONE

---

### Sub-Task 5.2: Stage 2 Inference Function
**File**: `src/inference/inference_pipeline.py` (already has `predict_stage2()`)

```python
def predict_stage2(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
    """Run Stage 2 apple disease classification."""
    # Already implemented ✅
```

**Status**: ✅ DONE

---

### Sub-Task 5.3: Confidence Threshold Logic
**File**: `src/inference/inference_pipeline.py` (already in `predict()`)

```python
# Stage 1 threshold check
if crop_confidence < self.crop_threshold:
    return PredictionResult(status=PredictionStatus.UNKNOWN_CROP, ...)

# Stage 2 threshold check
if disease_confidence < self.disease_threshold:
    return PredictionResult(status=PredictionStatus.UNKNOWN_DISEASE, ...)
```

**Status**: ✅ DONE

---

### Sub-Task 5.4: Hierarchical Pipeline Orchestration
**File**: `src/inference/inference_pipeline.py` (already in `predict()`)

```python
# Stage 1 always runs
crop_name, crop_confidence = self.predict_stage1(image_tensor)

# Stage 2 only if Apple
if crop_name == "Apple":
    disease_name, disease_confidence = self.predict_stage2(image_tensor)
else:
    return PredictionResult(crop_name=crop_name, ...)
```

**Status**: ✅ DONE

---

### Sub-Task 5.5: Property Test - Confidence Threshold Consistency
**File**: `tests/test_inference_pipeline.py` (NEW - needs to be created)

**What to Test**:
- Generate predictions with confidence scores 0.0 to 1.0
- Verify predictions below 0.7 return "unknown_crop"
- Verify predictions below 0.6 return "unknown_disease"
- Verify predictions above thresholds return success

**Example**:
```python
def test_confidence_threshold_consistency():
    """Property: Predictions below threshold return Unknown"""
    # For all confidence scores < 0.7:
    #   predict_stage1() should return unknown_crop
    # For all confidence scores >= 0.7:
    #   predict_stage1() should return crop name
```

**Status**: ⏳ TODO

---

### Sub-Task 5.6: Property Test - Apple-Only Stage 2 Execution
**File**: `tests/test_inference_pipeline.py` (NEW - needs to be created)

**What to Test**:
- Generate Stage 1 predictions for all 14 crops
- Verify Stage 2 executes ONLY for Apple with confidence ≥ 0.7
- Verify Stage 2 does NOT execute for non-Apple crops
- Verify Stage 2 does NOT execute for Apple with confidence < 0.7

**Example**:
```python
def test_apple_only_stage2_execution():
    """Property: Stage 2 executes only for Apple with confidence >= 0.7"""
    # For all crops != "Apple":
    #   Stage 2 should NOT execute
    # For crop == "Apple" and confidence < 0.7:
    #   Stage 2 should NOT execute
    # For crop == "Apple" and confidence >= 0.7:
    #   Stage 2 SHOULD execute
```

**Status**: ⏳ TODO

---

### Sub-Task 5.7: Input Validation Gate
**File**: `src/inference/inference_pipeline.py` (already has `validate_input()`)

```python
def validate_input(self, image_path: str) -> Tuple[bool, Optional[str]]:
    """Validate input image file."""
    # Check file exists
    # Check file size (max 10 MB)
    # Validate image format and integrity
```

**Status**: ✅ DONE

---

### Sub-Task 5.8: Basic Leaf Detection
**File**: `src/inference/inference_pipeline.py` (already has `detect_leaf()`)

```python
def detect_leaf(self, image: Image.Image) -> Tuple[bool, float, Optional[str]]:
    """Detect if image contains a valid leaf."""
    # Calculate green pixel ratio
    # Return: (is_leaf, confidence, warning)
```

**Status**: ✅ DONE

---

### Sub-Task 5.9: Borderline Handling
**File**: `src/inference/inference_pipeline.py` (already in `predict()`)

```python
# If leaf detection is borderline: proceed with warning
if leaf_warning:
    warnings.append(leaf_warning)
```

**Status**: ✅ DONE

---

### Sub-Task 5.10: Flask Web Application
**File**: `src/web/app.py` (already implemented)

**Endpoints**:
- `GET /api/health` - Health check
- `POST /api/predict` - Single image prediction
- `POST /api/predict-batch` - Batch prediction
- `GET /api/model-info` - Model information
- `GET /api/contract` - API contract documentation
- `GET /` - Frontend

**Status**: ✅ DONE

---

## What Actually Needs to Be Done

### 1. Create Test File
**File**: `tests/test_inference_pipeline.py`

```python
import pytest
from src.inference.inference_pipeline import InferencePipeline

def test_confidence_threshold_consistency():
    """Property: Predictions below threshold return Unknown"""
    # TODO: Implement

def test_apple_only_stage2_execution():
    """Property: Stage 2 executes only for Apple with confidence >= 0.7"""
    # TODO: Implement
```

### 2. Verify Models Load Correctly
- Ensure Stage 1 model loads from `./models/stage1/stage1_best.pth`
- Ensure Stage 2 model loads from `./models/stage2/stage2_best.pth`
- Test with dummy models if real models not available

### 3. Test End-to-End Pipeline
- Create test images (leaf, non-leaf, various crops)
- Run through complete pipeline
- Verify all routing rules work

### 4. Test Web API
- Test all 5 endpoints
- Test error handling
- Test batch processing

### 5. Create Frontend (Optional for MVP)
- Simple HTML form for image upload
- Display results
- Show error messages

---

## Implementation Order

1. **Create test file** with property tests
2. **Run tests** to verify pipeline works
3. **Test web API** endpoints manually
4. **Create simple frontend** (optional)
5. **Document API usage**

---

## Key Files to Know

**Inference**:
- `src/inference/inference_pipeline.py` - Main orchestrator (COMPLETE)
- `src/inference/recommendations.py` - Treatment lookup (needs implementation)

**Web**:
- `src/web/app.py` - Flask app (COMPLETE)
- `src/web/api_contract.py` - API spec (COMPLETE)

**Models**:
- `src/models/stage1.py` - Stage 1 model
- `src/models/stage2.py` - Stage 2 model

**Tests**:
- `tests/test_inference_pipeline.py` - NEW (needs creation)
- `tests/test_stage1_training.py` - Existing
- `tests/test_stage2_training.py` - Existing

---

## Locked Constraints

✅ **DO NOT CHANGE**:
- Confidence thresholds (0.7, 0.6, 0.15)
- Routing rules (Stage 2 only for Apple)
- API endpoints (5 endpoints locked)
- Error handling (status codes locked)
- Class mappings (14 crops, 4 diseases)

✅ **CAN CHANGE**:
- Frontend design
- Test implementation details
- Model paths (if needed)
- Logging/monitoring

---

## Success Criteria

Task 5 is complete when:

1. ✅ All 10 sub-tasks are implemented
2. ✅ Property tests pass (confidence thresholds, Apple routing)
3. ✅ Web API endpoints work correctly
4. ✅ Error handling returns correct status codes
5. ✅ Batch processing works
6. ✅ End-to-end pipeline tested
7. ✅ All tests pass

---

## Next Steps

1. Create `tests/test_inference_pipeline.py`
2. Implement property tests
3. Run tests to verify
4. Test web API
5. Create frontend (optional)
6. Document usage

**Ready to start? Let's go! 🚀**

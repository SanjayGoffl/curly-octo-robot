"""
Flask web application for crop disease identification.

Implements the locked API contract with exact request/response schemas.
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import os
import tempfile
from werkzeug.utils import secure_filename

from src.inference.inference_pipeline import InferencePipeline, PredictionStatus
from src.web.api_contract import APIContract, PredictionResponse


# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE_MB = 10

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

# Initialize inference pipeline (lazy load)
_pipeline = None


def get_pipeline():
    """Get or initialize inference pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = InferencePipeline(
                stage1_model_path="./models/stage1/stage1_best.pth",
                stage2_model_path="./models/stage2/stage2_best.pth",
                crop_threshold=0.7,
                disease_threshold=0.6
            )
        except Exception as e:
            app.logger.error(f"Failed to initialize pipeline: {str(e)}")
            _pipeline = None
    return _pipeline


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# ENDPOINT 1: Health Check
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    GET /api/health
    
    Returns:
        200 OK: {"status": "healthy", "models_loaded": true, ...}
        503 Service Unavailable: {"status": "unhealthy", "error": "..."}
    """
    pipeline = get_pipeline()
    
    if pipeline is None:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Models not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'device': str(pipeline.device),
        'version': '1.0'
    }), 200


# ============================================================================
# ENDPOINT 2: Single Image Prediction
# ============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict crop and disease from a single leaf image.
    
    POST /api/predict
    Content-Type: multipart/form-data
    Field: "image" (file)
    
    Returns:
        200 OK: PredictionResponse (see api_contract.py)
        400 Bad Request: Invalid input
        500 Internal Server Error: System error
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'status': 'invalid_input',
                'error': 'No image file provided',
                'model_version': '1.0'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'status': 'invalid_input',
                'error': 'No file selected',
                'model_version': '1.0'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'invalid_input',
                'error': 'Invalid file format. Please upload JPG or PNG.',
                'model_version': '1.0'
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get pipeline
            pipeline = get_pipeline()
            if pipeline is None:
                return jsonify({
                    'status': 'system_error',
                    'error': 'Models not loaded',
                    'model_version': '1.0'
                }), 500
            
            # Run inference
            result = pipeline.predict(filepath)
            
            # Convert to response format
            response_data = result.to_dict()
            
            return jsonify(response_data), 200
        
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'system_error',
            'error': f'System error: {str(e)}',
            'model_version': '1.0'
        }), 500


# ============================================================================
# ENDPOINT 3: Batch Prediction
# ============================================================================

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict crop and disease from multiple leaf images.
    
    POST /api/predict-batch
    Content-Type: multipart/form-data
    Field: "images" (multiple files)
    
    Returns:
        200 OK: {"status": "success", "results": [...], "total": N, ...}
        400 Bad Request: Invalid input
        500 Internal Server Error: System error
    """
    try:
        # Check if image files are present
        if 'images' not in request.files:
            return jsonify({
                'status': 'invalid_input',
                'error': 'No image files provided',
                'model_version': '1.0'
            }), 400
        
        files = request.files.getlist('images')
        
        if len(files) == 0:
            return jsonify({
                'status': 'invalid_input',
                'error': 'No files selected',
                'model_version': '1.0'
            }), 400
        
        if len(files) > 10:
            return jsonify({
                'status': 'invalid_input',
                'error': 'Maximum 10 files allowed',
                'model_version': '1.0'
            }), 400
        
        # Get pipeline
        pipeline = get_pipeline()
        if pipeline is None:
            return jsonify({
                'status': 'system_error',
                'error': 'Models not loaded',
                'model_version': '1.0'
            }), 500
        
        # Process each file
        results = []
        successful = 0
        failed = 0
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                results.append({
                    'status': 'invalid_input',
                    'error': f'Invalid format: {file.filename}',
                    'model_version': '1.0'
                })
                failed += 1
                continue
            
            # Save temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Run inference
                result = pipeline.predict(filepath)
                results.append(result.to_dict())
                
                if result.status == PredictionStatus.SUCCESS:
                    successful += 1
                else:
                    failed += 1
            
            except Exception as e:
                results.append({
                    'status': 'system_error',
                    'error': str(e),
                    'model_version': '1.0'
                })
                failed += 1
            
            finally:
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'total': len(results),
            'successful': successful,
            'failed': failed,
            'model_version': '1.0'
        }), 200
    
    except Exception as e:
        app.logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'status': 'system_error',
            'error': f'System error: {str(e)}',
            'model_version': '1.0'
        }), 500


# ============================================================================
# ENDPOINT 4: Model Info
# ============================================================================

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    Get information about loaded models.
    
    GET /api/model-info
    
    Returns:
        200 OK: Model information
    """
    pipeline = get_pipeline()
    
    if pipeline is None:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Models not loaded'
        }), 503
    
    return jsonify({
        'version': '1.0',
        'stage1': {
            'name': 'Stage 1: Crop Identification',
            'backbone': 'EfficientNet-B0',
            'num_classes': len(pipeline.CROP_CLASSES),
            'classes': pipeline.CROP_CLASSES,
            'confidence_threshold': pipeline.crop_threshold
        },
        'stage2': {
            'name': 'Stage 2: Apple Disease Classification',
            'backbone': 'EfficientNet-B0',
            'num_classes': len(pipeline.DISEASE_CLASSES),
            'classes': pipeline.DISEASE_CLASSES,
            'confidence_threshold': pipeline.disease_threshold
        },
        'device': str(pipeline.device),
        'input_size': [224, 224],
        'supported_formats': ['JPG', 'PNG'],
        'max_file_size_mb': MAX_FILE_SIZE_MB
    }), 200


# ============================================================================
# ENDPOINT 5: API Contract Documentation
# ============================================================================

@app.route('/api/contract', methods=['GET'])
def api_contract():
    """
    Get API contract specification.
    
    GET /api/contract
    
    Returns:
        200 OK: Complete API contract
    """
    return jsonify({
        'health_check': APIContract.health_check_spec(),
        'predict': APIContract.predict_spec(),
        'predict_batch': APIContract.predict_batch_spec(),
        'model_info': APIContract.model_info_spec(),
        'error_handling': APIContract.error_handling_spec(),
        'confidence_thresholds': APIContract.confidence_thresholds_spec(),
        'routing_rules': APIContract.routing_rules_spec()
    }), 200


# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render index page."""
    return render_template('index.html')


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'status': 'invalid_input',
        'error': f'File too large. Maximum {MAX_FILE_SIZE_MB} MB.',
        'model_version': '1.0'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'status': 'invalid_input',
        'error': 'Endpoint not found',
        'model_version': '1.0'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'status': 'system_error',
        'error': 'Internal server error',
        'model_version': '1.0'
    }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

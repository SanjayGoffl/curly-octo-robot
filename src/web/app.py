"""Flask web application."""

import os
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from src.utils.config import config
from src.utils.gpu import get_device
from src.inference.pipeline import InferencePipeline


def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__, template_folder="templates")
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = config.max_upload_size
    app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
    
    # Initialize models (placeholder - will be loaded when needed)
    app.stage1_model = None
    app.stage2_model = None
    app.pipeline = None
    app.device = get_device()
    
    @app.route('/')
    def index():
        """Home page."""
        return render_template('index.html')
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """Inference endpoint."""
        try:
            # Check if file is present
            if 'file' not in request.files:
                return jsonify({
                    "success": False,
                    "error": "No file provided"
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    "success": False,
                    "error": "No file selected"
                }), 400
            
            # Save uploaded file temporarily
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(temp_path)
            
            try:
                # Run inference
                if app.pipeline is None:
                    return jsonify({
                        "success": False,
                        "error": "Models not loaded. Please restart the application."
                    }), 500
                
                result = app.pipeline.predict(temp_path)
                
                return jsonify(result), 200
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Server error: {str(e)}"
            }), 500
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "ok",
            "models_loaded": app.pipeline is not None
        }), 200
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        debug=config.flask_debug,
        host='0.0.0.0',
        port=5000
    )

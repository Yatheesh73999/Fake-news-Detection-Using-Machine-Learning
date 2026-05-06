import json
import logging
from flask import Flask, request, jsonify, make_response
from pipeline import run_pipeline, PipelineConfig

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Manual CORS setup to avoid needing flask-cors package
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/verify', methods=['POST', 'OPTIONS'])
def verify_claim():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return make_response()
        
    data = request.get_json()
    if not data or 'claim' not in data:
        return jsonify({"error": "Missing 'claim' in request body."}), 400
        
    claim = data['claim']
    app.logger.info(f"Received claim for verification: {claim}")
    
    try:
        # Run the existing NLP pipeline
        # You can adjust max_articles in the config if you want it to be faster or more thorough
        config = PipelineConfig(max_articles=5, debug_print=True)
        result = run_pipeline(claim, config)
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"Error processing claim: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred during verification."}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000 as expected by the frontend
    app.run(host='127.0.0.1', port=5000, debug=True)

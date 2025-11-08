import os
import shap
import numpy as np
import pandas as pd
import onnxruntime as ort
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64

# Flask app setup
app = Flask(__name__)

# Allowed extensions for files
ALLOWED_EXTENSIONS = {'onnx', 'csv'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check allowed file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_image')
def testing():
    return render_template('result.html', shap_image='/home/josh/map.png')

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'model' not in request.files or 'data' not in request.files:
        return jsonify({'error': 'No model or data file part'})

    model_file = request.files['model']
    data_file = request.files['data']

    if model_file.filename == '' or data_file.filename == '':
        return jsonify({'error': 'No selected model or data file'})

    if model_file and allowed_file(model_file.filename) and data_file and allowed_file(data_file.filename):
        model_filename = secure_filename(model_file.filename)
        data_filename = secure_filename(data_file.filename)

        # Save the uploaded files
        model_filepath = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
        data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

        model_file.save(model_filepath)
        data_file.save(data_filepath)

        # Perform SHAP analysis and plot the graph
        try:
            shap_image = run_shap_analysis(model_filepath, data_filepath)
            return render_template('result.html', shap_image=shap_image)
        except Exception as e:
            return jsonify({'error': f'Error processing files: {str(e)}'})

    return jsonify({'error': 'Invalid file format'})

# Function to run SHAP analysis
def run_shap_analysis(model_path, data_path):
    # Load the model using onnxruntime
    session = ort.InferenceSession(model_path)

    # Load the training data (assumed to be in CSV format)
    data = pd.read_csv(data_path)
    
    # Ensure the data is in the correct format (convert to numpy array)
    input_data = data.values.astype(np.float32)
    
    # Get input details from the model (e.g., input name and shape)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Run the model to get the output for a sample batch (this could be adjusted based on the model)
    model_output = session.run(None, {input_name: input_data})[0]

    # Summarize background to speed up Kernel SHAP and avoid object-dtype results
    K = 50
    if len(input_data) > K:
        background = shap.sample(input_data, K)
    else:
        background = input_data

    # Robust prediction wrapper to ensure numpy numeric outputs
    def predict(x):
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        out = session.run(None, {input_name: x_arr})[0]
        out_arr = np.asarray(out)
        if out_arr.ndim == 1:
            out_arr = out_arr.reshape(-1, 1)
        return out_arr.astype(np.float64)

    # Create explainer and compute SHAP values
    explainer = shap.KernelExplainer(predict, background)
    shap_values = explainer.shap_values(input_data)

    # Plot SHAP values (summary plot)
    fig = plt.figure(figsize=(6, 6))
    shap.summary_plot(shap_values, input_data, show=False)

    # Convert plot to base64 for embedding in the HTML
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return f"data:image/png;base64,{img_str}"

# Running the Flask app
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

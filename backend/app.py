import os
import shap
import numpy as np
import pandas as pd
import onnxruntime as ort
from flask import Flask, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
from functions import make_waterfall_plot, make_bar_plot
from flask_cors import CORS
from llm import analyze_model


class ONNXModelWrapper:
    """Wrapper to make ONNX model compatible with SHAP."""
    
    def __init__(self, onnx_path: str):
        """
        Initialize ONNX model from file path.
        
        Args:
            onnx_path: Path to the .onnx model file
        """
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Get all outputs (some models have multiple outputs)
        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]
        
        # Use first output by default, or look for 'label' output for classifiers
        self.output_name = self.output_names[0]
        for name in self.output_names:
            if 'label' in name.lower():
                self.output_name = name
                break
        
        self.class_mapping = {}  # Will map string labels to integers
        print(f"ONNX Model initialized:")
        print(f"  Input: {self.input_name}")
        print(f"  Output: {self.output_name}")
        print(f"  Available outputs: {self.output_names}")
        
    def predict(self, X) -> np.ndarray:
        """
        Make predictions using ONNX model.
        
        Args:
            X: Input features as numpy array or DataFrame
            
        Returns:
            Predictions as numpy array (1D for regression/classification, 2D for probabilities)
        """
        # Convert DataFrame to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Ensure X is numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Ensure X is float32 (ONNX typically expects this)
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        # Run inference - get all outputs
        outputs = self.session.run(self.output_names, {self.input_name: X})
        
        # Try to find probability output for classifiers
        result = None
        for i, output_name in enumerate(self.output_names):
            current_output = outputs[i]
            
            # Look for probability output (usually has 'probabilities' in name or is 2D)
            if 'prob' in output_name.lower():
                result = current_output
                break
            # If we find a 2D numeric array, it's likely probabilities
            elif isinstance(current_output, np.ndarray) and len(current_output.shape) == 2:
                if np.issubdtype(current_output.dtype, np.number):
                    result = current_output
                    break
        
        # If no probability output found, use the main output
        if result is None:
            result = outputs[self.output_names.index(self.output_name)]
        
        # Handle dict/map output format (common in sklearn->ONNX conversions)
        if isinstance(result, (list, np.ndarray)) and len(result) > 0:
            if isinstance(result[0], dict):
                # Convert list of dicts to probability matrix
                # Each dict maps class label -> probability
                all_classes = sorted(set().union(*[d.keys() for d in result]))
                
                # Build class mapping if not exists
                for cls in all_classes:
                    if cls not in self.class_mapping:
                        self.class_mapping[cls] = len(self.class_mapping)
                
                # Create probability matrix
                prob_matrix = np.zeros((len(result), len(all_classes)))
                for i, pred_dict in enumerate(result):
                    for cls, prob in pred_dict.items():
                        prob_matrix[i, self.class_mapping[cls]] = prob
                
                result = prob_matrix
        
        # Ensure result is a proper numpy array
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        
        # Handle different output formats
        if result.dtype == object:
            # Check if it's strings
            try:
                first_elem = result.flat[0]
                if isinstance(first_elem, str):
                    # Output is strings (class labels) - convert to numeric indices
                    unique_labels = np.unique(result)
                    
                    # Build or update class mapping
                    for label in unique_labels:
                        if label not in self.class_mapping:
                            self.class_mapping[label] = len(self.class_mapping)
                    
                    # Convert strings to integers
                    result = np.array([self.class_mapping[label] for label in result])
            except:
                pass
        elif result.dtype.kind in ['U', 'S']:
            # String array
            unique_labels = np.unique(result)
            
            # Build or update class mapping
            for label in unique_labels:
                if label not in self.class_mapping:
                    self.class_mapping[label] = len(self.class_mapping)
            
            # Convert strings to integers
            result = np.array([self.class_mapping[label] for label in result])
        
        # Ensure it's a numeric type
        if not np.issubdtype(result.dtype, np.number):
            result = result.astype(np.float64)
        
        # Flatten if single column
        if len(result.shape) == 2 and result.shape[1] == 1:
            result = result.flatten()
        
        return result
    
    def __call__(self, X) -> np.ndarray:
        """Allow calling the wrapper directly."""
        return self.predict(X)

# Flask epp setup
app = Flask(__name__, static_folder="../frontend/dist/", static_url_path="/")
CORS(app) 

# Allowed extensions for files
ALLOWED_EXTENSIONS = {"onnx", "csv"}
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/test-cors")
def test_cors():
    return jsonify({"message": "CORS is working!"})

@app.route("/<path:other>")
def catch_all(other):
    return send_from_directory(app.static_folder, other)


@app.route("/")
def index():
    # Serve the index.html file from the React build
    return send_from_directory(app.static_folder, "index.html")


# Helper function to check allowed file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Route to handle file upload
@app.route("/upload", methods=["POST"])
def upload_file():
    if "model" not in request.files or "data" not in request.files:
        return jsonify({"error": "No model or data file part"}), 400

    model_file = request.files["model"]
    data_file = request.files["data"]

    if model_file.filename == "" or data_file.filename == "":
        return jsonify({"error": "No selected model or data file"}), 400

    if (
        model_file
        and allowed_file(model_file.filename)
        and data_file
        and allowed_file(data_file.filename)
    ):
        model_filename = secure_filename(model_file.filename)
        data_filename = secure_filename(data_file.filename)

        # Save the uploaded files
        model_filepath = os.path.join(app.config["UPLOAD_FOLDER"], model_filename)
        data_filepath = os.path.join(app.config["UPLOAD_FOLDER"], data_filename)
        model_file.save(model_filepath)
        data_file.save(data_filepath)

        # Perform SHAP analysis and plot the graph
        try:
            waterfall_image = make_waterfall_plot(model_filepath, data_filepath)
            bar_plot_image = make_bar_plot(model_filepath, data_filepath)
            onnx_path = "uploads/decision_tree.onnx"

            X_train = pd.read_csv("uploads/train_X.csv")
            wrapper = ONNXModelWrapper(onnx_path)
            predictions_onnx = wrapper.predict(X_train)
            print(f"ONNX predictions shape: {predictions_onnx.shape}")
            
            # Analyze the ONNX model
            print("\nAnalyzing ONNX model with SHAP...")
            data, summary = analyze_model(
                model_or_path=onnx_path,
                X_train=X_train,
                predictions=predictions_onnx,
                n_sample_explanations=3,
                nsamples=50  # Lower for faster testing
            )
            return jsonify(
                {
                    "waterfall": waterfall_image,
                    "bar": bar_plot_image,
                    "summary": summary,
                }
            )
        except Exception as e:
            return jsonify({"error": f"Error processing files: {str(e)}"}), 500

    return jsonify({"error": "Invalid file format"}), 400

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

    # Use a smaller background (summarize) to speed up Kernel SHAP and avoid passing
    # a large 2D Python sequence into the model which can produce object-dtype outputs.
    K = 50
    if len(input_data) > K:
        background = shap.sample(input_data, K)
    else:
        background = input_data

    # Debug: show background shape/dtype
    print(
        f"background shape={getattr(background, 'shape', None)} dtype={getattr(background, 'dtype', None)}"
    )

    # Robust prediction wrapper: ensure inputs are numpy arrays and outputs are numeric numpy arrays
    def predict(x):
        # Convert to numpy array of floats
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        outs = session.run(None, {input_name: x_arr})

        # If model returns multiple outputs, prefer the first numeric output
        selected = None
        for o in outs:
            o_arr = np.asarray(o)
            try:
                # try converting to float
                o_float = o_arr.astype(np.float64)
                selected = o_float
                # found numeric output
                break
            except Exception:
                continue

        if selected is None:
            # No numeric outputs found --- try to map string labels to integer codes
            out0 = np.asarray(outs[0])
            # build mapping keeping order of first occurrence
            flat = out0.ravel()
            unique = list(dict.fromkeys(flat.tolist()))
            mapping = {v: i for i, v in enumerate(unique)}
            mapped = np.vectorize(lambda s: mapping[s])(out0)
            selected = mapped.astype(np.float64)

        # Ensure 2D output (n_samples, n_outputs)
        if selected.ndim == 1:
            selected = selected.reshape(-1, 1)

        return selected

    # Sanity-check predictions on the background
    sample_pred = predict(background[: min(len(background), 5)])
    print(
        f"sample_pred type={type(sample_pred)} shape={sample_pred.shape} dtype={sample_pred.dtype}"
    )

    # Set up SHAP explainer (using KernelExplainer in this case)
    explainer = shap.KernelExplainer(predict, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(input_data)

    # Plot SHAP values (summary plot)
    fig = plt.figure(figsize=(6, 6))
    shap.summary_plot(shap_values, input_data, show=False)

    # Convert plot to base64 for embedding in the HTML
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    return f"data:image/png;base64,{img_str}"


# Running the Flask app
if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(debug=True)

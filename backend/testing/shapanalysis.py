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
def make_waterfall_thing(model_path, data_path):
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

    # Robust prediction wrapper: ensure inputs are numpy arrays and outputs are numeric numpy arrays
    def predict(x):
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        outs = session.run(None, {input_name: x_arr})

        selected = None
        for o in outs:
            o_arr = np.asarray(o)
            try:
                o_float = o_arr.astype(np.float64)
                selected = o_float
                break
            except Exception:
                continue

        if selected is None:
            out0 = np.asarray(outs[0])
            flat = out0.ravel()
            unique = list(dict.fromkeys(flat.tolist()))
            mapping = {v: i for i, v in enumerate(unique)}
            mapped = np.vectorize(lambda s: mapping[s])(out0)
            selected = mapped.astype(np.float64)

        if selected.ndim == 1:
            selected = selected.reshape(-1, 1)

        return selected

    # Sanity-check predictions on the background
    sample_pred = predict(background[:min(len(background), 5)])

    # Set up SHAP explainer (using KernelExplainer in this case)
    explainer = shap.KernelExplainer(predict, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(input_data)

    # --- NEW: Waterfall Plot for the first instance and first output ---
    # For multi-output models, we select the first output: shap_values[0, 0]
    # For single-output models, you can use shap_values[0] as is.

    # Check if the model has multiple outputs (i.e., shap_values is a 2D array)
    # Squeeze the extra dimension in shap_values to remove the 1 at the end
    shap_values_first_instance = shap_values[0].squeeze()

    # Create the SHAP Explanation object
    explanation = shap.Explanation(values=shap_values_first_instance,
                                base_values=explainer.expected_value[0], 
                                data=input_data[0], 
                                feature_names=data.columns)

    # Generate the waterfall plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Larger figure for better readability
    shap.waterfall_plot(explanation)

    # Adjust layout to avoid clipping
    plt.tight_layout()

    # Convert plot to base64 for embedding in the HTML
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return f"data:image/png;base64,{img_str}"



# Example usage
run_shap_analysis('sample stuff/decision_tree.onnx', 'backend/train_X.csv')

from PIL import Image
import io
import base64

# Call the existing function to get the base64 image string
img_base64 = run_shap_analysis('sample stuff/decision_tree.onnx', 'backend/train_X.csv')

# Decode the base64 string to image data
img_data = base64.b64decode(img_base64.split(',')[1])  # Remove the "data:image/png;base64," part

# Open the image using PIL
img = Image.open(io.BytesIO(img_data))

# Display the image
img.show()

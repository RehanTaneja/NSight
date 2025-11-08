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
    print(f"background shape={getattr(background, 'shape', None)} dtype={getattr(background, 'dtype', None)}")

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
    sample_pred = predict(background[:min(len(background), 5)])
    print(f"sample_pred type={type(sample_pred)} shape={sample_pred.shape} dtype={sample_pred.dtype}")

    # Set up SHAP explainer (using KernelExplainer in this case)
    explainer = shap.KernelExplainer(predict, background)

    # Compute SHAP values
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


run_shap_analysis('backend/decision_tree.onnx','backend/train_X.csv')

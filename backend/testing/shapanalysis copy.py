import shap
import onnxruntime as ort
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def make_dependence_plots(model_path, data_path):
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

    # Wrap shap_values in shap.Explanation for compatibility
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # For multi-output models, select the first output

    shap_values = shap.Explanation(values=shap_values, 
                                   base_values=explainer.expected_value[0], 
                                   data=input_data, 
                                   feature_names=data.columns)

    # --- Debug: Print available feature names in shap_values ---
    print("Available feature names in shap_values:")
    print(shap_values.feature_names)

    # Calculate the mean absolute SHAP values for feature importance
    mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)

    # --- NEW: Flatten mean_abs_shap_values if it's not 1D ---
    if mean_abs_shap_values.ndim > 1:
        mean_abs_shap_values = mean_abs_shap_values.flatten()

    # Create a DataFrame for feature importance ranking
    feature_importance = pd.DataFrame({
        'Feature': data.columns,
        'Mean |SHAP Value|': mean_abs_shap_values
    })

    # Sort the features by importance (mean absolute SHAP value)
    feature_importance = feature_importance.sort_values(by='Mean |SHAP Value|', ascending=False)

    # Get the top 3-5 features
    top_features = feature_importance['Feature'].head(5).values

    dependence_plots_base64 = []
    for feature in top_features:
        # Strip spaces from both feature and feature names for matching
        feature = feature.strip()

        # Find the index of the feature in the input data and shap_values
        feature_index = data.columns.get_loc(feature)

        # Debug: Check if the feature is matched correctly
        print(f"Using feature: {feature} (index {feature_index})")

        # Extract SHAP values for the current feature (column)
        feature_shap_values = shap_values.values[:, feature_index]

        # Extract the feature values (input data)
        feature_data = input_data[:, feature_index]

        # --- Fix for interaction feature handling ---
        # Pass both SHAP values and the corresponding feature values to the dependence plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Generate the SHAP dependence plot for each of the top features
        shap.dependence_plot(
            feature_index, shap_values.values, input_data, 
            interaction_index=feature_index, show=False
        )

        # Customizing the axis labels with the actual feature name
        ax.set_xlabel(f'{feature} (Feature Values)', fontsize=14)  # Set x-axis to feature name
        ax.set_ylabel('SHAP Value (Impact on Prediction)', fontsize=14)  # y-axis label for SHAP value

        # Optional: Set the title of the plot
        ax.set_title(f'Dependence Plot for {feature}', fontsize=16)

        # Convert plot to base64 for embedding in the HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        dependence_plots_base64.append(f"data:image/png;base64,{img_str}")

        # Optional: Close the plot to avoid too many open figures
        plt.close(fig)

    return dependence_plots_base64


# Example usage
# make_bar_plot('sample stuff/decision_tree.onnx', 'backend/train_X.csv')

from PIL import Image
import io
import base64

# Call the existing function to get the base64 image string
# Get the first base64 string from the list
img_base64 = make_dependence_plots('sample stuff/decision_tree.onnx', 'backend/train_X.csv')[0]

# Decode the base64 string to image data
img_data = base64.b64decode(img_base64.split(',')[1])  # Remove the "data:image/png;base64," part

# Open the image using PIL
img = Image.open(io.BytesIO(img_data))

# Display the image
img.show()

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

    # Set up SHAP explainer (using KernelExplainer in this case)
    explainer = shap.KernelExplainer(lambda x: session.run(None, {input_name: x})[0], input_data)

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
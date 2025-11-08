import onnxruntime as ort

# load a ONNX model and returns the models session object (for inference) and input name (for prediction)
def load_onnx_model(model_path: str):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session, input_name

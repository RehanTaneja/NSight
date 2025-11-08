import shap
import onnxruntime as ort
import numpy as np
import pandas as pd
from typing import Any


def _to_numpy_input(x: Any) -> np.ndarray:
    """Normalize various input types to a 2D numpy array of float32."""
    if isinstance(x, pd.DataFrame):
        return x.astype(np.float32).to_numpy()
    
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32)
    
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict):
        df = pd.DataFrame.from_records(x)
        return _to_numpy_input(df)
    
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float32)


def explain_model(model_path: str, data: pd.DataFrame):
    """Explain an ONNX model using SHAP using the data as-is, no padding/trimming."""
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    
    # Use all columns exactly as in CSV
    X_np = data.astype(np.float32).to_numpy()

    # Load ONNX model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Prediction function
    def pred_func(x: Any):
        arr = _to_numpy_input(x)
        out = session.run(None, {input_name: arr.astype(np.float32)})[0]
        out_arr = np.asarray(out)
        if out_arr.ndim == 1:
            out_arr = out_arr.reshape(-1, 1)
        return out_arr

    explainer = shap.Explainer(pred_func, X_np)
    shap_values = explainer(X_np)
    return shap_values

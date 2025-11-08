🎯 Project Overview

This project builds a modular explainability backend that allows users to upload trained ML models (in ONNX format) and datasets (CSV files) to generate SHAP-based explanations for model predictions.

The backend automatically:

Loads the ONNX model dynamically (supports multiple architectures).

Reads and validates the uploaded dataset.

Computes SHAP values to quantify feature importance.

Returns SHAP summaries or plots to visualize model behavior.

It’s designed for AI research, university machine learning projects, and model auditing, where transparency and interpretability are essential.

🧰 Tools & Libraries Used
🐍 Python (Backend Core)
Library	Purpose
FastAPI	Exposes REST endpoints /predict and /explain
uvicorn	ASGI web server for running the API
pandas	Data loading and preprocessing
numpy	Numerical array manipulation
onnxruntime	Model inference engine for ONNX models
shap	Explainability toolkit for SHAP value generation
matplotlib	For plotting SHAP summary and force plots
pydantic	Input validation for API schemas
python-dotenv	(Optional) Manage environment variables
pytest	For backend testing

🚀 Features & Workflow
🔹 Step 1: Model Upload

Users upload an ONNX model via /upload_model or place it in uploads/models/.

🔹 Step 2: Dataset Upload

Users upload a dataset CSV (train or test data).

🔹 Step 3: Model Explanation

Endpoint /explain performs:

Loads ONNX model using onnxruntime.InferenceSession.

Loads dataset into Pandas.

Aligns feature dimensions (if necessary).

Runs SHAP explainability:

Calculates SHAP values.

Generates summary plots.

Returns:

JSON summary (feature importances).

Optional base64 plot images.

🔹 Step 4: Visualization

Users can visualize:

Global importance (which features affect model predictions the most).

Local importance (why a specific sample got a particular prediction).

📦 Final Deliverables
Deliverable	Description
1. FastAPI Backend	Fully functional explainability API with endpoints for model upload, dataset upload, and SHAP computation.
2. ONNX Model Integration	Generic support for any ONNX model (e.g. XGBoost, Random Forest, Neural Nets).
3. SHAP Explainability Engine	Compute and visualize global/local feature importance.
4. Documentation & Setup Guide	Instructions for setup, usage, and extending the system.
5.  Frontend	Basic dashboard to upload models/data and view SHAP plots interactively.

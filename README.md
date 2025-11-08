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

**SHAP PLOTS**
1. Bar plot - Quick importance ranking: Simple feature importance ranking. Just shows which features matter most on average, no direction info. Great for executive summaries.
```
shap.summary_plot(shap_values, X, plot_type="bar")
``` 
2. Summary plot - Main insights page: Shows feature importance + impact direction. Each dot is a sample, color shows feature value (red=high, blue=low), x-axis shows SHAP value (impact on prediction). Best for overall model understanding.
```
shap.summary_plot(shap_values, X)
```
4. Waterfall plots - Let users explain individual predictions: Explains ONE prediction step-by-step. Shows base value → how each feature pushes prediction up/down → final output. Perfect for "why did model predict X for this customer?"
```
shap.waterfall_plot(shap_values[0])
```
6. Dependence plots - For top 3-5 features to show relationships: Shows how one feature's value affects predictions, with interaction effects colored by another feature. Reveals non-linear relationships like "age only matters for high income customers."
```
shap.dependence_plot("feature_name", shap_values, X)
```
8. Global importance numbers - For sortable tables/comparisons: Raw numbers for dashboards. Just calculate mean absolute SHAP per feature.
```
importance = np.abs(shap_values).mean(axis=0)
# Returns array of importance scores per feature
```

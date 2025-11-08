import numpy as np
import shap
import json
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


chat_session = None

# ============================================================================
# SHAP COMPUTATION
# ============================================================================

def compute_shap_values(model, X_train: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer (works for most models)."""
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_train)
    return shap_values


# ============================================================================
# HELPER FUNCTIONS - DETERMINE MODEL TYPE
# ============================================================================

def determine_model_info(shap_values: shap.Explanation, predictions: np.ndarray) -> Dict[str, Any]:
    """
    Determine if model is regression, binary classification, or multi-class.
    
    Returns: {
        'task_type': 'regression' | 'binary_classification' | 'multi_class',
        'n_classes': int (for classification),
        'target_class': int (which class to analyze for classification)
    }
    """
    # Check SHAP values shape
    if len(shap_values.values.shape) == 2:
        # Shape is (n_samples, n_features) - could be regression or binary classification
        # Check predictions to determine
        unique_preds = np.unique(predictions.round() if len(predictions.shape) == 1 else predictions.argmax(axis=1))
        
        if len(unique_preds) <= 10 and np.all(unique_preds == unique_preds.astype(int)):
            # Likely classification
            if len(unique_preds) == 2:
                return {
                    'task_type': 'binary_classification',
                    'n_classes': 2,
                    'target_class': 1  # Positive class
                }
            else:
                return {
                    'task_type': 'multi_class',
                    'n_classes': len(unique_preds),
                    'target_class': None  # Will analyze all classes
                }
        else:
            return {
                'task_type': 'regression',
                'n_classes': None,
                'target_class': None
            }
    
    elif len(shap_values.values.shape) == 3:
        # Shape is (n_samples, n_features, n_classes) - multi-class classification
        n_classes = shap_values.values.shape[2]
        
        if n_classes == 2:
            return {
                'task_type': 'binary_classification',
                'n_classes': 2,
                'target_class': 1  # Positive class
            }
        else:
            return {
                'task_type': 'multi_class',
                'n_classes': n_classes,
                'target_class': None  # Will analyze all classes
            }
    
    # Default to regression
    return {
        'task_type': 'regression',
        'n_classes': None,
        'target_class': None
    }


def extract_shap_for_class(shap_values: shap.Explanation, class_idx: Optional[int] = None) -> np.ndarray:
    """
    Extract SHAP values for a specific class or return as-is for regression/binary.
    
    Args:
        shap_values: SHAP explanation object
        class_idx: Which class to extract (None for regression or to keep all)
    
    Returns:
        SHAP values array of shape (n_samples, n_features)
    """
    if len(shap_values.values.shape) == 3:
        if class_idx is not None:
            return shap_values.values[:, :, class_idx]
        else:
            # For multi-class, return all classes - will be handled separately
            return shap_values.values
    else:
        return shap_values.values


def extract_base_value(shap_values: shap.Explanation, sample_idx: int, 
                       class_idx: Optional[int] = None) -> float:
    """
    Safely extract base value from SHAP explanation object.
    
    Handles all possible shapes and types of base_values:
    - Scalar (single value for entire dataset)
    - 1D array with length n_samples (one per sample)
    - 1D array with length n_classes (one per class)
    - 2D array (n_samples, n_classes)
    
    Args:
        shap_values: SHAP explanation object
        sample_idx: Index of the sample
        class_idx: Class index for multi-class (None for binary/regression)
    
    Returns:
        Base value as a float
    """
    base_values = shap_values.base_values
    
    # Case 1: base_values is a scalar (single value for all samples)
    if np.isscalar(base_values):
        return float(base_values)
    
    # Case 2: base_values is an array
    if isinstance(base_values, (np.ndarray, list)):
        base_values = np.array(base_values)
        
        # Check the shape
        if base_values.ndim == 0:
            # 0D array (scalar wrapped in array)
            return float(base_values.item())
        
        elif base_values.ndim == 1:
            # 1D array: could be (n_samples,) or (n_classes,)
            if len(base_values) == shap_values.values.shape[0]:
                # Shape is (n_samples,) - one base value per sample
                value = base_values[sample_idx]
                # Handle case where value might still be an array
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        return float(value.item())
                    else:
                        return float(value[0])
                return float(value)
            elif class_idx is not None and len(base_values) == shap_values.values.shape[2]:
                # Shape is (n_classes,) - one base value per class
                return float(base_values[class_idx])
            else:
                # Fallback: use first element or sample_idx
                if sample_idx < len(base_values):
                    value = base_values[sample_idx]
                    if isinstance(value, np.ndarray):
                        return float(value.item() if value.ndim == 0 else value[0])
                    return float(value)
                else:
                    value = base_values[0]
                    if isinstance(value, np.ndarray):
                        return float(value.item() if value.ndim == 0 else value[0])
                    return float(value)
        
        elif base_values.ndim == 2:
            # 2D array: shape is (n_samples, n_classes)
            if class_idx is not None:
                return float(base_values[sample_idx, class_idx])
            else:
                # For binary classification, might need the last column
                if base_values.shape[1] == 2:
                    return float(base_values[sample_idx, 1])
                else:
                    return float(base_values[sample_idx, 0])
    
    # Fallback: return 0
    print(f"Warning: Could not extract base_value, returning 0.0. base_values shape: {getattr(base_values, 'shape', 'no shape')}")
    return 0.0


# ============================================================================
# HELPER FUNCTIONS - EXTRACT DATA FOR LLM
# ============================================================================

def get_feature_importance(shap_values: shap.Explanation, X: pd.DataFrame, 
                          model_info: Dict[str, Any]) -> List[Dict]:
    """
    Get features ranked by importance with their average impact.
    
    For multi-class, aggregates importance across all classes.
    """
    if model_info['task_type'] == 'multi_class' and model_info['target_class'] is None:
        # Multi-class: aggregate across all classes
        # Use mean absolute SHAP across all classes and samples
        shap_vals = shap_values.values  # Shape: (n_samples, n_features, n_classes)
        mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))  # Average over samples and classes
        mean_shap = shap_vals.mean(axis=(0, 2))  # Average over samples and classes
    else:
        # Binary classification or regression
        shap_vals = extract_shap_for_class(shap_values, model_info.get('target_class'))
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        mean_shap = shap_vals.mean(axis=0)
    
    # Sort by importance
    indices = np.argsort(mean_abs_shap)[::-1]
    
    results = []
    for rank, idx in enumerate(indices, 1):
        avg_impact = float(mean_shap[idx])
        results.append({
            'rank': rank,
            'feature': X.columns[idx],
            'importance': float(mean_abs_shap[idx]),
            'avg_impact': avg_impact,
            'impact_direction': 'increases' if avg_impact > 0 else 'decreases'
        })
    
    return results


def get_feature_importance_per_class(shap_values: shap.Explanation, X: pd.DataFrame,
                                     n_classes: int) -> Dict[int, List[Dict]]:
    """
    For multi-class classification, get feature importance for each class separately.
    
    Returns: {
        0: [{'feature': 'age', 'importance': 0.45, ...}, ...],
        1: [...],
        2: [...]
    }
    """
    per_class_importance = {}
    
    for class_idx in range(n_classes):
        shap_vals = shap_values.values[:, :, class_idx]
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        mean_shap = shap_vals.mean(axis=0)
        
        indices = np.argsort(mean_abs_shap)[::-1][:10]  # Top 10 per class
        
        class_features = []
        for rank, idx in enumerate(indices, 1):
            avg_impact = float(mean_shap[idx])
            class_features.append({
                'rank': rank,
                'feature': X.columns[idx],
                'importance': float(mean_abs_shap[idx]),
                'avg_impact': avg_impact,
                'impact_direction': 'increases' if avg_impact > 0 else 'decreases'
            })
        
        per_class_importance[class_idx] = class_features
    
    return per_class_importance


def get_feature_stats(X: pd.DataFrame) -> Dict[str, Dict]:
    """
    Get basic statistics for each feature in the dataset.
    """
    stats = {}
    for col in X.columns:
        stats[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean()),
            'median': float(X[col].median())
        }
    return stats


def explain_prediction(shap_values: shap.Explanation, X: pd.DataFrame, 
                      prediction: float, sample_idx: int, model_info: Dict[str, Any],
                      top_n: int = 5) -> Dict:
    """
    Explain a single prediction by showing which features contributed most.
    """
    if model_info['task_type'] == 'multi_class' and model_info['target_class'] is None:
        # For multi-class, explain the predicted class
        if len(prediction.shape) > 0:
            predicted_class = int(np.argmax(prediction))
            prediction_value = float(prediction[predicted_class])
        else:
            predicted_class = int(prediction)
            prediction_value = float(prediction)
        
        sample_shap = shap_values.values[sample_idx, :, predicted_class]
        
        # Get base value for this class
        base_value = extract_base_value(shap_values, sample_idx, predicted_class)
    else:
        # Binary classification or regression
        sample_shap = extract_shap_for_class(shap_values, model_info.get('target_class'))[sample_idx]
        predicted_class = None
        prediction_value = float(prediction) if not hasattr(prediction, '__len__') else float(prediction[0])
        
        # Get base value
        base_value = extract_base_value(shap_values, sample_idx, None)
    
    # Get top contributing features
    top_indices = np.argsort(np.abs(sample_shap))[::-1][:top_n]
    
    contributors = []
    for idx in top_indices:
        contribution = float(sample_shap[idx])
        contributors.append({
            'feature': X.columns[idx],
            'value': float(X.iloc[sample_idx, idx]),
            'contribution': contribution,
            'effect': 'increases' if contribution > 0 else 'decreases'
        })
    
    result = {
        'prediction': prediction_value,
        'base_value': base_value,
        'contributors': contributors
    }
    
    if predicted_class is not None:
        result['predicted_class'] = predicted_class
    
    return result


def get_sample_indices(predictions: np.ndarray, n_samples: int = 5) -> List[int]:
    """
    Pick diverse sample predictions to explain (low, medium, high predictions).
    """
    # Handle multi-class predictions
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # For multi-class, use max probability as the metric
        predictions_1d = predictions.max(axis=1)
    else:
        predictions_1d = predictions.flatten() if len(predictions.shape) > 1 else predictions
    
    quantiles = np.linspace(0, 100, n_samples)
    indices = []
    for q in quantiles:
        target = np.percentile(predictions_1d, q)
        idx = int(np.argmin(np.abs(predictions_1d - target)))
        if idx not in indices:  # Avoid duplicates
            indices.append(idx)
    
    return indices


def get_prediction_summary(predictions: np.ndarray, model_info: Dict[str, Any]) -> Dict:
    """
    Get summary statistics about model predictions.
    """
    if model_info['task_type'] == 'multi_class':
        # For multi-class, show class distribution
        if len(predictions.shape) > 1:
            predicted_classes = predictions.argmax(axis=1)
            class_probs = predictions
        else:
            predicted_classes = predictions.astype(int)
            class_probs = None
        
        unique, counts = np.unique(predicted_classes, return_counts=True)
        class_distribution = {int(cls): int(count) for cls, count in zip(unique, counts)}
        
        summary = {
            'task_type': 'multi_class',
            'n_samples': len(predictions),
            'class_distribution': class_distribution,
            'class_percentages': {int(cls): float(count / len(predictions) * 100) 
                                 for cls, count in zip(unique, counts)}
        }
        
        if class_probs is not None:
            summary['avg_confidence'] = float(class_probs.max(axis=1).mean())
            summary['min_confidence'] = float(class_probs.max(axis=1).min())
            summary['max_confidence'] = float(class_probs.max(axis=1).max())
        
        return summary
    
    else:
        # Binary classification or regression
        predictions_1d = predictions.flatten() if len(predictions.shape) > 1 else predictions
        
        summary = {
            'task_type': model_info['task_type'],
            'n_samples': len(predictions_1d),
            'mean': float(predictions_1d.mean()),
            'min': float(predictions_1d.min()),
            'max': float(predictions_1d.max()),
            'median': float(np.median(predictions_1d)),
            'std': float(predictions_1d.std())
        }
        
        if model_info['task_type'] == 'binary_classification':
            # Add class distribution for binary
            predicted_classes = (predictions_1d > 0.5).astype(int)
            unique, counts = np.unique(predicted_classes, return_counts=True)
            summary['class_distribution'] = {int(cls): int(count) for cls, count in zip(unique, counts)}
            summary['class_percentages'] = {int(cls): float(count / len(predictions_1d) * 100) 
                                           for cls, count in zip(unique, counts)}
        
        return summary


# ============================================================================
# BUILD COMPLETE DATA PACKAGE FOR LLM
# ============================================================================

def build_llm_data(shap_values: shap.Explanation, X: pd.DataFrame, 
                   predictions: np.ndarray, n_samples: int = 5) -> Dict[str, Any]:
    """
    Build a complete data package with all insights for LLM to summarize.
    """
    # Determine model type
    model_info = determine_model_info(shap_values, predictions)
    
    # Get sample indices for detailed explanations
    sample_indices = get_sample_indices(predictions, n_samples)
    
    # Build the data package
    data = {
        'model_info': model_info,
        
        'dataset_info': {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist()
        },
        
        'predictions': get_prediction_summary(predictions, model_info),
        
        'feature_importance': get_feature_importance(shap_values, X, model_info),
        
        'feature_stats': get_feature_stats(X),
        
        'example_predictions': [
            explain_prediction(shap_values, X, predictions[idx], idx, model_info)
            for idx in sample_indices
        ]
    }
    
    # For multi-class, add per-class feature importance
    if model_info['task_type'] == 'multi_class' and model_info['n_classes'] is not None:
        data['feature_importance_per_class'] = get_feature_importance_per_class(
            shap_values, X, model_info['n_classes']
        )
    
    return data


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def create_prompt(data: Dict[str, Any]) -> str:
    """
    Create a single, comprehensive prompt for the LLM.
    """
    task_type = data['model_info']['task_type']
    
    # Adjust prompt based on task type
    if task_type == 'multi_class':
        task_description = f"This is a multi-class classification model with {data['model_info']['n_classes']} classes."
        additional_instructions = """
For multi-class models:
- Explain which features are most important overall across all classes
- If per-class feature importance is provided, mention any notable differences between classes
- Explain how different features affect different class predictions
"""
    elif task_type == 'binary_classification':
        task_description = "This is a binary classification model (predicting positive/negative or yes/no)."
        additional_instructions = """
For binary classification:
- Explain which features increase the probability of the positive class
- Mention the overall prediction distribution (how many positive vs negative)
"""
    else:
        task_description = "This is a regression model (predicting continuous values)."
        additional_instructions = """
For regression:
- Explain which features increase or decrease the predicted value
- Mention the range and distribution of predictions
"""
    
    prompt = f"""You are an AI explainability expert. Analyze this machine learning model and provide clear insights.

{task_description}

# MODEL DATA
{json.dumps(data, indent=2)}

# YOUR TASK
Generate a comprehensive yet readable summary with these sections:

1. **Overview**: 2-3 sentences describing what the model does and its overall behavior

2. **Key Drivers**: List the top 5 most important features and explain:
   - What each feature is and its typical impact
   - Whether it increases or decreases predictions
   - How much it matters (use the importance scores)

3. **Prediction Patterns**: Describe how the model's predictions are distributed and what this means

4. **Example Predictions**: For 2-3 of the example predictions provided, explain in plain English:
   - What the model predicted
   - Which features drove that specific prediction
   - Why those features had the effect they did

5. **Key Insights**: 3-4 bullet points with actionable insights about how this model makes decisions

{additional_instructions}

Keep the language clear and non-technical. Use specific numbers from the data. Make it useful for someone who needs to understand and trust this model's decisions.
"""
    return prompt


def get_gemini_response(prompt: str) -> str:
    """Call Gemini API to generate summary."""
    try:
        global chat_session
        if chat_session is None:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-2.5-flash")
            chat_session = model.start_chat(history=[])
            print("New chart session started")
        response = chat_session.send_message(prompt)
        return response
    except Exception as e:
        print(f"Error calling Gemini API: {e}")


def generate_summary(data: Dict[str, Any]) -> str:
    """Generate textual summary using Gemini."""
    prompt = create_prompt(data)
    summary = get_gemini_response(prompt)
    return summary


# ============================================================================
# MAIN FUNCTION - USE THIS
# ============================================================================

def analyze_model(model, X_train: pd.DataFrame,
                 predictions: np.ndarray,
                 n_sample_explanations: int = 5) -> Tuple[Dict, str]:
    """
    Complete SHAP analysis with LLM summary.
    
    Works for:
    - Binary classification (predictions as probabilities or class labels)
    - Multi-class classification (predictions as probability matrix or class labels)
    - Regression (predictions as continuous values)
    """
    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, X_train)
    
    print("Extracting insights...")
    data = build_llm_data(shap_values, X_train, predictions, n_sample_explanations)
    
    print("Generating summary...")
    summary = generate_summary(data)
    
    return data, summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    print("=" * 80)
    print("TESTING BINARY CLASSIFICATION")
    print("=" * 80)
    
    # Binary classification
    X_train, y_train = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_test, y_test = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=43)
    X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(10)])
    X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(10)])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_train)[:, 1]
    
    data, summary = analyze_model(
        model=model,
        X_train=X_train,
        predictions=predictions
    )
    
    print("\nBINARY CLASSIFICATION SUMMARY:")
    print("=" * 80)
    print(summary)
    print("=" * 80)
    
    print("\n\n")
    print("=" * 80)
    print("TESTING MULTI-CLASS CLASSIFICATION")
    print("=" * 80)
    
    # Multi-class classification
    X_train_mc, y_train_mc = make_classification(n_samples=1000, n_features=10, n_classes=4, 
                                                  n_informative=8, random_state=42)
    X_test_mc, y_test_mc = make_classification(n_samples=200, n_features=10, n_classes=4,
                                                n_informative=8, random_state=43)
    X_train_mc = pd.DataFrame(X_train_mc, columns=[f'feature_{i}' for i in range(10)])
    X_test_mc = pd.DataFrame(X_test_mc, columns=[f'feature_{i}' for i in range(10)])
    
    model_mc = RandomForestClassifier(n_estimators=100, random_state=42)
    model_mc.fit(X_train_mc, y_train_mc)
    predictions_mc = model_mc.predict_proba(X_train_mc)
    
    data_mc, summary_mc = analyze_model(
        model=model_mc,
        X_train=X_train_mc,
        predictions=predictions_mc
    )
    
    print("\nMULTI-CLASS CLASSIFICATION SUMMARY:")
    print("=" * 80)
    print(summary_mc)
    print("=" * 80)

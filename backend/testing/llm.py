import numpy as np
import shap
import json
from typing import Dict, List, Any, Tuple
import pandas as pd


# ============================================================================
# SHAP COMPUTATION
# ============================================================================

def compute_shap_values(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer (works for most models)."""
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_test)
    return shap_values


# ============================================================================
# HELPER FUNCTIONS - EXTRACT DATA FOR LLM
# ============================================================================

def get_feature_importance(shap_values: shap.Explanation, X: pd.DataFrame) -> List[Dict]:
    """
    Get features ranked by importance with their average impact.
    
    Returns: [
        {'feature': 'age', 'importance': 0.45, 'avg_impact': 0.23, 'impact_direction': 'increases'},
        ...
    ]
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    mean_shap = shap_values.values.mean(axis=0)
    
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


def get_feature_stats(X: pd.DataFrame) -> Dict[str, Dict]:
    """
    Get basic statistics for each feature in the dataset.
    
    Returns: {
        'age': {'min': 18, 'max': 90, 'mean': 45.2},
        ...
    }
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
                      prediction: float, sample_idx: int, top_n: int = 5) -> Dict:
    """
    Explain a single prediction by showing which features contributed most.
    
    Returns: {
        'prediction': 0.85,
        'base_value': 0.5,
        'contributors': [
            {'feature': 'income', 'value': 75000, 'contribution': 0.25, 'effect': 'increases'},
            ...
        ]
    }
    """
    sample_shap = shap_values.values[sample_idx]
    
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
    
    base_value = float(shap_values.base_values[sample_idx] 
                      if isinstance(shap_values.base_values, np.ndarray)
                      else shap_values.base_values)
    
    return {
        'prediction': float(prediction),
        'base_value': base_value,
        'contributors': contributors
    }


def get_sample_indices(predictions: np.ndarray, n_samples: int = 5) -> List[int]:
    """
    Pick diverse sample predictions to explain (low, medium, high predictions).
    """
    quantiles = np.linspace(0, 100, n_samples)
    indices = []
    for q in quantiles:
        target = np.percentile(predictions, q)
        idx = int(np.argmin(np.abs(predictions - target)))
        indices.append(idx)
    return indices


def get_prediction_summary(predictions: np.ndarray) -> Dict:
    """
    Get summary statistics about model predictions.
    
    Returns: {
        'mean': 0.67,
        'min': 0.12,
        'max': 0.98,
        'std': 0.15
    }
    """
    return {
        'mean': float(predictions.mean()),
        'min': float(predictions.min()),
        'max': float(predictions.max()),
        'median': float(np.median(predictions)),
        'std': float(predictions.std())
    }


# ============================================================================
# BUILD COMPLETE DATA PACKAGE FOR LLM
# ============================================================================

def build_llm_data(shap_values: shap.Explanation, X: pd.DataFrame, 
                   predictions: np.ndarray, n_samples: int = 5) -> Dict[str, Any]:
    """
    Build a complete data package with all insights for LLM to summarize.
    
    This contains everything the LLM needs to generate a comprehensive summary.
    """
    # Get sample indices for detailed explanations
    sample_indices = get_sample_indices(predictions, n_samples)
    
    # Build the data package
    data = {
        'dataset_info': {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist()
        },
        
        'predictions': get_prediction_summary(predictions),
        
        'feature_importance': get_feature_importance(shap_values, X),
        
        'feature_stats': get_feature_stats(X),
        
        'example_predictions': [
            explain_prediction(shap_values, X, predictions[idx], idx)
            for idx in sample_indices
        ]
    }
    
    return data


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def create_prompt(data: Dict[str, Any]) -> str:
    """
    Create a single, comprehensive prompt for the LLM.
    """
    prompt = f"""You are an AI explainability expert. Analyze this machine learning model and provide clear insights.

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

Keep the language clear and non-technical. Use specific numbers from the data. Make it useful for someone who needs to understand and trust this model's decisions.
"""
    return prompt


def generate_summary(data: Dict[str, Any], llm_function: callable, **llm_kwargs) -> str:
    """
    Generate textual summary using an LLM.
    
    Args:
        data: The complete data package from build_llm_data()
        llm_function: Your LLM function, e.g., lambda prompt: gemini.generate(prompt)
        **llm_kwargs: Additional arguments for your LLM (temperature, max_tokens, etc.)
    
    Returns:
        Generated text summary
    """
    prompt = create_prompt(data)
    summary = llm_function(prompt, **llm_kwargs)
    return summary


# ============================================================================
# MAIN FUNCTION - USE THIS
# ============================================================================

def analyze_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 predictions: np.ndarray, llm_function: callable,
                 n_sample_explanations: int = 5, **llm_kwargs) -> Tuple[Dict, str]:
    """
    Complete SHAP analysis with LLM summary.
    
    Args:
        model: Your trained model
        X_train: Training data (for SHAP background)
        X_test: Test data to explain
        predictions: Model predictions on X_test
        llm_function: Function to call your LLM, e.g., lambda prompt: gemini.generate(prompt)
        n_sample_explanations: How many example predictions to explain in detail
        **llm_kwargs: Additional arguments for your LLM
    
    Returns:
        (data_dict, text_summary) - Both the structured data and human-readable summary
    
    Example:
        data, summary = analyze_model(
            model=my_model,
            X_train=train_df,
            X_test=test_df,
            predictions=test_predictions,
            llm_function=lambda p: gemini.generate_content(p).text,
            temperature=0.7
        )
    """
    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, X_train, X_test)
    
    print("Extracting insights...")
    data = build_llm_data(shap_values, X_test, predictions, n_sample_explanations)
    
    print("Generating summary...")
    summary = generate_summary(data, llm_function, **llm_kwargs)
    
    return data, summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    # Example usage:
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Train a model
    X_train, y_train = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_test, y_test = make_classification(n_samples=200, n_features=10, random_state=43)
    X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(10)])
    X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(10)])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Define your LLM function
    def my_llm(prompt, **kwargs):
        # Call your LLM API here
        # return gemini.generate_content(prompt, **kwargs).text
        return "[Your LLM response here]"
    
    # Analyze!
    data, summary = analyze_model(
        model=model,
        X_train=X_train,
        X_test=X_test,
        predictions=predictions,
        llm_function=my_llm,
        temperature=0.7
    )
    
    print("SUMMARY:")
    print(summary)
    """
    
    print("SHAP Analysis Module Ready!")
    print("\nMain function: analyze_model()")
    print("Helper functions available:")
    print("  - compute_shap_values()")
    print("  - get_feature_importance()")
    print("  - explain_prediction()")
    print("  - build_llm_data()")
    print("  - generate_summary()")

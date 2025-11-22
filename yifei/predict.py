"""
Prediction module
Load trained model and make predictions on new data
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, List, Optional


def load_model(
    model_dir: str = "models",
    preprocessor_file: str = "preprocessor.pkl",
    model_file: str = "xgb_model.pkl"
) -> tuple:
    """
    Load trained preprocessor and model
    
    Args:
        model_dir: Directory containing model files
        preprocessor_file: Preprocessor filename
        model_file: Model filename
    
    Returns:
        Tuple of (preprocessor, model)
    """
    preprocessor_path = os.path.join(model_dir, preprocessor_file)
    model_path = os.path.join(model_dir, model_file)
    
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded preprocessor from {preprocessor_path}")
    print(f"Loaded model from {model_path}")
    
    return preprocessor, model


def predict_single_trial(
    trial_data: Dict[str, Any],
    preprocessor: Any,
    model: Any,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predict success probability for a single trial
    
    Args:
        trial_data: Dictionary with trial features
        preprocessor: Fitted preprocessor
        model: Trained model
        threshold: Classification threshold
    
    Returns:
        Dictionary with prediction results
    """
    # Convert to DataFrame
    df = pd.DataFrame([trial_data])
    
    # Prepare features (same as training)
    feature_cols = [
        'text_features',
        'enrollment',
        'trial_duration_months',
        'intervention_count',
        'arm_count',
        'primary_outcome_count',
        'secondary_outcome_count',
        'location_count',
        'phase',
        'study_type',
        'allocation',
        'masking',
        'intervention_model',
        'sponsor_class',
        'sex',
        'is_randomized',
        'is_blinded',
        'is_industry_sponsored'
    ]
    
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in df.columns:
            if col == 'text_features':
                df[col] = ""
            elif col in ['is_randomized', 'is_blinded', 'is_industry_sponsored']:
                df[col] = 0
            else:
                df[col] = None
    
    X = df[feature_cols]
    
    # Transform and predict
    try:
        X_processed = preprocessor.transform(X)
        success_proba = model.predict_proba(X_processed)[0, 1]
        success_pred = int(success_proba >= threshold)
        
        result = {
            'success_probability': float(success_proba),
            'success_prediction': success_pred,
            'confidence': 'high' if abs(success_proba - 0.5) > 0.3 else 'medium' if abs(success_proba - 0.5) > 0.15 else 'low'
        }
    except Exception as e:
        result = {
            'error': str(e),
            'success_probability': None,
            'success_prediction': None
        }
    
    return result


def predict_batch(
    df: pd.DataFrame,
    preprocessor: Any,
    model: Any,
    threshold: float = 0.5,
    nct_id_col: str = 'nct_id'
) -> pd.DataFrame:
    """
    Predict success probabilities for a batch of trials
    
    Args:
        df: DataFrame with trial features
        preprocessor: Fitted preprocessor
        model: Trained model
        threshold: Classification threshold
        nct_id_col: Column name for trial ID
    
    Returns:
        DataFrame with predictions added
    """
    # Prepare features
    feature_cols = [
        'text_features',
        'enrollment',
        'trial_duration_months',
        'intervention_count',
        'arm_count',
        'primary_outcome_count',
        'secondary_outcome_count',
        'location_count',
        'phase',
        'study_type',
        'allocation',
        'masking',
        'intervention_model',
        'sponsor_class',
        'sex',
        'is_randomized',
        'is_blinded',
        'is_industry_sponsored'
    ]
    
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in df.columns:
            if col == 'text_features':
                df[col] = ""
            elif col in ['is_randomized', 'is_blinded', 'is_industry_sponsored']:
                df[col] = 0
            else:
                df[col] = None
    
    X = df[feature_cols]
    
    # Transform and predict
    X_processed = preprocessor.transform(X)
    success_proba = model.predict_proba(X_processed)[:, 1]
    success_pred = (success_proba >= threshold).astype(int)
    
    # Add predictions to DataFrame
    result_df = df.copy()
    result_df['success_probability'] = success_proba
    result_df['success_prediction'] = success_pred
    result_df['confidence'] = result_df['success_probability'].apply(
        lambda x: 'high' if abs(x - 0.5) > 0.3 else 'medium' if abs(x - 0.5) > 0.15 else 'low'
    )
    
    return result_df


def predict_from_file(
    input_file: str,
    output_file: str = "data/results/predictions.csv",
    model_dir: str = "models",
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Load data from file, make predictions, and save results
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save predictions
        model_dir: Directory containing model files
        threshold: Classification threshold
    
    Returns:
        DataFrame with predictions
    """
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} trials")
    
    # Load model
    print("Loading model...")
    preprocessor, model = load_model(model_dir)
    
    # Make predictions
    print("Making predictions...")
    result_df = predict_batch(df, preprocessor, model, threshold)
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("Prediction Summary")
    print("="*50)
    print(f"Total trials: {len(result_df)}")
    print(f"Predicted success: {(result_df['success_prediction'] == 1).sum()}")
    print(f"Predicted failure: {(result_df['success_prediction'] == 0).sum()}")
    print(f"\nAverage success probability: {result_df['success_probability'].mean():.4f}")
    print(f"Median success probability: {result_df['success_probability'].median():.4f}")
    print(f"\nHigh confidence predictions: {(result_df['confidence'] == 'high').sum()}")
    print(f"Medium confidence predictions: {(result_df['confidence'] == 'medium').sum()}")
    print(f"Low confidence predictions: {(result_df['confidence'] == 'low').sum()}")
    
    return result_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "data/results/predictions.csv"
        predict_from_file(input_file, output_file)
    else:
        # Example: predict on processed data if it exists
        data_path = "data/processed/trials_processed.csv"
        if os.path.exists(data_path):
            print("Predicting on processed data...")
            df = pd.read_csv(data_path)
            
            # Remove label column if exists
            if 'success_label' in df.columns:
                df = df.drop(columns=['success_label'])
            
            # Load model and predict
            try:
                preprocessor, model = load_model()
                result_df = predict_batch(df, preprocessor, model)
                
                # Save results
                output_path = "data/results/predictions.csv"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                result_df.to_csv(output_path, index=False)
                print(f"\nPredictions saved to {output_path}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please train the model first using model_training.py")
        else:
            print("Usage: python predict.py <input_file> [output_file]")
            print("Or ensure data/processed/trials_processed.csv exists")


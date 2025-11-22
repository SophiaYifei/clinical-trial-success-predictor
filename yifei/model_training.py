"""
Model training module using XGBoost
Handles feature engineering, model training, and evaluation
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_curve,
    f1_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


def create_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing pipeline for mixed feature types
    
    Returns:
        ColumnTransformer with text, numerical, and categorical transformers
    """
    # Text features
    text_transformer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Numerical features
    numerical_features = [
        'enrollment',
        'trial_duration_months',
        'intervention_count',
        'arm_count',
        'primary_outcome_count',
        'secondary_outcome_count',
        'location_count',
        'is_randomized',
        'is_blinded',
        'is_industry_sponsored'
    ]
    
    num_transformer = StandardScaler()
    
    # Categorical features
    categorical_features = [
        'phase',
        'study_type',
        'allocation',
        'masking',
        'intervention_model',
        'sponsor_class',
        'sex'
    ]
    
    cat_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        drop='if_binary'
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'text_features'),
            ('num', num_transformer, numerical_features),
            ('cat', cat_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training
    
    Args:
        df: Processed DataFrame
    
    Returns:
        Tuple of (X, y) for training
    """
    # Select features
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
    
    X = df[feature_cols].copy()
    y = df['success_label'].copy()
    
    return X, y


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    scale_pos_weight: float = None
) -> Tuple[Any, Any]:
    """
    Train XGBoost model with preprocessing pipeline
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        scale_pos_weight: Weight for positive class (for imbalanced data)
    
    Returns:
        Tuple of (preprocessor, trained_model)
    """
    # Calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        if pos_count > 0:
            scale_pos_weight = neg_count / pos_count
        else:
            scale_pos_weight = 1.0
    
    print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    # Fit preprocessor and transform features
    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    print(f"Processed training shape: {X_train_processed.shape}")
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1,
        verbosity=1
    )
    
    # Use validation set if provided
    if X_val is not None and y_val is not None:
        X_val_processed = preprocessor.transform(X_val)
        model.fit(
            X_train_processed, y_train,
            eval_set=[(X_val_processed, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
    else:
        model.fit(X_train_processed, y_train)
    
    print("Model training completed!")
    
    return preprocessor, model


def evaluate_model(
    preprocessor: Any,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate model performance on test set
    
    Args:
        preprocessor: Fitted preprocessor
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Get predictions
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = np.trapz(precision, recall)
    
    metrics = {
        'roc_auc': auc,
        'pr_auc': pr_auc,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'y_proba': y_proba,
        'y_pred': y_pred,
        'y_true': y_test.values
    }
    
    return metrics


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_model: bool = True,
    model_dir: str = "models"
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Complete training and evaluation pipeline
    
    Args:
        df: Processed DataFrame with features and labels
        test_size: Proportion of test set
        random_state: Random seed
        save_model: Whether to save the model
        model_dir: Directory to save models
    
    Returns:
        Tuple of (preprocessor, model, metrics)
    """
    # Prepare data
    X, y = prepare_training_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"\nTraining label distribution:")
    print(y_train.value_counts())
    print(f"\nTest label distribution:")
    print(y_test.value_counts())
    
    # Train model
    preprocessor, model = train_xgboost_model(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    metrics = evaluate_model(preprocessor, model, X_test, y_test)
    
    print(f"\nROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"\nClassification Report:")
    print(metrics['classification_report'])
    
    # Cross-validation on training set
    print("\n" + "="*50)
    print("Cross-validation on training set...")
    print("="*50)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Create a temporary pipeline for CV
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=random_state,
            eval_metric='auc',
            n_jobs=-1
        ))
    ])
    
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print(f"CV ROC AUC scores: {cv_scores}")
    print(f"CV ROC AUC mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"\nPreprocessor saved to {preprocessor_path}")
        
        # Save model
        model_path = os.path.join(model_dir, "xgb_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    
    return preprocessor, model, metrics


if __name__ == "__main__":
    # Test training
    data_path = "data/processed/trials_processed.csv"
    if os.path.exists(data_path):
        print("Loading processed data...")
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        
        print("\nStarting training pipeline...")
        preprocessor, model, metrics = train_and_evaluate(df)
        
        print("\nTraining completed successfully!")
    else:
        print(f"Please run data_preprocessing.py first to create {data_path}")


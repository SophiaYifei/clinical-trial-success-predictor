"""
Main pipeline script for Clinical Trial Success Prediction
Complete workflow from data fetching to predictions
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_fetch import fetch_alzheimer_trials, save_raw_data
from data_preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate
from predict import predict_from_file, load_model, predict_batch
import pandas as pd


def run_full_pipeline(
    max_pages: int = 10,
    page_size: int = 100,
    phase_filter: str = "PHASE3",
    test_size: float = 0.2,
    random_state: int = 42,
    skip_fetch: bool = False,
    skip_preprocess: bool = False,
    skip_train: bool = False
):
    """
    Run complete pipeline: fetch -> preprocess -> train -> predict
    
    Args:
        max_pages: Maximum pages to fetch from API
        page_size: Studies per page
        phase_filter: Phase filter (PHASE3, PHASE2, etc.)
        test_size: Proportion for test set
        random_state: Random seed
        skip_fetch: Skip data fetching step
        skip_preprocess: Skip preprocessing step
        skip_train: Skip training step
    """
    print("="*70)
    print("Clinical Trial Success Prediction Pipeline")
    print("="*70)
    
    # Step 1: Fetch data
    raw_data_path = "data/raw/trials_raw.json"
    if not skip_fetch:
        print("\n[Step 1/4] Fetching data from ClinicalTrials.gov API...")
        print("-" * 70)
        studies = fetch_alzheimer_trials(
            max_pages=max_pages,
            page_size=page_size,
            phase_filter=phase_filter
        )
        save_raw_data(studies, raw_data_path)
    else:
        print("\n[Step 1/4] Skipping data fetch (using existing data)")
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {raw_data_path}")
    
    # Step 2: Preprocess data
    processed_data_path = "data/processed/trials_processed.csv"
    if not skip_preprocess:
        print("\n[Step 2/4] Preprocessing data...")
        print("-" * 70)
        try:
            df_processed = load_and_preprocess_data(
                raw_data_path=raw_data_path,
                output_path=processed_data_path,
                filter_phase=phase_filter
            )
        except ValueError as e:
            print(f"\nError during preprocessing: {e}")
            print("\nPlease check:")
            print("1. Data was successfully fetched from API")
            print("2. Raw data file exists and contains valid JSON")
            print("3. Try running data_fetch.py separately to debug")
            return
    else:
        print("\n[Step 2/4] Skipping preprocessing (using existing data)")
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at {processed_data_path}")
        df_processed = pd.read_csv(processed_data_path)
    
    # Step 3: Train model
    if not skip_train:
        print("\n[Step 3/4] Training model...")
        print("-" * 70)
        preprocessor, model, metrics = train_and_evaluate(
            df_processed,
            test_size=test_size,
            random_state=random_state,
            save_model=True
        )
    else:
        print("\n[Step 3/4] Skipping training (using existing model)")
    
    # Step 4: Make predictions
    print("\n[Step 4/4] Making predictions...")
    print("-" * 70)
    
    # Load model
    try:
        preprocessor, model = load_model()
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first.")
        return
    
    # Predict on processed data
    df_for_prediction = df_processed.copy()
    if 'success_label' in df_for_prediction.columns:
        # Keep labels for comparison but don't use them for prediction
        df_labels = df_for_prediction[['nct_id', 'success_label']].copy()
        df_for_prediction = df_for_prediction.drop(columns=['success_label'])
    
    result_df = predict_batch(df_for_prediction, preprocessor, model)
    
    # Add labels back if they exist
    if 'success_label' in df_processed.columns:
        result_df = result_df.merge(df_labels, on='nct_id', how='left')
    
    # Save predictions
    predictions_path = "data/results/predictions.csv"
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    result_df.to_csv(predictions_path, index=False)
    
    print(f"\nPredictions saved to {predictions_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("Pipeline Summary")
    print("="*70)
    print(f"Total trials processed: {len(result_df)}")
    
    if 'success_label' in result_df.columns:
        print(f"\nActual vs Predicted:")
        print(f"  Actual success: {(result_df['success_label'] == 1).sum()}")
        print(f"  Predicted success: {(result_df['success_prediction'] == 1).sum()}")
        
        # Calculate accuracy if labels available
        if result_df['success_label'].notna().any():
            accuracy = (result_df['success_prediction'] == result_df['success_label']).mean()
            print(f"\n  Accuracy: {accuracy:.4f}")
    
    print(f"\nAverage success probability: {result_df['success_probability'].mean():.4f}")
    print(f"Median success probability: {result_df['success_probability'].median():.4f}")
    
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Success Prediction Pipeline"
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=10,
        help='Maximum pages to fetch from API (default: 10)'
    )
    parser.add_argument(
        '--page-size',
        type=int,
        default=100,
        help='Studies per page (default: 100)'
    )
    parser.add_argument(
        '--phase',
        type=str,
        default='PHASE3',
        help='Phase filter: PHASE3, PHASE2, etc. (default: PHASE3)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip data fetching step'
    )
    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip preprocessing step'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training step'
    )
    
    args = parser.parse_args()
    
    run_full_pipeline(
        max_pages=args.max_pages,
        page_size=args.page_size,
        phase_filter=args.phase,
        test_size=args.test_size,
        random_state=args.random_state,
        skip_fetch=args.skip_fetch,
        skip_preprocess=args.skip_preprocess,
        skip_train=args.skip_train
    )


if __name__ == "__main__":
    main()


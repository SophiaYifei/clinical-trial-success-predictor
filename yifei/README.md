# Clinical Trial Success Prediction Pipeline

Complete machine learning pipeline for predicting Alzheimer's Disease Phase 3 clinical trial success using XGBoost.

## Overview

This pipeline predicts the success probability of Phase 3 clinical trials for Alzheimer's Disease based on trial design features, intervention characteristics, and textual descriptions. The pipeline includes:

1. **Data Fetching**: Retrieves trial data from ClinicalTrials.gov API v2
2. **Data Preprocessing**: Extracts and engineers features from raw API responses
3. **Model Training**: Trains an XGBoost classifier with cross-validation
4. **Prediction**: Generates success probability predictions for new trials

## Features Used

### Text Features
- Brief title
- Brief summary
- Detailed description
- Eligibility criteria

### Design Features
- Phase (filtered to Phase 3)
- Study type
- Enrollment count
- Allocation (randomized vs non-randomized)
- Masking (blinding)
- Intervention model

### Numerical Features
- Trial duration (months)
- Number of interventions
- Number of arms
- Number of primary/secondary outcomes
- Number of locations
- Binary indicators (randomized, blinded, industry-sponsored)

### Categorical Features
- Phase
- Study type
- Allocation
- Masking
- Intervention model
- Sponsor class
- Sex eligibility

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Complete Pipeline (Recommended)

Run the complete pipeline from data fetching to predictions:

```bash
python main.py
```

### Step-by-Step Execution

#### 1. Fetch Data
```bash
python data_fetch.py
```

This will:
- Fetch Alzheimer Disease Phase 3 trials from ClinicalTrials.gov API
- Save raw JSON data to `data/raw/trials_raw.json`

#### 2. Preprocess Data
```bash
python data_preprocessing.py
```

This will:
- Extract features from raw JSON
- Create success labels (COMPLETED + HasResults = Success)
- Save processed CSV to `data/processed/trials_processed.csv`

#### 3. Train Model
```bash
python model_training.py
```

This will:
- Train XGBoost model with cross-validation
- Evaluate on test set
- Save model and preprocessor to `models/` directory

#### 4. Make Predictions
```bash
python predict.py data/processed/trials_processed.csv data/results/predictions.csv
```

Or predict on new data:
```bash
python predict.py <input_file> <output_file>
```

### Command Line Options

```bash
python main.py --help

Options:
  --max-pages INT      Maximum pages to fetch (default: 10)
  --page-size INT      Studies per page (default: 100)
  --phase STR          Phase filter: PHASE3, PHASE2, etc. (default: PHASE3)
  --test-size FLOAT    Test set proportion (default: 0.2)
  --random-state INT   Random seed (default: 42)
  --skip-fetch         Skip data fetching step
  --skip-preprocess    Skip preprocessing step
  --skip-train         Skip training step
```

### Examples

```bash
# Fetch more data
python main.py --max-pages 20 --page-size 100

# Use existing data, retrain model
python main.py --skip-fetch --skip-preprocess

# Only make predictions (using existing model)
python main.py --skip-fetch --skip-preprocess --skip-train
```

## Project Structure

```
yifei/
├── main.py                 # Main pipeline script
├── data_fetch.py           # Data fetching from API
├── data_preprocessing.py   # Feature extraction and preprocessing
├── model_training.py       # Model training and evaluation
├── predict.py              # Prediction module
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   ├── raw/               # Raw API responses (JSON)
│   ├── processed/         # Processed features (CSV)
│   └── results/           # Prediction results (CSV)
└── models/
    ├── preprocessor.pkl   # Fitted preprocessor
    └── xgb_model.pkl      # Trained XGBoost model
```

## Model Details

### Algorithm
- **XGBoost Classifier** with the following hyperparameters:
  - `n_estimators`: 200
  - `max_depth`: 5
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `scale_pos_weight`: Automatically calculated to handle class imbalance

### Preprocessing
- **Text Features**: TF-IDF vectorization (max_features=3000, ngram_range=(1,2))
- **Numerical Features**: StandardScaler normalization
- **Categorical Features**: OneHotEncoder

### Evaluation Metrics
- ROC AUC
- Precision-Recall AUC
- F1 Score
- Confusion Matrix
- Classification Report

## Handling Class Imbalance

The dataset typically has imbalanced classes (more failures than successes). The pipeline handles this by:
1. Using `scale_pos_weight` in XGBoost
2. Stratified train-test split
3. Stratified cross-validation
4. Reporting PR-AUC in addition to ROC-AUC

## Output Files

### Predictions CSV
The prediction output includes:
- `nct_id`: Trial identifier
- `success_probability`: Predicted probability of success (0-1)
- `success_prediction`: Binary prediction (0 or 1)
- `confidence`: High/Medium/Low based on probability distance from 0.5
- All original features

## Notes

- The API has rate limits; the pipeline includes delays between requests
- Success label is defined as: `COMPLETED` status AND `hasResults=True`
- Trials with unknown status (ongoing) are filtered out during preprocessing
- Model performance may vary based on data quality and quantity

## Troubleshooting

### API Errors
- Check internet connection
- Verify API endpoint is accessible
- Reduce `--max-pages` if hitting rate limits

### Missing Features
- Ensure all required feature columns exist in input data
- Check preprocessing step completed successfully

### Model Not Found
- Run training step first: `python model_training.py`
- Check `models/` directory exists and contains model files

## Future Improvements

- Add more features (intervention types, endpoints, etc.)
- Experiment with other models (LightGBM, CatBoost, Neural Networks)
- Implement feature importance analysis
- Add model interpretability (SHAP values)
- Support for other phases (Phase 1, Phase 2)
- Support for other conditions beyond Alzheimer's Disease


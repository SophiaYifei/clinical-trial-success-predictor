import matplotlib
# Set backend before importing pyplot to avoid SystemError on macOS
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import requests
import pandas as pd
import numpy as np
import time
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Configuration
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Big Pharma companies list
BIG_PHARMA_LIST = [
    'biogen', 'lilly', 'pfizer', 'roche', 'genentech', 'merck', 
    'abbvie', 'novartis', 'janssen', 'eisai', 'astrazeneca', 'bayer',
    'gsk', 'glaxosmithkline', 'sanofi', 'takeda', 'bristol', 'myers',
    'squibb', 'bms', 'amgen', 'gilead', 'regeneron'
]

# Data Fetching Functions

def fetch_all_ad_trials_robust():
    studies = []
    next_page_token = None
    max_retries = 3
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }

    print("Starting data extraction (download all, filter locally)...")
    
    while True:
        params = {
            "query.cond": "Alzheimer", 
            "pageSize": 100,
            "fields": "ProtocolSection" 
        }
        
        if next_page_token:
            params["pageToken"] = next_page_token
        
        for attempt in range(max_retries):
            try:
                response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if data is None:
                    raise ValueError("Empty response from API")
                
                current_batch = data.get("studies", []) or []
                
                if not current_batch:
                    break

                studies.extend(current_batch)
                next_page_token = data.get("nextPageToken")
                
                print(f"   Fetched {len(current_batch)} studies... Total: {len(studies)}")
                
                if not next_page_token:
                    break
                    
                time.sleep(0.2)
                break 
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"Error fetching data: {e}")
                    return studies
        
        if not next_page_token:
            break
            
    print(f"Download complete. Total raw studies: {len(studies)}")
    return studies

# Helper Functions

def is_big_pharma(sponsor_name):
    if not sponsor_name: return 0
    sponsor_lower = str(sponsor_name).lower()
    return 1 if any(company in sponsor_lower for company in BIG_PHARMA_LIST) else 0

def extract_planned_duration(outcomes):
    if outcomes is None or not outcomes: return None
    max_weeks = 0
    MONTHS_TO_WEEKS = 52 / 12
    pattern = r'(\d+)\s*(week|month|day|year)s?\b'
    for outcome in outcomes:
        if not isinstance(outcome, dict): continue
        time_frame = str(outcome.get("timeFrame", "") or "").lower()
        if not time_frame: continue
        matches = re.findall(pattern, time_frame)
        for num, unit in matches:
            try:
                val = float(num)
                if 'year' in unit: weeks = val * 52
                elif 'month' in unit: weeks = val * MONTHS_TO_WEEKS
                elif 'day' in unit: weeks = val / 7
                else: weeks = val
                if weeks > max_weeks: max_weeks = weeks
            except: continue
    return max_weeks / MONTHS_TO_WEEKS if max_weeks > 0 else None

# Data Parsing and Feature Extraction
def parse_and_filter_phase1(studies_raw):
    rows = []
    print("Filtering data locally for Phase 1 trials...")
    
    for study in studies_raw:
        if not isinstance(study, dict): continue
            
        proto = study.get("protocolSection", {}) or {}
        design = proto.get("designModule", {}) or {}
        status_mod = proto.get("statusModule", {}) or {}
        
        if design.get("studyType") != "INTERVENTIONAL": continue
        
        phases = design.get("phases", []) or []
        if not isinstance(phases, list) or "PHASE1" not in phases: 
            continue
        
        overall_status = str(status_mod.get("overallStatus", "")).upper()
        if overall_status not in ['COMPLETED', 'TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
            continue

        row = {}
        row['nct_id'] = proto.get("identificationModule", {}).get("nctId")
        
        start_date_struct = status_mod.get("startDateStruct", {}) or {}
        start_date_str = str(start_date_struct.get("date", "") or "")
        match = re.search(r'\d{4}', start_date_str)
        row['start_year'] = int(match.group()) if match else 0

        # Label
        if overall_status == 'COMPLETED':
            is_success = 1
        else:
            is_success = 0
        row['is_success'] = is_success

        # Features
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {}) or {}
        lead_sponsor = sponsor_mod.get("leadSponsor", {}) or {}
        sponsor_class = str(lead_sponsor.get("class", "UNKNOWN") or "UNKNOWN")
        sponsor_name = str(lead_sponsor.get("name", "") or "")
        row['is_industry_sponsor'] = 1 if sponsor_class == "INDUSTRY" else 0
        row['is_big_pharma'] = is_big_pharma(sponsor_name)

        enrollment_info = design.get("enrollmentInfo", {}) or {}
        row['enrollment'] = enrollment_info.get("count")
        
        # Duration
        outcomes_module = proto.get("outcomesModule", {}) or {}
        primary_outcomes = outcomes_module.get("primaryOutcomes", []) or []
        row['planned_duration_months'] = extract_planned_duration(primary_outcomes)

        eligibility = proto.get("eligibilityModule", {}) or {}
        criteria = str(eligibility.get("eligibilityCriteria", "") or "").lower()
        healthy_keywords = ['healthy volunteer', 'healthy subject', 'healthy participant', 'normal volunteer']
        row['is_healthy_volunteers'] = 1 if any(kw in criteria for kw in healthy_keywords) else 0

        description_mod = proto.get("descriptionModule", {}) or {}
        brief_summary = str(description_mod.get("briefSummary", "") or "").lower()
        intervention_model = str(design.get("designInfo", {}).get("interventionModelDescription", "") or "").lower()
        full_text = brief_summary + " " + intervention_model
        
        row['is_dose_escalation'] = 1 if any(kw in full_text for kw in ['escalation', 'ascending', 'titration', 'sad', 'mad']) else 0
        row['is_single_dose'] = 1 if 'single' in full_text and 'dose' in full_text else 0
        row['is_multiple_dose'] = 1 if 'multiple' in full_text and 'dose' in full_text else 0

        primary_text = " ".join([str(o.get("measure", "")) for o in primary_outcomes if isinstance(o, dict)]).lower()
        row['has_safety_endpoint'] = 1 if any(kw in primary_text for kw in ['safety', 'adverse event', 'tolerability', 'toxicity']) else 0
        row['has_pk_endpoint'] = 1 if any(kw in primary_text for kw in ['pharmacokinetic', 'pk', 'auc', 'cmax', 'half-life']) else 0
        row['has_efficacy_endpoint'] = 1 if any(kw in primary_text for kw in ['adas', 'mmse', 'cognition', 'cdr']) else 0

        interventions_module = proto.get("armsInterventionsModule", {}) or {}
        interventions = interventions_module.get("interventions", []) or []
        intervention_desc = ""
        for i in interventions:
            if isinstance(i, dict):
                intervention_desc += " " + str(i.get("description", "") or "") + " " + str(i.get("name", "") or "")
        intervention_desc = intervention_desc.lower()
        
        row['is_oral'] = 1 if any(kw in intervention_desc for kw in ['oral', 'tablet', 'capsule', 'mouth']) else 0
        row['is_iv'] = 1 if any(kw in intervention_desc for kw in ['intravenous', 'infusion', 'injection', 'iv']) else 0

        arm_groups = interventions_module.get("armGroups", []) or []
        row['num_arms'] = len(arm_groups) if isinstance(arm_groups, list) else 0
        row['has_placebo'] = 1 if 'placebo' in intervention_desc else 0

        rows.append(row)
    
    return pd.DataFrame(rows)

# Main Execution
if __name__ == "__main__":
    raw_data = fetch_all_ad_trials_robust()
    if not raw_data: exit(1)

    df = parse_and_filter_phase1(raw_data)
    if df.empty:
        print("No Phase 1 trials found after filtering.")
        exit(1)

    df = df[df['start_year'] > 0].copy()
    
    # Fill numeric NaNs
    for col in ['enrollment', 'num_arms', 'planned_duration_months']:
        df[col] = df[col].fillna(df[col].median())

    if len(df) < 50:
        split_year = 2018
    else:
        split_year = int(df['start_year'].quantile(0.80))
        if split_year < 2000 or split_year > 2030: split_year = 2018
    
    print(f"Splitting data at year: {split_year}")
    
    train_mask = df['start_year'] < split_year
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    if len(train_df) < 10 or len(test_df) < 5:
        print("Warning: Fallback to random split")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_success'])

    y_train = train_df['is_success']
    y_test = test_df['is_success']

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train Class Dist: {y_train.value_counts().to_dict()}")

    numeric_features = [
        'enrollment', 'num_arms', 'planned_duration_months'
    ]
    
    categorical_features = [
        'is_industry_sponsor', 'is_big_pharma', 
        'is_healthy_volunteers',
        'is_dose_escalation',
        'is_single_dose',
        'is_multiple_dose',
        'has_safety_endpoint',
        'has_pk_endpoint',
        'has_efficacy_endpoint',
        'is_oral',
        'is_iv',
        'has_placebo'
    ]
    
    available_numeric = [f for f in numeric_features if f in train_df.columns]
    available_categorical = [f for f in categorical_features if f in train_df.columns]
    
    # Convert categorical columns to string type to prevent imputer errors
    for col in available_categorical:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
    
    X_train = train_df[available_numeric + available_categorical]
    X_test = test_df[available_numeric + available_categorical]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, available_numeric),
            ('cat', categorical_transformer, available_categorical)
        ])
    
    if y_train.sum() > 0:
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    else:
        ratio = 1.0

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            scale_pos_weight=ratio,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    
    print("\nTraining model (Phase 1)...")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\nResults:")
    try:
        if y_test.nunique() > 1:
            print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
            print(f"PR-AUC:  {average_precision_score(y_test, y_prob):.4f}")
    except: pass
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    try:
        importances = model.named_steps['classifier'].feature_importances_
        ohe = model.named_steps['preprocessor'].named_transformers_['cat']['onehot']
        cat_names = ohe.get_feature_names_out(available_categorical)
        all_feat_names = available_numeric + list(cat_names)
        
        indices = np.argsort(importances)[-15:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [all_feat_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance (Phase 1 Success)')
        plt.tight_layout()
        plt.savefig('phase1_feature_importance.png')
        print("\nSaved 'phase1_feature_importance.png'")
        plt.close()

        if y_test.nunique() > 1:
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label='XGBoost')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve (Phase 1)')
            plt.legend()
            plt.savefig('phase1_pr_curve.png')
            print("Saved 'phase1_pr_curve.png'")
            plt.close()

    except Exception as e:
        print(f"Plotting Error: {e}")
'''
AI Citation and Acknowledgement

For this project, AI assistance was utilized strictly as a productivity tool for
code implementation, debugging, and initial ideation. The core scientific logic,
feature strategy, and final decision-making remained human-driven.

Timeline of AI Usage:

1. Ideation & Feasibility (November 11, 2025)
   Model: Gemini 2 Pro
   - Assisted with initial topic brainstorming.
   - Evaluated the feasibility of data availability for different disease scopes.

2. Logic Discussion & Prototyping (November 22, 2025)
   Models: ChatGPT 5.1, Claude Sonnet 4.5
   - Discussed the logic for the "White-List" labeling approach versus using API status fields.
   - Provided initial syntax examples for regex patterns to extract text-based features.
   - Drafted basic pandas operations for data cleaning.

3. Implementation & Engineering (December 1, 2025)
   Model: Gemini 3 Pro
   - Generated boilerplate code for the ClinicalTrials.gov API fetcher (pagination and retry logic).
   - Refactored the codebase for modularity and standardized comment styles.
   - Debugged syntax errors within the sklearn Pipeline (specifically the ColumnTransformer).
   - Resolved data type inconsistencies (string vs integer) during the imputation process.

Statement of Originality:
The critical intellectual contributions—including the project hypothesis, the specific
selection of Phase features, the decision to use XGBoost, and the final interpretation
of the PR-AUC/Confusion Matrix results—were developed manually by the team. AI served
as a coding assistant for execution rather than a replacement for analytical reasoning.
'''
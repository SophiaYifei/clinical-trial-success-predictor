import matplotlib
# 必须在导入 pyplot 之前设置 backend，解决 SystemError 和 MacOS 兼容性问题
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

# ==========================================
# 1. Configuration & Whitelist
# ==========================================

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# 获批药物白名单 (Ground Truth)
APPROVED_DRUGS_SET = {
    "aducanumab", "aduhelm", "biib037", "biib-037",
    "lecanemab", "leqembi", "ban2401", "ban-2401",
    "donanemab", "kisunla", "ly3002813", "ly-3002813",
    "donepezil", "aricept", "e2020",
    "rivastigmine", "exelon",
    "galantamine", "razadyne", "reminyl",
    "memantine", "namenda", "ebixa",
    "namzaric",
    "brexpiprazole", "rexulti",
    "benzgalantamine", "zunveyl"
}

# 大药企列表 (Big Pharma) - 这是一个非常好的特征，反映资源实力
BIG_PHARMA_LIST = [
    'biogen', 'lilly', 'pfizer', 'roche', 'genentech', 'merck', 
    'abbvie', 'novartis', 'janssen', 'eisai', 'astrazeneca', 'bayer',
    'gsk', 'glaxosmithkline', 'sanofi', 'takeda', 'bristol', 'myers',
    'squibb', 'bms', 'amgen', 'gilead', 'regeneron'
]

# ==========================================
# 2. ETL Module: Robust Data Fetching
# ==========================================

def fetch_all_ad_trials():
    studies = []
    next_page_token = None
    max_retries = 3
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }

    print("🚀 Starting Data Extraction...")
    
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
                break  # Success
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"❌ Error fetching data: {e}")
                    return studies
        
        if not next_page_token:
            break
            
    print(f"✅ Download Complete. Total Studies: {len(studies)}")
    return studies

# ==========================================
# 3. Feature Engineering Helper Functions
# ==========================================

def extract_planned_duration(outcomes):
    """
    无泄露时长提取：只从 timeFrame 文本中提取计划时长。
    不使用 completionDate。
    """
    if outcomes is None or not outcomes:
        return None
    
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
                
                if weeks > max_weeks:
                    max_weeks = weeks
            except:
                continue
    
    return max_weeks / MONTHS_TO_WEEKS if max_weeks > 0 else None

def is_big_pharma(sponsor_name):
    """Check if sponsor is a Big Pharma company"""
    if not sponsor_name: return 0
    sponsor_lower = str(sponsor_name).lower()
    return 1 if any(company in sponsor_lower for company in BIG_PHARMA_LIST) else 0

def is_antibody(intervention_name):
    """Check if intervention is an antibody (ends with -mab)"""
    if not intervention_name: return 0
    name_lower = str(intervention_name).lower()
    return 1 if name_lower.endswith('mab') else 0

def classify_substance_type(intervention_name):
    """Simple substance classification"""
    if not intervention_name: return 'Small_Molecule'
    name_lower = str(intervention_name).lower()
    if (name_lower.endswith('mab') or 'monoclonal' in name_lower or 'antibody' in name_lower):
        return 'Biologic'
    if ('extract' in name_lower or 'vitamin' in name_lower):
        return 'Natural_Product'
    return 'Small_Molecule'

# ==========================================
# 4. Parsing Logic
# ==========================================

def parse_and_filter_studies(studies_raw):
    rows = []
    print("🧹 Filtering data locally...")
    
    for study in studies_raw:
        if not isinstance(study, dict): continue
            
        proto = study.get("protocolSection", {}) or {}
        design = proto.get("designModule", {}) or {}
        status_mod = proto.get("statusModule", {}) or {}
        
        # --- Filter ---
        if design.get("studyType") != "INTERVENTIONAL": continue
        phases = design.get("phases", []) or []
        if not isinstance(phases, list) or "PHASE3" not in phases: continue
        
        overall_status = str(status_mod.get("overallStatus", "")).upper()
        if overall_status not in ['COMPLETED', 'TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
            continue

        row = {}
        row['nct_id'] = proto.get("identificationModule", {}).get("nctId")
        
        # Start Year
        start_date_struct = status_mod.get("startDateStruct", {}) or {}
        start_date_str = str(start_date_struct.get("date", "") or "")
        match = re.search(r'\d{4}', start_date_str)
        row['start_year'] = int(match.group()) if match else 0

        # --- Label Generation (Target) - FIXED Logic ---
        interventions = proto.get("armsInterventionsModule", {}) .get("interventions", []) or []
        drug_names_in_study = []
        is_success = 0
        
        # Check if any drug in the study is in the whitelist
        is_approved_drug = False
        for item in interventions:
            if not isinstance(item, dict): continue
            if item.get("type") == "DRUG":
                name = str(item.get("name", "") or "").lower()
                if name:
                    drug_names_in_study.append(name)
                    for approved in APPROVED_DRUGS_SET:
                        if approved in name:
                            is_approved_drug = True
                            break
        
        # Strict Label Logic: Must be Approved Drug AND Completed Trial
        if is_approved_drug and overall_status == 'COMPLETED':
            is_success = 1
        else:
            is_success = 0
        
        row['is_success'] = is_success
        
        # Extract primary drug name for features (NOT for label)
        primary_drug_name = drug_names_in_study[0] if drug_names_in_study else ""

        # --- Features ---
        
        # Sponsor
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {}) or {}
        lead_sponsor = sponsor_mod.get("leadSponsor", {}) or {}
        sponsor_class = str(lead_sponsor.get("class", "UNKNOWN") or "UNKNOWN")
        sponsor_name = str(lead_sponsor.get("name", "") or "")
        
        row['is_industry_sponsor'] = 1 if sponsor_class == "INDUSTRY" else 0
        row['is_big_pharma'] = is_big_pharma(sponsor_name)

        # Enrollment
        enrollment_info = design.get("enrollmentInfo", {}) or {}
        row['enrollment'] = enrollment_info.get("count")
        
        # Planned Duration (Safe)
        outcomes = proto.get("outcomesModule", {}).get("primaryOutcomes", []) or []
        row['planned_duration_months'] = extract_planned_duration(outcomes)

        # Design
        arm_groups = proto.get("armsInterventionsModule", {}).get("armGroups", []) or []
        row['num_arms'] = len(arm_groups) if isinstance(arm_groups, list) else 0
        
        design_info = design.get("designInfo", {}) or {}
        row['has_dmc'] = 1 if design_info.get("hasDmc") is True else 0
        
        masking_info = design_info.get("maskingInfo", {}) or {}
        masking = str(masking_info.get("masking", "") or "").lower()
        row['is_blinded'] = 1 if ('double' in masking or 'quadruple' in masking) else 0

        # Criteria Keywords
        eligibility = proto.get("eligibilityModule", {}) or {}
        criteria = str(eligibility.get("eligibilityCriteria", "") or "").lower()
        
        # Mechanism Features (Safe Domain Knowledge)
        row['criteria_biomarker'] = 1 if any(x in criteria for x in ['amyloid', 'tau', 'pet', 'csf']) else 0
        row['target_amyloid'] = 1 if any(x in criteria for x in ['amyloid', 'beta', 'aβ', 'abeta']) else 0
        row['target_tau'] = 1 if 'tau' in criteria else 0
        row['criteria_early_stage'] = 1 if any(x in criteria for x in ['mci', 'prodromal', 'early']) else 0
        
        # Min Age
        min_age_str = str(eligibility.get("minimumAge", "0") or "0")
        match_age = re.search(r'\d+', min_age_str)
        try:
            row['min_age'] = int(match_age.group()) if match_age else 60
        except:
            row['min_age'] = 60

        # Endpoint
        outcome_text = " ".join([str(o.get("measure", "")) for o in outcomes if isinstance(o, dict)]).lower()
        row['endpoint_cognitive'] = 1 if any(x in outcome_text for x in ['adas', 'mmse', 'cdr', 'cognit']) else 0

        # Drug Type Features (Safe)
        row['substance_type'] = classify_substance_type(primary_drug_name)
        row['is_antibody'] = is_antibody(primary_drug_name)

        rows.append(row)
    
    return pd.DataFrame(rows)

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    # 1. Fetch & Parse
    raw_data = fetch_all_ad_trials()
    if not raw_data: exit(1)

    df = parse_and_filter_studies(raw_data)
    if df.empty: exit(1)

    # 2. Preprocess
    df = df[df['start_year'] > 0].copy()
    
    # 3. Train/Test Split (Time-based)
    if len(df) < 20:
        split_year = 2018
    else:
        split_year = int(df['start_year'].quantile(0.80)) # 80% split
        if split_year < 2000 or split_year > 2030: split_year = 2018
    
    print(f"✂️  Splitting data at year: {split_year}")
    
    train_mask = df['start_year'] < split_year
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    # Fallback to random if time split fails
    if len(train_df) < 10 or len(test_df) < 5:
        print("⚠️ Fallback to Random Split")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_success'])

    y_train = train_df['is_success']
    y_test = test_df['is_success']

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train Class Dist: {y_train.value_counts().to_dict()}")

    # 4. Feature Selection (Removed Leakage Features!)
    numeric_features = [
        'enrollment', 'planned_duration_months', 'min_age', 'num_arms'
    ]
    
    categorical_features = [
        'is_industry_sponsor', 'is_big_pharma', 'has_dmc', 'is_blinded',
        'criteria_biomarker', 'target_amyloid', 'target_tau', 
        'criteria_early_stage', 'endpoint_cognitive', 'substance_type', 'is_antibody'
    ]
    
    # Verify features exist
    available_numeric = [f for f in numeric_features if f in train_df.columns]
    available_categorical = [f for f in categorical_features if f in train_df.columns]
    
    X_train = train_df[available_numeric + available_categorical]
    X_test = test_df[available_numeric + available_categorical]

    # 5. Pipeline & Model
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
    
    # Handle Imbalance
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
    
    print("\n🔧 Training Model...")
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\n🏆 Results:")
    try:
        if y_test.nunique() > 1:
            print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
            print(f"PR-AUC:  {average_precision_score(y_test, y_prob):.4f}")
    except: pass
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Plotting (Save to file instead of show)
    try:
        # Feature Importance Plot
        importances = model.named_steps['classifier'].feature_importances_
        
        # Get feature names from OneHotEncoder
        ohe = model.named_steps['preprocessor'].named_transformers_['cat']['onehot']
        cat_names = ohe.get_feature_names_out(available_categorical)
        all_feat_names = available_numeric + list(cat_names)
        
        indices = np.argsort(importances)[-15:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [all_feat_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig('ym_feature_importance.png') # Save file
        print("\n✅ Saved 'ym_feature_importance.png'")
        plt.close()

        # PR Curve Plot
        if y_test.nunique() > 1:
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label='XGBoost')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.savefig('ym_pr_curve.png') # Save file
            print("✅ Saved 'ym_pr_curve.png'")
            plt.close()

    except Exception as e:
        print(f"Plotting Error: {e}")
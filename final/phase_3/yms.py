import matplotlib
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
                break
                
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
# 3. Enhanced Feature Engineering Helpers
# ==========================================

def extract_planned_duration(outcomes):
    """Extract planned duration from Primary Outcome timeFrame text"""
    if outcomes is None or not outcomes:
        return None, 0
    
    max_weeks = 0
    MONTHS_TO_WEEKS = 52 / 12
    pattern = r'(\d+)\s*(week|month|day|year)s?\b'
    
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        
        time_frame = str(outcome.get("timeFrame", "") or "").lower()
        if not time_frame:
            continue
        
        matches = re.findall(pattern, time_frame)
        for num, unit in matches:
            try:
                val = float(num)
                if 'year' in unit:
                    weeks = val * 52
                elif 'month' in unit:
                    weeks = val * MONTHS_TO_WEEKS
                elif 'day' in unit:
                    weeks = val / 7
                else:
                    weeks = val
                
                if weeks > max_weeks:
                    max_weeks = weeks
            except:
                continue
    
    duration_months = max_weeks / MONTHS_TO_WEEKS if max_weeks > 0 else None
    return duration_months, max_weeks

def count_criteria_items(criteria_text):
    """Count inclusion and exclusion criteria items"""
    if not criteria_text:
        return 0, 0
    
    criteria_lower = str(criteria_text).lower()
    
    # Try to find inclusion/exclusion sections
    inclusion_match = re.search(
        r'inclusion criteria:?(.+?)(?:exclusion criteria|$)',
        criteria_lower, re.IGNORECASE | re.DOTALL
    )
    exclusion_match = re.search(
        r'exclusion criteria:?(.+)',
        criteria_lower, re.IGNORECASE | re.DOTALL
    )
    
    inclusion_count = 0
    exclusion_count = 0
    
    if inclusion_match:
        inclusion_text = inclusion_match.group(1)
        # Count bullet points or numbered items
        inclusion_count = len(re.findall(r'[\n\r]\s*[-•*\d]', inclusion_text))
    
    if exclusion_match:
        exclusion_text = exclusion_match.group(1)
        exclusion_count = len(re.findall(r'[\n\r]\s*[-•*\d]', exclusion_text))
    
    # Fallback: count total length as proxy
    if inclusion_count == 0 and exclusion_count == 0:
        total_length = len(criteria_text)
        inclusion_count = max(1, total_length // 200)  # Rough estimate
        exclusion_count = max(1, total_length // 300)
    
    return inclusion_count, exclusion_count

def extract_age_range(eligibility):
    """Extract min and max age, calculate range"""
    min_age_str = str(eligibility.get("minimumAge", "0") or "0")
    max_age_str = str(eligibility.get("maximumAge", "N/A") or "N/A")
    
    match_min = re.search(r'\d+', min_age_str)
    match_max = re.search(r'\d+', max_age_str)
    
    try:
        min_age = int(match_min.group()) if match_min else 60
    except:
        min_age = 60
    
    try:
        max_age = int(match_max.group()) if match_max and max_age_str != "N/A" else 100
    except:
        max_age = 100
    
    age_range = max_age - min_age if max_age > min_age else 0
    
    return min_age, max_age, age_range

def is_big_pharma(sponsor_name):
    """Check if sponsor is a Big Pharma company"""
    if not sponsor_name:
        return 0
    sponsor_lower = str(sponsor_name).lower()
    return 1 if any(company in sponsor_lower for company in BIG_PHARMA_LIST) else 0

def is_antibody(intervention_name):
    """Check if intervention is an antibody"""
    if not intervention_name:
        return 0
    name_lower = str(intervention_name).lower()
    return 1 if name_lower.endswith('mab') else 0

def classify_substance_type(intervention_name):
    """Classify substance type"""
    if not intervention_name:
        return 'Small_Molecule'
    name_lower = str(intervention_name).lower()
    if (name_lower.endswith('mab') or 'monoclonal' in name_lower or 'antibody' in name_lower):
        return 'Biologic'
    if ('extract' in name_lower or 'vitamin' in name_lower):
        return 'Natural_Product'
    return 'Small_Molecule'

def extract_location_features(locations_module):
    """Extract geographic and site features"""
    if not isinstance(locations_module, dict):
        locations = []
    else:
        locations = locations_module.get("locations", []) or []
    
    countries = set()
    us_states = set()
    
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        
        # Try multiple paths to extract country and state
        country = ""
        state = ""
        
        # Path 1: loc -> facility -> address -> country/state
        facility = loc.get("facility", {}) or {}
        if isinstance(facility, dict):
            address = facility.get("address", {}) or {}
            if isinstance(address, dict):
                country = str(address.get("country", "") or "").strip()
                state = str(address.get("state", "") or "").strip()
        
        # Path 2: loc -> country/state directly
        if not country:
            country = str(loc.get("country", "") or "").strip()
        if not state:
            state = str(loc.get("state", "") or "").strip()
        
        # Path 3: facility -> country/state directly
        if not country and isinstance(facility, dict):
            country = str(facility.get("country", "") or "").strip()
        if not state and isinstance(facility, dict):
            state = str(facility.get("state", "") or "").strip()
        
        if country:
            countries.add(country)
        
        if country == "United States" and state:
            us_states.add(state)
    
    num_locations = len(locations)
    num_countries = len(countries)
    num_us_states = len(us_states)
    is_international = 1 if num_countries > 1 else 0
    includes_us = 1 if "United States" in countries else 0
    includes_europe = 1 if any(c in countries for c in [
        'United Kingdom', 'Germany', 'France', 'Spain', 'Italy',
        'Netherlands', 'Belgium', 'Switzerland'
    ]) else 0
    includes_asia = 1 if any(c in countries for c in [
        'Japan', 'China', 'Korea', 'Taiwan', 'Singapore', 'India'
    ]) else 0
    is_multisite = 1 if num_locations > 1 else 0
    is_large_network = 1 if num_locations > 20 else 0
    
    return {
        'num_locations': num_locations,
        'num_countries': num_countries,
        'num_us_states': num_us_states,
        'is_international': is_international,
        'includes_us': includes_us,
        'includes_europe': includes_europe,
        'includes_asia': includes_asia,
        'is_multisite': is_multisite,
        'is_large_network': is_large_network
    }

# ==========================================
# 4. Enhanced Parsing Logic
# ==========================================

def parse_and_filter_studies(studies_raw):
    rows = []
    print("🧹 Filtering data locally...")
    
    for study in studies_raw:
        if not isinstance(study, dict):
            continue
            
        proto = study.get("protocolSection", {}) or {}
        design = proto.get("designModule", {}) or {}
        status_mod = proto.get("statusModule", {}) or {}
        
        # Filter
        if design.get("studyType") != "INTERVENTIONAL":
            continue
        phases = design.get("phases", []) or []
        if not isinstance(phases, list) or "PHASE3" not in phases:
            continue
        
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

        # Label Generation
        interventions_module = proto.get("armsInterventionsModule", {}) or {}
        interventions = interventions_module.get("interventions", []) or []
        drug_names_in_study = []
        is_success = 0
        
        for item in interventions:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "DRUG":
                name = str(item.get("name", "") or "").lower()
                if name:
                    drug_names_in_study.append(name)
                    for approved in APPROVED_DRUGS_SET:
                        if approved in name:
                            is_success = 1
                            break
        
        row['is_success'] = is_success
        primary_drug_name = drug_names_in_study[0] if drug_names_in_study else ""

        # ============================================================
        # ENHANCED FEATURES (Based on Shreya's approach)
        # ============================================================
        
        # Sponsor Features
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {}) or {}
        lead_sponsor = sponsor_mod.get("leadSponsor", {}) or {}
        sponsor_class = str(lead_sponsor.get("class", "UNKNOWN") or "UNKNOWN")
        sponsor_name = str(lead_sponsor.get("name", "") or "")
        
        row['is_industry_sponsor'] = 1 if sponsor_class == "INDUSTRY" else 0
        row['is_big_pharma'] = is_big_pharma(sponsor_name)
        
        # Collaborators
        collaborators = sponsor_mod.get("collaborators", []) or []
        row['num_collaborators'] = len(collaborators) if isinstance(collaborators, list) else 0
        row['has_industry_collab'] = 0
        row['has_nih_collab'] = 0
        
        for collab in collaborators:
            if not isinstance(collab, dict):
                continue
            collab_class = str(collab.get("class", "") or "").upper()
            if collab_class == "INDUSTRY":
                row['has_industry_collab'] = 1
            elif collab_class == "NIH":
                row['has_nih_collab'] = 1

        # Enrollment
        enrollment_info = design.get("enrollmentInfo", {}) or {}
        row['enrollment'] = enrollment_info.get("count")

        # Outcome Measures Features
        outcomes_module = proto.get("outcomesModule", {}) or {}
        primary_outcomes = outcomes_module.get("primaryOutcomes", []) or []
        secondary_outcomes = outcomes_module.get("secondaryOutcomes", []) or []
        
        row['num_primary_outcomes'] = len(primary_outcomes) if isinstance(primary_outcomes, list) else 0
        row['num_secondary_outcomes'] = len(secondary_outcomes) if isinstance(secondary_outcomes, list) else 0
        row['total_outcome_measures'] = row['num_primary_outcomes'] + row['num_secondary_outcomes']
        
        # Extract outcome text for endpoint detection
        primary_text = " ".join([
            str(o.get("measure", "")) for o in primary_outcomes if isinstance(o, dict)
        ]).lower()
        
        secondary_text = " ".join([
            str(o.get("measure", "")) for o in secondary_outcomes if isinstance(o, dict)
        ]).lower()
        
        all_outcome_text = primary_text + " " + secondary_text
        
        # Endpoint type detection
        cognitive_keywords = ['adas', 'mmse', 'cognition', 'cognitive', 'memory', 'cdr']
        row['has_cognitive_endpoint'] = 1 if any(kw in all_outcome_text for kw in cognitive_keywords) else 0
        
        biomarker_keywords = ['amyloid', 'tau', 'pet', 'csf', 'biomarker', 'plasma']
        row['has_biomarker_endpoint'] = 1 if any(kw in all_outcome_text for kw in biomarker_keywords) else 0
        
        functional_keywords = ['adl', 'activities of daily living', 'function', 'iadl']
        row['has_functional_endpoint'] = 1 if any(kw in all_outcome_text for kw in functional_keywords) else 0
        
        # Duration features
        duration_months, max_weeks = extract_planned_duration(primary_outcomes)
        row['planned_duration_months'] = duration_months
        row['max_followup_weeks'] = max_weeks
        row['has_longterm_followup'] = 1 if max_weeks > 24 else 0  # > 6 months

        # Design Features
        arm_groups = interventions_module.get("armGroups", []) or []
        row['num_arms'] = len(arm_groups) if isinstance(arm_groups, list) else 0
        
        design_info = design.get("designInfo", {}) or {}
        row['has_dmc'] = 1 if design_info.get("hasDmc") is True else 0
        
        # Enhanced Blinding Features
        masking_info = design_info.get("maskingInfo", {}) or {}
        masking = str(masking_info.get("masking", "") or "").lower()
        who_masked = masking_info.get("whoMasked", []) or []
        who_masked_str = " ".join([str(w) for w in who_masked]).lower()
        
        row['is_blinded'] = 1 if ('double' in masking or 'triple' in masking or 'quadruple' in masking) else 0
        row['is_triple_blind'] = 1 if 'triple' in masking else 0
        row['is_quadruple_blind'] = 1 if 'quadruple' in masking else 0
        row['masks_participant'] = 1 if 'participant' in who_masked_str else 0
        row['masks_investigator'] = 1 if 'investigator' in who_masked_str else 0
        row['masks_outcomes_assessor'] = 1 if 'outcomes assessor' in who_masked_str else 0
        
        # Study Model
        intervention_model = str(design_info.get("interventionModel", "") or "").lower()
        row['is_parallel'] = 1 if 'parallel' in intervention_model else 0
        row['is_crossover'] = 1 if 'crossover' in intervention_model else 0
        row['is_factorial'] = 1 if 'factorial' in intervention_model else 0

        # Intervention Features
        intervention_names = [str(i.get("name", "") or "") for i in interventions if isinstance(i, dict)]
        intervention_names_lower = " ".join(intervention_names).lower()
        
        row['has_placebo'] = 1 if 'placebo' in intervention_names_lower else 0
        
        intervention_descriptions = [str(i.get("description", "") or "") for i in interventions if isinstance(i, dict)]
        desc_text = " ".join(intervention_descriptions).lower()
        row['has_dose_escalation'] = 1 if ('dose escalation' in desc_text or 'dose finding' in desc_text) else 0
        row['has_multiple_doses'] = 1 if desc_text.count('dose') > 2 or desc_text.count('mg') > 2 else 0

        # Eligibility Features
        eligibility = proto.get("eligibilityModule", {}) or {}
        criteria = str(eligibility.get("eligibilityCriteria", "") or "")
        criteria_lower = criteria.lower()
        
        # Count criteria items
        num_inclusion, num_exclusion = count_criteria_items(criteria)
        row['num_inclusion_criteria'] = num_inclusion
        row['num_exclusion_criteria'] = num_exclusion
        row['criteria_length'] = len(criteria)
        
        # Mechanism and Population Features
        row['criteria_biomarker'] = 1 if any(x in criteria_lower for x in ['amyloid', 'tau', 'pet', 'csf']) else 0
        row['target_amyloid'] = 1 if any(x in criteria_lower for x in ['amyloid', 'beta', 'aβ', 'abeta']) else 0
        row['target_tau'] = 1 if 'tau' in criteria_lower else 0
        row['requires_biomarker'] = 1 if any(kw in criteria_lower for kw in ['amyloid', 'pet positive', 'csf', 'apoe']) else 0
        row['requires_genetic_test'] = 1 if ('apoe' in criteria_lower or 'genetic' in criteria_lower) else 0
        
        # AD Stage Targeting
        row['targets_mci'] = 1 if ('mci' in criteria_lower or 'mild cognitive impairment' in criteria_lower) else 0
        row['targets_mild_ad'] = 1 if ('mild alzheimer' in criteria_lower or 'mild ad' in criteria_lower) else 0
        row['targets_moderate_ad'] = 1 if 'moderate' in criteria_lower else 0
        row['criteria_early_stage'] = 1 if any(x in criteria_lower for x in ['mci', 'prodromal', 'early']) else 0
        
        # Age Features
        min_age, max_age, age_range = extract_age_range(eligibility)
        row['min_age'] = min_age
        row['max_age'] = max_age
        row['age_range'] = age_range

        # Drug Type Features
        row['substance_type'] = classify_substance_type(primary_drug_name)
        row['is_antibody'] = is_antibody(primary_drug_name)

        # Geographic Features
        locations_module = proto.get("contactsLocationsModule", {}) or {}
        location_features = extract_location_features(locations_module)
        row.update(location_features)

        rows.append(row)
    
    return pd.DataFrame(rows)

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    # 1. Fetch & Parse
    raw_data = fetch_all_ad_trials()
    if not raw_data:
        exit(1)

    df = parse_and_filter_studies(raw_data)
    if df.empty:
        exit(1)

    # 2. Preprocess
    df = df[df['start_year'] > 0].copy()
    
    # Fill missing values for numeric features
    numeric_cols_to_fill = [
        'enrollment', 'planned_duration_months', 'min_age', 'max_age',
        'num_arms', 'num_primary_outcomes', 'num_secondary_outcomes',
        'num_inclusion_criteria', 'num_exclusion_criteria',
        'num_locations', 'num_countries', 'num_us_states', 'num_collaborators'
    ]
    
    for col in numeric_cols_to_fill:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_val)
    
    # Ensure numeric types
    for col in numeric_cols_to_fill:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 3. Train/Test Split (Time-based)
    if len(df) < 20:
        split_year = 2018
    else:
        split_year = int(df['start_year'].quantile(0.80))
        if split_year < 2000 or split_year > 2030:
            split_year = 2018
    
    print(f"✂️  Splitting data at year: {split_year}")
    
    train_mask = df['start_year'] < split_year
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    # Fallback to random if time split fails
    if len(train_df) < 10 or len(test_df) < 5:
        print("⚠️ Fallback to Random Split")
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42,
            stratify=df['is_success'] if df['is_success'].nunique() > 1 else None
        )

    y_train = train_df['is_success']
    y_test = test_df['is_success']

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train Class Dist: {y_train.value_counts().to_dict()}")

    # 4. Enhanced Feature Selection
    numeric_features = [
        'enrollment', 'planned_duration_months', 'min_age', 'max_age', 'age_range',
        'num_arms', 'num_primary_outcomes', 'num_secondary_outcomes', 'total_outcome_measures',
        'max_followup_weeks', 'num_inclusion_criteria', 'num_exclusion_criteria', 'criteria_length',
        'num_locations', 'num_countries', 'num_us_states', 'num_collaborators'
    ]
    
    categorical_features = [
        'is_industry_sponsor', 'is_big_pharma', 'has_industry_collab', 'has_nih_collab',
        'has_dmc', 'is_blinded', 'is_triple_blind', 'is_quadruple_blind',
        'masks_participant', 'masks_investigator', 'masks_outcomes_assessor',
        'is_parallel', 'is_crossover', 'is_factorial',
        'has_placebo', 'has_dose_escalation', 'has_multiple_doses',
        'has_cognitive_endpoint', 'has_biomarker_endpoint', 'has_functional_endpoint',
        'has_longterm_followup',
        'criteria_biomarker', 'target_amyloid', 'target_tau',
        'requires_biomarker', 'requires_genetic_test',
        'targets_mci', 'targets_mild_ad', 'targets_moderate_ad', 'criteria_early_stage',
        'is_international', 'includes_us', 'includes_europe', 'includes_asia',
        'is_multisite', 'is_large_network',
        'substance_type', 'is_antibody'
    ]
    
    # Verify features exist
    available_numeric = [f for f in numeric_features if f in train_df.columns]
    available_categorical = [f for f in categorical_features if f in train_df.columns]
    
    print(f"\n📊 Feature Summary:")
    print(f"  Numeric features: {len(available_numeric)}")
    print(f"  Categorical features: {len(available_categorical)}")
    print(f"  Total: {len(available_numeric) + len(available_categorical)}")
    
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
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"PR-AUC:  {pr_auc:.4f}")
    except:
        pass
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Plotting
    try:
        # Feature Importance Plot
        importances = model.named_steps['classifier'].feature_importances_
        
        # Get feature names from OneHotEncoder
        ohe = model.named_steps['preprocessor'].named_transformers_['cat']['onehot']
        cat_names = ohe.get_feature_names_out(available_categorical)
        all_feat_names = available_numeric + list(cat_names)
        
        indices = np.argsort(importances)[-20:]  # Top 20 features
        
        plt.figure(figsize=(12, 10))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [all_feat_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance (Enhanced Features)')
        plt.tight_layout()
        plt.savefig('yms_feature_importance.png')
        print("\n✅ Saved 'yms_feature_importance.png'")
        plt.close()

        # PR Curve Plot
        if y_test.nunique() > 1:
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            baseline = y_test.mean()
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'XGBoost (AP={pr_auc:.3f})')
            plt.axhline(baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('yms_pr_curve.png')
            print("✅ Saved 'yms_pr_curve.png'")
            plt.close()

    except Exception as e:
        print(f"Plotting Error: {e}")


"""
Data preprocessing and feature engineering module
Converts raw API JSON to structured DataFrame with features
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import json
import os


def flatten_study(study: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a single study dict into a structured row with all features
    
    Args:
        study: Raw study dictionary from API
    
    Returns:
        Flattened dictionary with extracted features
    """
    protocol = study.get("protocolSection", {}) or {}
    ident = protocol.get("identificationModule", {}) or {}
    desc = protocol.get("descriptionModule", {}) or {}
    conds = protocol.get("conditionsModule", {}) or {}
    design = protocol.get("designModule", {}) or {}
    status = protocol.get("statusModule", {}) or {}
    eligibility = protocol.get("eligibilityModule", {}) or {}
    interventions = protocol.get("armsInterventionsModule", {}) or {}
    outcomes = protocol.get("outcomesModule", {}) or {}
    sponsor = protocol.get("sponsorCollaboratorsModule", {}) or {}
    locations = protocol.get("contactsLocationsModule", {}) or {}
    
    # Extract design info
    design_info = design.get("designInfo", {}) or {}
    
    # Extract sponsor info
    lead_sponsor = sponsor.get("leadSponsor", {}) or {}
    
    # Extract enrollment
    enrollment_info = design.get("enrollmentInfo", {}) or {}
    
    # Extract dates
    start_struct = status.get("startDateStruct", {}) or {}
    primary_struct = status.get("primaryCompletionDateStruct", {}) or {}
    completion_struct = status.get("completionDateStruct", {}) or {}
    
    # Extract interventions
    intervention_list = interventions.get("interventions", []) or []
    intervention_names = []
    intervention_types = []
    for interv in intervention_list:
        if isinstance(interv, dict):
            intervention_names.append(interv.get("name", ""))
            intervention_types.append(interv.get("type", ""))
    
    # Extract arms (if available, otherwise estimate from interventions)
    arms_list = interventions.get("arms", []) or []
    arm_count = len(arms_list)
    # If arms not available, estimate from intervention count
    if arm_count == 0 and intervention_names:
        # Rough estimate: usually 1-2 arms per intervention type
        arm_count = max(1, len(set(intervention_types)))
    
    # Extract outcomes
    primary_outcomes = outcomes.get("primaryOutcomes", []) or []
    secondary_outcomes = outcomes.get("secondaryOutcomes", []) or []
    
    # Extract locations
    locations_list = locations.get("locations", []) or []
    location_count = len(locations_list)
    
    # Calculate trial duration (in months)
    trial_duration_months = None
    if start_struct.get("date") and primary_struct.get("date"):
        try:
            start_date = pd.to_datetime(start_struct.get("date"), errors='coerce')
            end_date = pd.to_datetime(primary_struct.get("date"), errors='coerce')
            if pd.notna(start_date) and pd.notna(end_date):
                trial_duration_months = (end_date - start_date).days / 30.0
        except:
            pass
    
    row = {
        # Core identification
        "nct_id": ident.get("nctId") or study.get("nctId"),
        "brief_title": ident.get("briefTitle") or study.get("briefTitle"),
        "acronym": ident.get("acronym"),
        
        # Text features
        "brief_summary": desc.get("briefSummary") or "",
        "detailed_description": desc.get("detailedDescription") or "",
        
        # Conditions
        "conditions": " | ".join(conds.get("conditions", [])) if conds.get("conditions") else "",
        
        # Design features
        "phase": " | ".join(design.get("phases", [])) if design.get("phases") else None,
        "study_type": design.get("studyType"),
        "enrollment": enrollment_info.get("count"),
        "allocation": design_info.get("allocation"),
        "masking": design_info.get("maskingInfo", {}).get("maskingType") if isinstance(design_info.get("maskingInfo"), dict) else None,
        "intervention_model": design_info.get("interventionModel"),
        
        # Status
        "overall_status": status.get("overallStatus") or study.get("overallStatus"),
        "has_results": study.get("hasResults", False),
        "start_date": start_struct.get("date"),
        "primary_completion_date": primary_struct.get("date"),
        "completion_date": completion_struct.get("date"),
        "trial_duration_months": trial_duration_months,
        
        # Eligibility
        "minimum_age": eligibility.get("minimumAge"),
        "maximum_age": eligibility.get("maximumAge"),
        "sex": eligibility.get("sex"),
        "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
        
        # Interventions
        "intervention_names": " | ".join(intervention_names) if intervention_names else "",
        "intervention_types": " | ".join(set(intervention_types)) if intervention_types else "",
        "intervention_count": len(intervention_names),
        "arm_count": arm_count,
        
        # Outcomes
        "primary_outcome_count": len(primary_outcomes),
        "secondary_outcome_count": len(secondary_outcomes),
        
        # Sponsor
        "sponsor_name": lead_sponsor.get("name"),
        "sponsor_class": lead_sponsor.get("class"),
        
        # Locations
        "location_count": location_count,
    }
    
    return row


def build_trials_dataframe(raw_studies: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of raw study dicts into a structured DataFrame
    
    Args:
        raw_studies: List of raw study dictionaries from API
    
    Returns:
        DataFrame with all extracted features
    """
    if not raw_studies:
        print("Warning: No studies provided, returning empty DataFrame")
        return pd.DataFrame()
    
    flattened = [flatten_study(s) for s in raw_studies]
    df = pd.DataFrame(flattened)
    
    # Check if DataFrame is empty or has no nct_id column
    if df.empty:
        print("Warning: Empty DataFrame after flattening")
        return df
    
    if "nct_id" not in df.columns:
        print("Warning: 'nct_id' column not found in DataFrame")
        return df
    
    # Basic cleaning
    df = df.dropna(subset=["nct_id"]).drop_duplicates(subset=["nct_id"])
    
    return df


def add_success_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create success label based on trial status and results availability
    Success = COMPLETED status AND has results reported
    
    Args:
        df: DataFrame with trial data
    
    Returns:
        DataFrame with success_label column added
    """
    df = df.copy()
    df["has_results"] = df["has_results"].fillna(False)
    
    def label_row(row):
        status = str(row.get("overall_status") or "").upper()
        has_res = bool(row.get("has_results"))
        
        # Success: completed and has results
        if status == "COMPLETED" and has_res:
            return 1
        # Failure: terminated, suspended, withdrawn, or completed without results
        elif status in ["TERMINATED", "SUSPENDED", "WITHDRAWN"]:
            return 0
        elif status == "COMPLETED" and not has_res:
            return 0
        # Unknown/ongoing: return -1 for filtering
        else:
            return -1
    
    df["success_label"] = df.apply(label_row, axis=1)
    
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for modeling: handle missing values and create derived features
    
    Args:
        df: DataFrame with raw features
    
    Returns:
        DataFrame with prepared features
    """
    df = df.copy()
    
    # Fill missing text features
    df["brief_title"] = df["brief_title"].fillna("")
    df["brief_summary"] = df["brief_summary"].fillna("")
    df["detailed_description"] = df["detailed_description"].fillna("")
    df["eligibility_criteria"] = df["eligibility_criteria"].fillna("")
    
    # Create combined text feature
    df["text_features"] = (
        df["brief_title"] + " " + 
        df["brief_summary"] + " " + 
        df["detailed_description"]
    ).str.strip()
    
    # Fill missing categorical features
    df["phase"] = df["phase"].fillna("UNKNOWN")
    df["study_type"] = df["study_type"].fillna("UNKNOWN")
    df["overall_status"] = df["overall_status"].fillna("UNKNOWN")
    df["allocation"] = df["allocation"].fillna("UNKNOWN")
    df["masking"] = df["masking"].fillna("UNKNOWN")
    df["intervention_model"] = df["intervention_model"].fillna("UNKNOWN")
    df["sponsor_class"] = df["sponsor_class"].fillna("UNKNOWN")
    df["sex"] = df["sex"].fillna("UNKNOWN")
    
    # Fill missing numerical features with median or 0
    df["enrollment"] = df["enrollment"].fillna(df["enrollment"].median() if df["enrollment"].notna().any() else 0)
    df["trial_duration_months"] = df["trial_duration_months"].fillna(
        df["trial_duration_months"].median() if df["trial_duration_months"].notna().any() else 0
    )
    df["intervention_count"] = df["intervention_count"].fillna(0)
    df["arm_count"] = df["arm_count"].fillna(0)
    df["primary_outcome_count"] = df["primary_outcome_count"].fillna(0)
    df["secondary_outcome_count"] = df["secondary_outcome_count"].fillna(0)
    df["location_count"] = df["location_count"].fillna(0)
    
    # Create binary features
    df["is_randomized"] = (df["allocation"] == "RANDOMIZED").astype(int)
    df["is_blinded"] = df["masking"].str.contains("BLIND", case=False, na=False).astype(int)
    df["is_industry_sponsored"] = (df["sponsor_class"] == "INDUSTRY").astype(int)
    
    return df


def load_and_preprocess_data(
    raw_data_path: str = "data/raw/trials_raw.json",
    output_path: str = "data/processed/trials_processed.csv",
    filter_phase: str = None
) -> pd.DataFrame:
    """
    Load raw JSON data and preprocess into final DataFrame
    
    Args:
        raw_data_path: Path to raw JSON file
        output_path: Path to save processed CSV
    
    Returns:
        Processed DataFrame ready for modeling
    """
    # Load raw data
    if raw_data_path.endswith('.json'):
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            raw_studies = json.load(f)
    else:
        raise ValueError("Only JSON format supported for raw data")
    
    # Build DataFrame
    df = build_trials_dataframe(raw_studies)
    
    if df.empty:
        raise ValueError(
            "No data available after preprocessing. "
            "Please check:\n"
            "1. API query returned data\n"
            "2. Data file exists and is not empty\n"
            "3. API parameters are correct"
        )
    
    print(f"Initial DataFrame shape: {df.shape}")
    
    # Add labels
    df = add_success_label(df)
    
    # Filter out unknown labels (ongoing trials)
    df_labeled = df[df["success_label"] != -1].copy()
    print(f"After filtering unknown labels: {df_labeled.shape}")
    print(f"\nLabel distribution:")
    print(df_labeled["success_label"].value_counts())
    
    # Filter by phase if specified (in case API didn't filter correctly)
    if filter_phase:
        phase_upper = filter_phase.upper()
        if phase_upper.startswith("PHASE"):
            phase_num = phase_upper.replace("PHASE", "").strip()
            # Filter rows where phase contains the target phase number
            df_labeled = df_labeled[
                df_labeled["phase"].str.contains(phase_num, case=False, na=False)
            ].copy()
            print(f"After filtering by {filter_phase}: {df_labeled.shape[0]} trials")
    
    # Prepare features
    df_processed = prepare_features(df_labeled)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    
    return df_processed


if __name__ == "__main__":
    # Test preprocessing
    if os.path.exists("data/raw/trials_raw.json"):
        df = load_and_preprocess_data()
        print(f"\nProcessed DataFrame shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
    else:
        print("Please run data_fetch.py first to fetch raw data")


"""
Data fetching module for ClinicalTrials.gov API v2
Fetches Alzheimer Disease Phase 3 trials data
"""

import requests
import pandas as pd
import time
from typing import List, Dict, Any
import os


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_alzheimer_trials(
    max_pages: int = 10, 
    page_size: int = 100,
    phase_filter: str = "PHASE3"
) -> List[Dict[str, Any]]:
    """
    Fetch Alzheimer Disease Phase 3 trials from ClinicalTrials.gov API v2
    
    Args:
        max_pages: Maximum number of pages to fetch
        page_size: Number of studies per page
        phase_filter: Phase filter (PHASE3, PHASE2, etc.)
    
    Returns:
        List of study dictionaries
    """
    all_studies: List[Dict[str, Any]] = []
    next_page_token = None

    # Field list - using only validated fields from API
    # Removed invalid fields like armsInterventionsModule.arms
    fields = ",".join([
        # Core identification (top-level convenience fields)
        "NCTId",
        "BriefTitle",
        "OverallStatus",
        "HasResults",
        
        # Protocol section - Identification
        "protocolSection.identificationModule.nctId",
        "protocolSection.identificationModule.briefTitle",
        "protocolSection.identificationModule.acronym",
        
        # Description
        "protocolSection.descriptionModule.briefSummary",
        "protocolSection.descriptionModule.detailedDescription",
        
        # Conditions
        "protocolSection.conditionsModule.conditions",
        
        # Design module - critical features
        "protocolSection.designModule.phases",
        "protocolSection.designModule.studyType",
        "protocolSection.designModule.enrollmentInfo.count",
        "protocolSection.designModule.designInfo.allocation",
        "protocolSection.designModule.designInfo.maskingInfo",
        "protocolSection.designModule.designInfo.interventionModel",
        
        # Status module
        "protocolSection.statusModule.overallStatus",
        "protocolSection.statusModule.startDateStruct.date",
        "protocolSection.statusModule.primaryCompletionDateStruct.date",
        "protocolSection.statusModule.completionDateStruct.date",
        "protocolSection.statusModule.enrollmentCount",
        
        # Eligibility
        "protocolSection.eligibilityModule.minimumAge",
        "protocolSection.eligibilityModule.maximumAge",
        "protocolSection.eligibilityModule.sex",
        "protocolSection.eligibilityModule.eligibilityCriteria",
        
        # Interventions (arms is not a valid field, removed)
        "protocolSection.armsInterventionsModule.interventions",
        
        # Outcomes
        "protocolSection.outcomesModule.primaryOutcomes",
        "protocolSection.outcomesModule.secondaryOutcomes",
        
        # Sponsor
        "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
        "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
        
        # Locations
        "protocolSection.contactsLocationsModule.locations",
    ])

    print(f"Fetching Alzheimer Disease {phase_filter} trials...")
    
    # Build query string - API v2 uses query.term for combined queries
    # Format: condition AND phase (use lowercase "phase 3" format)
    # Convert PHASE3 -> phase 3, PHASE2 -> phase 2, etc.
    if phase_filter.upper().startswith("PHASE"):
        phase_num = phase_filter.upper().replace("PHASE", "").strip()
        phase_query = f"phase {phase_num}" if phase_num else "phase"
    else:
        phase_query = phase_filter.lower()
    
    query_term = f"Alzheimer Disease AND {phase_query}"
    print(f"Query: {query_term}")
    
    for page in range(max_pages):
        params = {
            "format": "json",
            "pageSize": page_size,
            "query.term": query_term,
            "fields": fields,
        }
        
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            resp = requests.get(BASE_URL, params=params, timeout=60)
            
            # Check for errors and print response if needed
            if resp.status_code != 200:
                error_text = resp.text[:1000]
                print(f"API Error {resp.status_code}: {error_text}")
                
                # If it's a field error, try with minimal fields
                if "invalid field name" in error_text.lower() and page == 0:
                    print("\nTrying with minimal field set...")
                    minimal_fields = ",".join([
                        "NCTId",
                        "BriefTitle",
                        "OverallStatus",
                        "HasResults",
                        "protocolSection.identificationModule.nctId",
                        "protocolSection.identificationModule.briefTitle",
                        "protocolSection.descriptionModule.briefSummary",
                        "protocolSection.designModule.phases",
                        "protocolSection.designModule.studyType",
                        "protocolSection.designModule.enrollmentInfo.count",
                        "protocolSection.statusModule.overallStatus",
                        "protocolSection.statusModule.startDateStruct.date",
                        "protocolSection.statusModule.primaryCompletionDateStruct.date",
                    ])
                    params["fields"] = minimal_fields
                    resp = requests.get(BASE_URL, params=params, timeout=60)
                    if resp.status_code == 200:
                        print("Minimal fields query succeeded!")
                        fields = minimal_fields  # Update fields for next pages
                    else:
                        print(f"Still failed: {resp.status_code}")
                        break
                # Try without phase filter as fallback
                elif page == 0:
                    print(f"Retrying with simpler query (Alzheimer Disease only)...")
                    params["query.term"] = "Alzheimer Disease"
                    resp = requests.get(BASE_URL, params=params, timeout=60)
                    if resp.status_code == 200:
                        print("Simple query succeeded. Will filter by phase in post-processing.")
                    else:
                        print(f"Still failed: {resp.status_code}")
                        print(f"Response: {resp.text[:500]}")
                        break
                else:
                    break
            
            resp.raise_for_status()
            data = resp.json()

            studies = data.get("studies", [])
            if not studies:
                print(f"No more studies found at page {page + 1}")
                break

            all_studies.extend(studies)
            print(f"Fetched page {page + 1}: {len(studies)} studies (Total: {len(all_studies)})")

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                print("No more pages available")
                break
                
            # Be polite to the API
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page + 1}: {e}")
            if resp.status_code == 400:
                print("Bad request - likely query format issue")
                print(f"Tried query: {query_term}")
            break

    print(f"\nTotal studies fetched: {len(all_studies)}")
    
    if len(all_studies) == 0:
        print("\nWARNING: No studies were fetched!")
        print("Possible reasons:")
        print("1. API query parameters may be incorrect")
        print("2. Network connection issues")
        print("3. API endpoint may have changed")
        print("\nTry running with a simpler query or check API documentation.")
    
    return all_studies


def save_raw_data(studies: List[Dict[str, Any]], output_path: str = "data/raw/trials_raw.json"):
    """
    Save raw API response to JSON file
    
    Args:
        studies: List of study dictionaries
        output_path: Path to save the JSON file
    """
    import json
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(studies, f, indent=2, ensure_ascii=False)
    
    print(f"Raw data saved to {output_path}")


if __name__ == "__main__":
    # Test data fetching
    studies = fetch_alzheimer_trials(max_pages=5, page_size=100)
    save_raw_data(studies, "data/raw/trials_raw.json")
    print(f"\nSample study keys: {list(studies[0].keys()) if studies else 'No studies'}")


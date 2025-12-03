# Predicting Alzheimer's Clinical Trial Success: A Machine Learning Approach

## 📌 Project Overview
This project aims to predict the success of **Alzheimer's Disease (AD)** clinical trials using machine learning models trained on pre-trial design metadata. 

Given the notoriously high failure rate (>99%) in AD drug development, early identification of high-risk trials is critical for resource optimization. Our project focuses on **Phase 1 Completion Prediction**, utilizing data extracted from the ClinicalTrials.gov API to classify trials as either **Completed (Success)** or **Terminated/Withdrawn (Failure)** based solely on information available at study initiation.

## 📂 Repository Structure

```text
clinical-trial-success-predictor
├── .gitignore
├── LICENSE
├── phase1_completion_feature_importance.png
├── phase1_completion_pr_curve.png
├── phase1_completion_prediction.py
├── README.md
└── requirements.txt
```

## 🛠️ Methodology

### 1. Data Pipeline
* **Source:** ClinicalTrials.gov API (v2).
* **Scope:** Alzheimer's Disease, Interventional, Phase 1.
* **Filtering:** We filtered for trials with definitive statuses (`COMPLETED`, `TERMINATED`, `WITHDRAWN`, `SUSPENDED`) to create a labeled dataset. Ongoing trials (Recruiting/Active) were excluded to ensure ground truth.

### 2. Labeling Strategy (Phase 1 Focus)
For Phase 1 trials, the primary goal is safety and operational feasibility rather than market approval.
* **Success (1):** Status = `COMPLETED`
* **Failure (0):** Status = `TERMINATED`, `WITHDRAWN`, or `SUSPENDED`

### 3. Feature Engineering & Leakage Prevention
To ensure the model only uses data available *before* the trial starts (preventing data leakage), we engineered features using **Regex (Regular Expressions)** on protocol text:
* **Planned Duration:** Extracted from the `timeFrame` field (e.g., "52 weeks") rather than using the actual completion date in the database.
* **Phase 1 Specifics:**
    * **"Healthy Volunteers"**: Flagged via NLP keywords in eligibility criteria (critical for Phase 1 safety studies).
    * **"Dose Escalation"**: Identified SAD/MAD designs via study descriptions.
    * **"PK/Safety Endpoints"**: Detected specific keywords (AUC, Cmax, Adverse Events) in outcome measures.
* **Sponsor:** Binary flags for "Big Pharma" backing (e.g., Pfizer, Biogen) to capture resource availability.

### 4. Modeling Strategy
* **Algorithm:** **XGBoost Classifier** (Selected for its ability to handle tabular data and complex feature interactions).
* **Validation:** **Temporal Split** (Train on older studies, Test on recent studies). Split year is calculated as the 80th percentile of trial start years, with fallback to 2018 if needed. We avoided random splitting to mimic real-world forecasting scenarios.
* **Imbalance Handling:** Used `scale_pos_weight` to address the class imbalance (as ~88% of Phase 1 trials complete successfully).

## 📊 Key Results

Our final model (`phase1_completion_prediction.py`) achieved strong predictive signals on the held-out test set:

* **PR-AUC (Precision-Recall Area Under Curve): ~0.89**
    * *Interpretation:* Demonstrates high effectiveness in identifying the minority class (early terminations), significantly outperforming the random baseline (~0.14).
* **ROC-AUC: ~0.71**
    * *Interpretation:* Indicates a decent ability to rank trials by risk level.
* **Risk Identification:** The model successfully captured **57%** of the trials destined for early termination. In a business context, this serves as a valuable "second opinion" for feasibility reviews, potentially saving millions in wasted R&D costs.

> **Note:** We initially attempted to predict Phase 3 success but pivoted to Phase 1 completion prediction. The extreme scarcity of successful Phase 3 AD trials made stable modeling difficult, whereas Phase 1 offered a more balanced and definable target.

## 🚀 How to Run

**Important:** All operations should be performed on the `main` branch.

1.  **Ensure you're on the main branch:**
    ```bash
    git checkout main
    git pull origin main
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Prediction Script:**
    ```bash
    python phase1_completion_prediction.py
    ```
    * *This script will fetch live data, train the model, print metrics to the console, and save visualization plots locally.*

---

## 🤖 AI Citation and Acknowledgement

For this project, AI assistance was utilized strictly as a productivity tool for code implementation, debugging, documentation, and initial ideation. The core scientific logic, feature strategy, and final decision-making remained human-driven.

**Timeline of AI Usage:**

1.  **Ideation & Feasibility (November 11, 2025)**
    * **Model:** Gemini 2 Pro
    * **Role:** Assisted with initial topic brainstorming and evaluated the feasibility of data availability for different disease scopes (AD vs. Oncology).

2.  **Logic Discussion & Prototyping (November 22, 2025)**
    * **Models:** ChatGPT 5.1, Claude Sonnet 4.5
    * **Role:** Discussed the logic for the "White-List" labeling approach versus using API status fields. Provided initial syntax examples for regex patterns to extract text-based features from unstructured clinical notes.

3.  **Implementation & Engineering (December 1, 2025)**
    * **Model:** Gemini 3 Pro
    * **Role:**
        * Generated boilerplate code for the ClinicalTrials.gov API fetcher (pagination and retry logic).
        * Refactored the codebase for modularity and standardized comment styles.
        * Debugged syntax errors within the `sklearn` Pipeline (specifically the `ColumnTransformer` and Imputer types).
        * Resolved data type inconsistencies (string vs. integer) during the imputation process.

4.  **Documentation & Reporting (December 2, 2025)**
    * **Model:** Gemini 3 Pro
    * **Role:** Assisted in structuring this `README.md` file and drafting the project write-up outline based on the final experimental results. The human team verified all metrics, refined the business impact analysis, and finalized the narrative strategy.

**Statement of Originality:**
The critical intellectual contributions—including the project pivot from Phase 3 to Phase 1, the specific selection of domain-relevant features (e.g., Healthy Volunteers, Dose Escalation), the decision to use XGBoost, and the final interpretation of the business impact—were developed manually by the team. AI served as a coding assistant and editor for execution rather than a replacement for analytical reasoning.
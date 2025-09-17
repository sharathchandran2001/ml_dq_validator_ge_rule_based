import os
import re
import io
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st

# ML specific imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="ML-based Data Quality Classifier", layout="wide")
st.title("🤖 ML-based Data Quality Classifier for Bank Customer Data")
st.markdown(
    "Upload a CSV/Excel file. The app will (1) **generate rule-based data quality labels**, "
    "(2) **train an ML model** on those labels, and (3) **classify** records. "
    "You’ll see **both** rule-based and ML-based outcomes with reasons."
)

# ------------------ PATH SETUP ------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "input")
output_dir = os.path.join(base_dir, "output")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

MODEL_FILE = os.path.join(output_dir, "dq_classifier_model.joblib")
LABEL_ENCODER_FILE = os.path.join(output_dir, "dq_label_encoder.joblib")
FEATURE_COLS_FILE = os.path.join(output_dir, "dq_feature_columns.json")
RULES_FILE_PATH = os.path.join(input_dir, "data_quality_rules.txt")

# ------------------ UTILS ------------------
def read_text(file_path: str) -> str:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

@dataclass
class DQRule:
    id: str
    content: str
    dimension: str  # e.g., "Accuracy", "Completeness"

def parse_rules_file(text_content: str) -> List[DQRule]:
    rules = []
    current_dimension = "Unknown"
    current_rule_id = None
    current_rule_content_lines = []

    lines = text_content.splitlines()
    for line in lines:
        if line.strip().startswith("## "):
            if current_rule_id:
                rules.append(
                    DQRule(
                        id=current_rule_id,
                        content="\n".join(current_rule_content_lines).strip(),
                        dimension=current_dimension,
                    )
                )
                current_rule_id = None
                current_rule_content_lines = []
            current_dimension = line.strip().lstrip("## ").strip()
        elif line.strip().startswith("- **Rule:"):
            if current_rule_id:
                rules.append(
                    DQRule(
                        id=current_rule_id,
                        content="\n".join(current_rule_content_lines).strip(),
                        dimension=current_dimension,
                    )
                )

            match = re.match(r"- \*\*Rule: ([A-Za-z0-9\-_.]+)\*\*:(.*)", line.strip())
            if match:
                current_rule_id = match.group(1).strip()
                current_rule_content_lines = [match.group(2).strip()]
            else:
                current_rule_id = f"UNKNOWN-RULE-{len(rules)}"
                current_rule_content_lines = [line.strip()]
        elif current_rule_id:
            current_rule_content_lines.append(line)

    if current_rule_id:
        rules.append(
            DQRule(
                id=current_rule_id,
                content="\n".join(current_rule_content_lines).strip(),
                dimension=current_dimension,
            )
        )

    return rules

# ------------------ SESSION STATE ------------------
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None
if "ml_model" not in st.session_state:
    st.session_state.ml_model = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = []
if "rules_text" not in st.session_state:
    st.session_state.rules_text = ""
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "rule_id_to_description_map" not in st.session_state:
    st.session_state.rule_id_to_description_map = {}

# Load rules
if not os.path.exists(RULES_FILE_PATH):
    st.session_state.rules_text = ""
    st.error(f"`{RULES_FILE_PATH}` not found in the `input` directory. Please create it with your data quality rules.")
else:
    st.session_state.rules_text = read_text(RULES_FILE_PATH)
    if not st.session_state.rules_text.strip():
        st.warning(f"`{RULES_FILE_PATH}` is empty. Please add your data quality rules to it.")
    else:
        parsed_rules = parse_rules_file(st.session_state.rules_text)
        st.session_state.rule_id_to_description_map = {rule.id: rule.content for rule in parsed_rules}

# Load a previously trained model
if not st.session_state.model_trained:
    if os.path.exists(MODEL_FILE) and os.path.exists(LABEL_ENCODER_FILE) and os.path.exists(FEATURE_COLS_FILE):
        try:
            st.session_state.ml_model = joblib.load(MODEL_FILE)
            st.session_state.label_encoder = joblib.load(LABEL_ENCODER_FILE)
            with open(FEATURE_COLS_FILE, 'r') as f:
                st.session_state.feature_columns = json.load(f)
            st.session_state.model_trained = True
            st.sidebar.success("Previously trained model loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"Error loading saved model: {e}. You may need to retrain.")
            st.session_state.ml_model = None
            st.session_state.label_encoder = None
            st.session_state.feature_columns = []
            st.session_state.model_trained = False

# ------------------ RULE ENGINE ------------------
def generate_features_and_labels(df: pd.DataFrame, rules_text: str) -> Tuple[pd.DataFrame, List[str]]:
    df_processed = df.copy()
    dq_rules = parse_rules_file(rules_text)

    df_processed['DQ_FAIL_COUNT'] = 0
    df_processed['DQ_VIOLATED_RULES_SUMMARY'] = ''

    feature_columns = []

    # Precalc helpers
    df_processed['Age'] = np.nan
    if 'DateOfBirth' in df_processed.columns:
        df_processed['DateOfBirth_dt'] = pd.to_datetime(df_processed['DateOfBirth'], errors='coerce')
        valid_dob_mask = df_processed['DateOfBirth_dt'].notna()
        df_processed.loc[valid_dob_mask, 'Age'] = (
            (datetime.now() - df_processed.loc[valid_dob_mask, 'DateOfBirth_dt']).dt.days / 365.25
        )

    df_processed['YearsSinceAccountOpen'] = np.nan
    if 'AccountOpenDate' in df_processed.columns:
        df_processed['AccountOpenDate_dt'] = pd.to_datetime(df_processed['AccountOpenDate'], errors='coerce')
        valid_aod_mask = df_processed['AccountOpenDate_dt'].notna()
        df_processed.loc[valid_aod_mask, 'YearsSinceAccountOpen'] = (
            (datetime.now() - df_processed.loc[valid_aod_mask, 'AccountOpenDate_dt']).dt.days / 365.25
        )

    if 'CreditScore' in df_processed.columns:
        df_processed['CreditScore_num'] = pd.to_numeric(df_processed['CreditScore'], errors='coerce')
    if 'LoanAmount' in df_processed.columns:
        df_processed['LoanAmount_num'] = pd.to_numeric(df_processed['LoanAmount'], errors='coerce')
    if 'AccountBalance' in df_processed.columns:
        df_processed['AccountBalance_num'] = pd.to_numeric(df_processed['AccountBalance'], errors='coerce')

    customer_id_duplicates_series = pd.Series(False, index=df_processed.index)
    if 'CustomerID' in df_processed.columns and not df_processed['CustomerID'].isnull().all():
        customer_id_duplicates_series = (
            df_processed['CustomerID'].duplicated(keep=False) & df_processed['CustomerID'].notna()
        )

    full_row_duplicates_series = df_processed.duplicated(keep=False)

    # Apply rules
    for rule in dq_rules:
        violation_flag = pd.Series(0, index=df_processed.index, dtype=int)

        # Completeness
        if 'COMP-CRITICAL-001' in rule.id:
            cols_to_check = ['CustomerID', 'FirstName', 'LastName', 'Email', 'DateOfBirth', 'AccountOpenDate']
            for col in cols_to_check:
                if col in df_processed.columns:
                    violation_flag = violation_flag | df_processed[col].isnull().astype(int)

        elif 'COMP-PHONE-001' in rule.id and 'PhoneNumber' in df_processed.columns:
            violation_flag = violation_flag | df_processed['PhoneNumber'].isnull().astype(int)

        # Uniqueness
        elif 'UNIQUE-CUSTID-001' in rule.id:
            violation_flag = violation_flag | customer_id_duplicates_series.astype(int)
        elif 'UNIQUE-ROW-001' in rule.id:
            violation_flag = violation_flag | full_row_duplicates_series.astype(int)

        # Validity
        elif 'VAL-EMAIL-001' in rule.id and 'Email' in df_processed.columns:
            invalid_emails_mask = ~df_processed['Email'].astype(str).str.match(
                r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", na=True
            )
            violation_flag = violation_flag | invalid_emails_mask.astype(int)

        elif 'VAL-PHONE-001' in rule.id and 'PhoneNumber' in df_processed.columns:
            cleaned_phones = df_processed['PhoneNumber'].astype(str).str.replace(r'\D', '', regex=True)
            invalid_phones_mask = ~cleaned_phones.str.match(r"^\d{7,15}$", na=True)
            violation_flag = violation_flag | invalid_phones_mask.astype(int)

        elif 'VAL-DATE-001' in rule.id:
            date_cols = ['DateOfBirth', 'AccountOpenDate', 'LastTransactionDate']
            for col in date_cols:
                if col in df_processed.columns:
                    invalid_date_mask = pd.to_datetime(df_processed[col], errors='coerce').isnull() & df_processed[col].notnull()
                    violation_flag = violation_flag | invalid_date_mask.astype(int)

        elif 'VAL-ACCTTYPE-001' in rule.id and 'AccountType' in df_processed.columns:
            valid_types = ["Checking", "Savings", "Loan", "CreditCard"]
            invalid_acc_type_mask = ~df_processed['AccountType'].astype(str).isin(valid_types)
            violation_flag = violation_flag | invalid_acc_type_mask.astype(int)

        # Accuracy
        elif 'ACC-CS-001' in rule.id and 'CreditScore_num' in df_processed.columns:
            out_of_range_scores_mask = (df_processed['CreditScore_num'] < 300) | (df_processed['CreditScore_num'] > 850)
            violation_flag = violation_flag | out_of_range_scores_mask.astype(int)

        elif 'ACC-AGE-001' in rule.id and 'Age' in df_processed.columns:
            out_of_range_age_mask = (df_processed['Age'] < 18) | (df_processed['Age'] > 100)
            violation_flag = violation_flag | out_of_range_age_mask.astype(int)

        elif 'ACC-LA-001' in rule.id and 'LoanAmount_num' in df_processed.columns:
            negative_loan_mask = (df_processed['LoanAmount_num'] < 0)
            violation_flag = violation_flag | negative_loan_mask.astype(int)

        elif 'ACC-BALANCE-001' in rule.id and 'AccountBalance_num' in df_processed.columns:
            unrealistic_balance_mask = (df_processed['AccountBalance_num'] < -10000) | (df_processed['AccountBalance_num'] > 10000000)
            violation_flag = violation_flag | unrealistic_balance_mask.astype(int)

        # Timeliness
        elif 'TIME-TRANS-001' in rule.id and 'LastTransactionDate' in df_processed.columns:
            df_processed['LastTransactionDate_dt'] = pd.to_datetime(df_processed['LastTransactionDate'], errors='coerce')
            one_year_ago = datetime.now() - pd.DateOffset(years=1)
            outdated_mask = (df_processed['LastTransactionDate_dt'] < one_year_ago)
            violation_flag = violation_flag | outdated_mask.astype(int)

        elif 'TIME-DOB-001' in rule.id and 'DateOfBirth_dt' in df_processed.columns:
            future_dob_mask = (df_processed['DateOfBirth_dt'] > datetime.now())
            violation_flag = violation_flag | future_dob_mask.astype(int)

        # Consistency
        elif 'CONS-LS-CS-001' in rule.id and 'LoanStatus' in df_processed.columns and 'CreditScore_num' in df_processed.columns:
            inconsistent_loan_mask = (df_processed['LoanStatus'] == 'Approved') & (df_processed['CreditScore_num'] < 600)
            violation_flag = violation_flag | inconsistent_loan_mask.astype(int)

        elif 'CONS-DATE-001' in rule.id and 'AccountOpenDate_dt' in df_processed.columns and 'LastTransactionDate_dt' in df_processed.columns:
            inconsistent_date_order_mask = (df_processed['AccountOpenDate_dt'] > df_processed['LastTransactionDate_dt'])
            violation_flag = violation_flag | inconsistent_date_order_mask.astype(int)

        # Store feature
        feature_name = f"rule_violation_{rule.id.replace('-', '_').lower()}"
        df_processed[feature_name] = violation_flag
        feature_columns.append(feature_name)

        # Aggregate
        df_processed['DQ_FAIL_COUNT'] += violation_flag
        df_processed.loc[violation_flag == 1, 'DQ_VIOLATED_RULES_SUMMARY'] = (
            df_processed.loc[violation_flag == 1, 'DQ_VIOLATED_RULES_SUMMARY'].apply(
                lambda x: f"{x}|{rule.id}" if pd.notna(x) and x != '' else rule.id
            )
        )

    # Final rule-based label
    df_processed['DQ_RULE_STATUS'] = df_processed['DQ_FAIL_COUNT'].apply(lambda x: 'Fail' if x > 0 else 'Pass')

    # Cleanup temps
    for col in ['DateOfBirth_dt', 'AccountOpenDate_dt', 'CreditScore_num', 'LoanAmount_num', 'AccountBalance_num', 'LastTransactionDate_dt']:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])

    df_processed['DQ_VIOLATED_RULES_SUMMARY'] = df_processed['DQ_VIOLATED_RULES_SUMMARY'].str.lstrip('|')

    return df_processed, feature_columns

# ------------------ UI: RULES PREVIEW ------------------
st.info("Ensure your `data_quality_rules.txt` file is in the `input` directory.")
if st.session_state.rules_text.strip():
    with st.expander("🧾 View Loaded Data Quality Rules", expanded=False):
        st.text_area("Rules defined in data_quality_rules.txt:", value=st.session_state.rules_text, height=300, disabled=True)
else:
    st.warning("No data quality rules loaded. The ML model cannot be trained without rules.")

st.markdown("---")

# ------------------ FILE UPLOAD ------------------
st.subheader("📥 Upload Customer Data File (for Training & Prediction)")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], key="main_data_uploader")

# ------------------ HELPER: RULE REASONS ------------------
def build_rule_reason(rule_ids_str: str) -> str:
    if pd.isna(rule_ids_str) or rule_ids_str.strip() == "":
        return "No rules violated"
    rule_ids = [r for r in rule_ids_str.split("|") if r]
    descs = []
    for rid in rule_ids:
        desc = st.session_state.rule_id_to_description_map.get(rid, "Description Not Found")
        descs.append(f"**{rid}**: {desc}")
    return "\n".join(descs)

# ------------------ HELPER: ML REASONS (lightweight, no SHAP) ------------------
def build_ml_reason(
    row_idx: int,
    x_row: pd.Series,
    feature_cols: List[str],
    model: RandomForestClassifier,
    class_labels: np.ndarray,
    proba_row: np.ndarray,
    top_k: int = 3
) -> str:
    # Basic probability statement
    parts = []
    try:
        pass_idx = np.where(class_labels == 'Pass')[0][0] if 'Pass' in class_labels else None
        fail_idx = np.where(class_labels == 'Fail')[0][0] if 'Fail' in class_labels else None
    except Exception:
        pass_idx = fail_idx = None

    if pass_idx is not None and fail_idx is not None:
        parts.append(f"Predicted **{ 'Pass' if proba_row[pass_idx] >= proba_row[fail_idx] else 'Fail' }** "
                     f"(Pass={proba_row[pass_idx]:.2f}, Fail={proba_row[fail_idx]:.2f}).")
    else:
        parts.append("Predicted label; per-class probabilities unavailable.")

    # Use feature importances to rank *active* rule violations (value==1)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        active_feats = [(feature_cols[i], importances[i]) for i in range(len(feature_cols)) if x_row.iloc[i] == 1]
        inactive_feats = [(feature_cols[i], importances[i]) for i in range(len(feature_cols)) if x_row.iloc[i] == 0]

        active_feats_sorted = sorted(active_feats, key=lambda t: t[1], reverse=True)[:top_k]
        inactive_feats_sorted = sorted(inactive_feats, key=lambda t: t[1], reverse=True)[:top_k]

        # Map back to rule IDs for readability
        def pretty(name: str) -> str:
            rid = name.replace("rule_violation_", "").upper()
            rid = rid.replace("_", "-")
            return rid

        if active_feats_sorted:
            parts.append(
                "**Top active rule-violation features:** "
                + ", ".join([f"{pretty(n)} (imp={w:.3f})" for n, w in active_feats_sorted])
            )
        else:
            parts.append("No active rule-violation features; key violations are absent.")

        if inactive_feats_sorted:
            parts.append(
                "_Absence_ of influential violations: "
                + ", ".join([f"{pretty(n)} (imp={w:.3f})" for n, w in inactive_feats_sorted])
            )
    else:
        parts.append("Model explanation unavailable (no feature_importances_).")

    return " ".join(parts)

# ------------------ MAIN FLOW ------------------
if uploaded_file is not None:
    try:
        # Load once per distinct file
        if st.session_state.uploaded_df is None or uploaded_file.name != st.session_state.last_uploaded_filename:
            with st.spinner(f"Loading '{uploaded_file.name}'..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.uploaded_df = df
                st.session_state.last_uploaded_filename = uploaded_file.name
                st.success(f"File '{uploaded_file.name}' loaded successfully! Displaying first 5 rows.")
                st.dataframe(df.head())
        else:
            st.info(f"File '{st.session_state.last_uploaded_filename}' is currently loaded. Displaying first 5 rows.")
            st.dataframe(st.session_state.uploaded_df.head())

        # ------------------ TRAIN ------------------
        st.markdown("### ⚙️ Generate Labels (Rules) & Train ML Model")
        if st.session_state.uploaded_df is not None and not st.session_state.uploaded_df.empty and st.session_state.rules_text.strip():
            if st.button("Generate Labels & Train ML Model", use_container_width=True, type="primary"):
                with st.spinner("Generating features and labels, then training ML model..."):
                    df_labeled, feature_cols = generate_features_and_labels(
                        st.session_state.uploaded_df, st.session_state.rules_text
                    )
                    # Save feature columns
                    st.session_state.feature_columns = feature_cols
                    with open(FEATURE_COLS_FILE, 'w') as f:
                        json.dump(feature_cols, f)

                    X_raw = df_labeled[feature_cols]
                    y_rule_labels = df_labeled['DQ_RULE_STATUS']  # <- rule-based label

                    if X_raw.empty or y_rule_labels.empty:
                        st.warning("No features or labels could be generated from the data based on the rules. Check data format or rules.")
                        st.stop()

                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y_rule_labels)
                    st.session_state.label_encoder = le
                    joblib.dump(le, LABEL_ENCODER_FILE)

                    # Stratified split if possible
                    class_counts = y_rule_labels.value_counts()
                    if class_counts.min() < 2:
                        st.warning(
                            f"Not enough samples in one or both classes for stratified split "
                            f"(e.g., only {class_counts.min()} of '{class_counts.idxmin()}' class). "
                            "Training without stratification."
                        )
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_raw, y_encoded, test_size=0.2, random_state=42
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                        )

                    model = RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight='balanced'
                    )
                    model.fit(X_train, y_train)
                    st.session_state.ml_model = model
                    joblib.dump(model, MODEL_FILE)

                    st.session_state.model_trained = True
                    st.success("ML Model Trained Successfully! Model saved to disk.")

                    # --- Model Performance ---
                    st.markdown("### Model Performance on Test Data (vs Rule Labels)")
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    cm = confusion_matrix(y_test, y_pred)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.2f}")
                    col2.metric("Precision", f"{precision:.2f}")
                    col3.metric("Recall", f"{recall:.2f}")
                    col4.metric("F1-Score", f"{f1:.2f}")

                    st.markdown("#### Confusion Matrix")
                    st.dataframe(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

                    st.markdown("#### Classification Report")
                    st.text(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        else:
            if st.session_state.model_trained:
                st.info("Model already trained in this session or loaded from disk. You can now classify data.")
            else:
                st.info("Click the button above to generate labels and train the ML model.")

    except Exception as e:
        st.error(f"Error processing file or training model: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")

else:
    st.info("Upload a CSV or Excel file to get started.")

st.markdown("---")
st.subheader("Classify Records with Trained Model")

# ------------------ CLASSIFY (shows Rule vs ML + reasons) ------------------
if st.session_state.model_trained and st.session_state.uploaded_df is not None:
    if st.button("Classify Current Data with ML Model", use_container_width=True, type="secondary"):
        with st.spinner("Classifying data..."):
            # Re-run rule engine to get features + DQ_RULE_STATUS + rule summary
            df_processed, feature_cols_again = generate_features_and_labels(
                st.session_state.uploaded_df, st.session_state.rules_text
            )
            # Use the saved training feature order to ensure alignment
            final_X_cols = st.session_state.feature_columns
            X_predict_raw = df_processed[final_X_cols].copy()
            dq_rule_status = df_processed['DQ_RULE_STATUS']
            dq_rule_summary = df_processed['DQ_VIOLATED_RULES_SUMMARY']

            # Build RULE_REASON column
            rule_reason_col = dq_rule_summary.apply(build_rule_reason)

            # Predict with ML
            model = st.session_state.ml_model
            le = st.session_state.label_encoder
            class_labels = le.classes_

            predictions_encoded = model.predict(X_predict_raw)
            predictions_labels = le.inverse_transform(predictions_encoded)
            probabilities = model.predict_proba(X_predict_raw)

            # Build ML_REASON column row-by-row (lightweight explanation)
            ml_reasons = []
            for i in range(len(X_predict_raw)):
                ml_reasons.append(
                    build_ml_reason(
                        i,
                        X_predict_raw.iloc[i],
                        final_X_cols,
                        model,
                        class_labels,
                        probabilities[i],
                        top_k=3
                    )
                )

            # Assemble results
            df_results = st.session_state.uploaded_df.copy()
            df_results['DQ_RULE_STATUS'] = dq_rule_status.values
            df_results['RULE_REASON'] = rule_reason_col.values

            df_results['DQ_ML_STATUS'] = predictions_labels
            # probs
            pass_idx = np.where(class_labels == 'Pass')[0][0] if 'Pass' in class_labels else None
            fail_idx = np.where(class_labels == 'Fail')[0][0] if 'Fail' in class_labels else None
            if pass_idx is not None:
                df_results['DQ_ML_PROB_PASS'] = probabilities[:, pass_idx]
            else:
                df_results['DQ_ML_PROB_PASS'] = np.nan
            if fail_idx is not None:
                df_results['DQ_ML_PROB_FAIL'] = probabilities[:, fail_idx]
            else:
                df_results['DQ_ML_PROB_FAIL'] = np.nan

            df_results['ML_REASON'] = ml_reasons

            # Optional: final status where rules override ML (toggle this if you prefer strict rules)
            df_results['DQ_FINAL_STATUS'] = np.where(
                dq_rule_summary.fillna("").str.strip() != "",
                "Fail",
                df_results['DQ_ML_STATUS']
            )

            st.markdown("### Classification Results (First 100 rows)")
            st.dataframe(df_results.head(100))

            csv_output = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Classified Data (CSV)",
                csv_output,
                file_name=f"classified_dq_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
else:
    if not st.session_state.model_trained:
        st.warning("No ML model is trained or loaded. Please train a model first using the button above.")
    elif st.session_state.uploaded_df is None:
        st.info("Upload data to classify it with the loaded model.")

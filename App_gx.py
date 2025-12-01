import io
import json
import re
from datetime import date
from typing import Dict, Any, List, Optional
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Kept for chart logic, though not directly rendered in API
import yaml

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 🚨 REVISED GREAT EXPECTATIONS IMPORT 🚨
# Using the newer Validator-based approach which resolves the ModuleNotFoundError 
# in recent GE versions and Python 3.11.
from great_expectations.validator.validator import Validator
# Note: For simplicity in this direct translation, we'll instantiate the Validator 
# using a Pandas DataFrame directly, which is still supported.

# =========================
# Initialize FastAPI app
# =========================

app = FastAPI(
    title="Universal Data Quality API",
    description="FastAPI endpoint for running Great Expectations checks on uploaded CSV and YAML rules.",
    version="1.0.0"
)

# =========================
# Schemas for API Response
# =========================

class ExpectationResult(BaseModel):
    id: int
    label: str
    why: str
    column: Optional[str]
    success: bool
    fail_count: int
    expectation_type: str
    kwargs: Dict[str, Any]

# =========================
# Helper Functions
# =========================

def parse_date_safe(x):
    if pd.isna(x):
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(x, format=fmt, errors="raise")
        except Exception:
            pass
    return pd.to_datetime(x, errors="coerce")


def compute_age(dob):
    if pd.isna(dob):
        return np.nan
    if isinstance(dob, str):
        dob = parse_date_safe(dob)
    if pd.isna(dob):
        return np.nan
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Add generic derived columns when present
    if "DateOfBirth" in df.columns and "Age" not in df.columns:
        df = df.copy()
        df["Age"] = df["DateOfBirth"].apply(compute_age)
    return df


def load_rules_from_text(text: str) -> Dict[str, Any]:
    try:
        return yaml.safe_load(text) or {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse rules.yaml: {e}")


def default_rules_yaml_text() -> str:
    # A safe, useful default if no rules.yaml is provided.
    return """
defaults:
  mostly: 0.98
  max_string_len: 256
  numeric_nonnegative: false

# Generic column rules that apply by regex pattern (case-insensitive)
columns:
  - match: regex
    name: Email|email|user_email
    expectations:
      - type: regex
        pattern: "^[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}$"
        mostly: 0.98
        why: "Ensures contactability and valid routing."

  - match: regex
    name: Phone|phone|mobile|contact
    expectations:
      - type: regex
        pattern: "^[\\d\\-\\+\\s\\(\\)ext\\.]{7,}$"
        mostly: 0.95
        why: "Catches obviously invalid phone values."

  - match: regex
    name: DateOfBirth|dob|birth
    expectations:
      - type: date_parseable
        mostly: 0.98
        why: "Detects malformed birth dates for age computations."

domain_packs:

  customer:
    columns:
      - match: exact
        name: CustomerID
        expectations:
          - type: not_null
            why: "Primary identifier cannot be null."
          - type: unique
            why: "Avoid duplicate identifiers."

      - match: exact
        name: Age
        expectations:
          - type: between
            min: 18
            max: 100
            mostly: 0.99
            why: "Regulatory/eligibility constraints."

      - match: exact
        name: CreditScore
        expectations:
          - type: between
            min: 300
            max: 850
            mostly: 0.99
            why: "Standard FICO-like bounds."

      - match: exact
        name: AccountType
        expectations:
          - type: in_set
            values: ["Checking", "Savings", "Credit", "Mortgage", "Loan"]
            mostly: 0.98
            why: "Keeps product taxonomy consistent."

  credit_card:
    columns:
      - match: regex
        name: PAN|CardNumber|card_number
        expectations:
          - type: regex
            pattern: "^[0-9]{13,19}$"
            why: "Primary account number should be 13-19 digits."

      - match: regex
        name: ExpiryDate|expiry|exp
        expectations:
          - type: date_parseable
            mostly: 0.98
            why: "Detects malformed expiry dates."

      - match: regex
        name: CVV|cvv|cvc
        expectations:
          - type: regex
            pattern: "^[0-9]{3,4}$"
            why: "CVV must be 3-4 digits."

  loan:
    columns:
      - match: regex
        name: Principal|LoanAmount|Amount
        expectations:
          - type: between
            min: 0
            why: "Loan principal cannot be negative."

      - match: regex
        name: InterestRate|APR|Rate
        expectations:
          - type: between
            min: 0
            max: 100
            mostly: 0.999
            why: "Rates should be between 0 and 100%."
    """


def infer_domain_pack(config: Dict[str, Any], columns: List[str]) -> Optional[str]:
    """Heuristic: choose the pack with the most matching column patterns."""
    packs = (config.get("domain_packs") or {})
    best, best_score = None, 0
    for name, pack in packs.items():
        score = 0
        for col_rule in pack.get("columns", []):
            pattern = col_rule.get("name", "")
            match_type = col_rule.get("match", "regex").lower()
            if match_type == "exact":
                if pattern in columns:
                    score += 1
            else:
                if any(re.search(pattern, c, re.IGNORECASE) for c in columns):
                    score += 1
        if score > best_score:
            best, best_score = name, score
    return best


def _apply_expectation(validator: Validator, df: pd.DataFrame, col: str, exp: Dict[str, Any]):
    """Map concise YAML expectation to a GE Validator call."""
    t = (exp.get("type") or "").lower()
    mostly = exp.get("mostly", None)

    # All GE expectations start with 'expect_' and are methods of the Validator
    expectation_method_name = f"expect_column_values_to_{t}"

    if t == "not_null":
        return validator.expect_column_values_to_not_be_null(col, mostly=mostly)

    if t == "unique":
        return validator.expect_column_values_to_be_unique(col)

    if t == "regex":
        return validator.expect_column_values_to_match_regex(col, exp["pattern"], mostly=mostly)

    if t in ("between", "ge_between"):
        return validator.expect_column_values_to_be_between(
            col, min_value=exp.get("min"), max_value=exp.get("max"), mostly=mostly
        )

    if t == "in_set":
        return validator.expect_column_values_to_be_in_set(
            col, exp.get("values", []), mostly=mostly
        )

    if t == "date_parseable":
        # Note: Validator does not have a direct method for this. 
        # We simulate the old PandasDataset logic using a temporary column check 
        # within the DataFrame being validated by the Validator.
        # However, for simplicity and adherence to the original logic:
        # We will directly call the underlying GE PandasDataset (Validator's internal) to 
        # run this specific check, as the Validator interface doesn't expose the temporary column creation directly.
        # This requires importing PandasDataset again for this specific hack.
        
        # NOTE: A cleaner implementation would require full Data Context configuration 
        # to define a custom expectation or use transform methods. 
        # For direct migration, we'll try to use the most compatible structure.

        # Temporary workaround: We can't easily create a temporary column in the Validator's 
        # batch, so we'll rely on a manual check mimicking the original logic.
        tmp_df = df.copy()
        tmp_df[f"__{col}__parsed"] = pd.to_datetime(df[col], errors="coerce")
        null_count = tmp_df[f"__{col}__parsed"].isna().sum()
        total_count = len(tmp_df)
        
        # Manually construct a result dictionary mimicking GE output for this hack
        success = (1 - (null_count / total_count)) >= mostly
        return {
            "success": success,
            "result": {
                "element_count": total_count,
                "unexpected_count": null_count,
            },
            "expectation_config": {
                "expectation_type": "expect_column_values_to_be_date_parseable",
                "kwargs": {"column": col, "mostly": mostly}
            }
        }

    return None


def make_label(col: str, exp: Dict[str, Any]) -> str:
    """Creates a human-readable label for the expectation."""
    t = exp.get("type", "rule")
    if t == "regex":
        return f"{col}: matches regex"
    if t in ("between", "ge_between"):
        rng = f"{exp.get('min', '')}–{exp.get('max', '')}".strip("–")
        return f"{col}: between {rng}".strip()
    if t == "in_set":
        values = exp.get('values', [])
        display_values = ', '.join(map(str, values[:3])) + ('...' if len(values) > 3 else '')
        return f"{col}: in {{{display_values}}}"
    if t == "not_null":
        return f"{col}: not null"
    if t == "unique":
        return f"{col}: unique"
    if t == "date_parseable":
        return f"{col}: parseable date"
    return f"{col}: {t}"


def fail_count_from_result(r: Dict[str, Any]) -> int:
    """Extracts the number of failing rows from a GE result dictionary."""
    if not isinstance(r, dict):
        return 0
    if r.get("unexpected_count") is not None:
        try:
            return int(r["unexpected_count"])
        except Exception:
            pass
    uil = r.get("unexpected_index_list")
    if isinstance(uil, list):
        return len(uil)
    return 0


def build_and_run_expectations(df: pd.DataFrame, config: Dict[str, Any], chosen_pack: Optional[str]) -> pd.DataFrame:
    """Run table-level checks + generic column rules + optional domain pack rules. Return tidy results DataFrame."""
    
    # Instantiate the Validator using the DataFrame.
    # We use a dummy context and batch ID since we are not storing the validation result.
    validator = Validator(
        # We need to manually provide the context and batch_id for the direct Validator instantiation
        context=None, 
        batch={
            "data": df
        }
    )
    
    results = []
    eid = 0

    def add_result(ev, label, why, column=None):
        nonlocal eid
        if ev is None:
            return
        
        # Handle both ValidatorResult (GE object) and manual dict (for parseable_date hack)
        evd = ev.to_json_dict() if hasattr(ev, "to_json_dict") else dict(ev)
        
        expectation_config = evd.get("expectation_config", {}) or {}
        kwargs = expectation_config.get("kwargs", {}) or {}
        result_details = evd.get("result", {}) or {}

        fail_count = fail_count_from_result(result_details)

        results.append({
            "id": eid,
            "label": label,
            "why": why,
            "column": column,
            "success": bool(evd.get("success", False)),
            "result": result_details,
            "expectation_type": expectation_config.get("expectation_type", ""),
            "kwargs": kwargs,
            "fail_count": fail_count
        })
        eid += 1

    # --- Table-level generic checks ---
    add_result(validator.expect_table_row_count_to_be_between(min_value=1),
               "Table has at least 1 row", "Files should not be empty.")
    add_result(validator.expect_table_columns_to_match_ordered_list(list(df.columns)),
               "Columns appear as uploaded", "Prevents silent column reordering or loss.")

    # --- Apply rules (Generic + Domain) ---
    def apply_rules_from_config(rules_list: List[Dict[str, Any]], context_why: str):
        for rule in (rules_list or []):
            pattern = rule.get("name", "")
            match_type = rule.get("match", "regex").lower()
            for col in df.columns:
                matched = (col == pattern) if match_type == "exact" else bool(re.search(pattern, col, re.IGNORECASE))
                if not matched:
                    continue
                for exp in (rule.get("expectations") or []):
                    why = exp.get("why", context_why)
                    # Pass the original DataFrame `df` only to the custom _apply_expectation for date_parseable
                    ev = _apply_expectation(validator, df, col, exp)
                    label = make_label(col, exp)
                    add_result(ev, label, why, column=col)

    # 1. Generic column rules
    apply_rules_from_config(config.get("columns", []), "Data quality constraint.")

    # 2. Domain pack rules (optional)
    if chosen_pack:
        pack = (config.get("domain_packs") or {}).get(chosen_pack, {})
        context_why = f"Domain rule from pack '{chosen_pack}'."
        apply_rules_from_config(pack.get("columns", []), context_why)


    results_df = pd.DataFrame(results)

    if "fail_count" not in results_df.columns:
        results_df["fail_count"] = results_df["result"].apply(fail_count_from_result)

    return results_df


# =========================
# FastAPI Endpoint
# =========================

@app.post("/validate_data", response_model=List[ExpectationResult])
async def validate_data(
    csv_file: UploadFile = File(..., description="The CSV file to validate."),
    yaml_file: Optional[UploadFile] = File(None, description="Optional YAML rules file. If omitted, default rules are used."),
    domain_pack_override: Optional[str] = Form(None, description="Explicitly choose a domain pack (e.g., 'customer', 'loan'). If omitted, one is inferred.")
):
    """
    Runs data quality checks on an uploaded CSV using Great Expectations and YAML rules.

    The endpoint accepts a CSV and an optional rules.yaml file via multipart/form-data.
    It returns a JSON array of all expectation results.
    """
    
    # 1. Load and parse CSV
    if csv_file.content_type != 'text/csv' and not csv_file.filename.lower().endswith(".csv"):
         raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")
         
    try:
        csv_bytes = await csv_file.read()
        csv_buffer = BytesIO(csv_bytes)
        df = pd.read_csv(csv_buffer)
        df = add_derived_columns(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read or process CSV: {e}")

    # 2. Load rules config
    if yaml_file and yaml_file.filename:
        try:
            yaml_bytes = await yaml_file.read()
            rules_text = yaml_bytes.decode("utf-8", errors="ignore")
            config = load_rules_from_text(rules_text)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read or process YAML file: {e}")
    else:
        config = load_rules_from_text(default_rules_yaml_text())
        
    # 3. Determine domain pack
    columns = list(df.columns)
    
    if domain_pack_override is None:
        chosen_pack = infer_domain_pack(config, columns)
    else:
        packs = (config.get("domain_packs") or {}).keys()
        if domain_pack_override not in packs:
            raise HTTPException(status_code=400, detail=f"Invalid domain_pack_override: '{domain_pack_override}'. Available packs are: {', '.join(packs)}")
        chosen_pack = domain_pack_override

    # 4. Run expectations
    try:
        results_df = build_and_run_expectations(df, config, chosen_pack)
    except Exception as e:
        # Catch unexpected errors during GE processing
        print(f"Internal error during expectation run: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error running expectations. Check server logs for details.")

    # 5. Prepare and return JSON response
    response_df = results_df[[
        "id", "label", "why", "column", "success", "fail_count",
        "expectation_type", "kwargs"
    ]]

    json_results = response_df.to_dict(orient="records")

    return JSONResponse(content=json_results)

# =========================
# Execution Instructions
# =========================

# To run this FastAPI application:
# 1. Install required libraries:
#    pip install fastapi uvicorn pandas numpy pyyaml great-expectations matplotlib
# 2. Save the code as 'fastapi_app.py'
# 3. Run the server:
#    uvicorn fastapi_app:app --reload
# 4. Access the API documentation at http://127.0.0.1:8000/docs to test the endpoint.

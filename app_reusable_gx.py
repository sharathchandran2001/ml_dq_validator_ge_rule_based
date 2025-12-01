import io
import json
import re
from datetime import date
from typing import Dict, Any, List, Optional
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Kept for the chart logic, though not directly rendered in API
import yaml

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Great Expectations (dataset API)
# Ensure great_expectations is installed: pip install great-expectations
from great_expectations.dataset import PandasDataset


# Initialize FastAPI app
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
    # 'result' field is a complex dict, simplifying the schema for top-level response

# =========================
# Helpers (Modified to remove Streamlit dependencies)
# =========================

def parse_date_safe(x):
    if pd.isna(x):
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            # Use 'raise' in errors to force specific format parsing
            return pd.to_datetime(x, format=fmt, errors="raise")
        except Exception:
            pass
    # Fallback with 'coerce'
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
        # Ensure 'DateOfBirth' is properly parsed before calculating age
        df["Age"] = df["DateOfBirth"].apply(compute_age)
    return df


def load_rules_from_text(text: str) -> Dict[str, Any]:
    try:
        # Use safe_load to prevent arbitrary code execution from YAML
        return yaml.safe_load(text) or {}
    except Exception as e:
        # In a FastAPI context, raise an HTTPException instead of st.error
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
                # regex
                # Use re.IGNORECASE for case-insensitive matching in any()
                if any(re.search(pattern, c, re.IGNORECASE) for c in columns):
                    score += 1
        if score > best_score:
            best, best_score = name, score
    return best


def _apply_expectation(gdf: PandasDataset, df: pd.DataFrame, col: str, exp: Dict[str, Any]):
    """Map concise YAML expectation to a GE PandasDataset call."""
    t = (exp.get("type") or "").lower()
    mostly = exp.get("mostly", None)

    if t == "not_null":
        return gdf.expect_column_values_to_not_be_null(col, mostly=mostly)

    if t == "unique":
        return gdf.expect_column_values_to_be_unique(col)

    if t == "regex":
        return gdf.expect_column_values_to_match_regex(col, exp["pattern"], mostly=mostly)

    if t in ("between", "ge_between"):
        return gdf.expect_column_values_to_be_between(
            col, min_value=exp.get("min"), max_value=exp.get("max"), mostly=mostly
        )

    if t == "in_set":
        return gdf.expect_column_values_to_be_in_set(
            col, exp.get("values", []), mostly=mostly
        )

    if t == "date_parseable":
        # GE doesn't have a direct 'parseable date' expectation.
        # This implementation creates a temporary column with coerced date types.
        tmp = PandasDataset(df.copy())
        # Use errors='coerce' to turn unparseable dates into NaT
        tmp[f"__{col}__parsed"] = pd.to_datetime(df[col], errors="coerce")
        # Then check that the parsed column is not null (i.e., successfully parsed)
        return tmp.expect_column_values_to_not_be_null(f"__{col}__parsed", mostly=mostly)

    # Add other types as needed
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
        # Format set nicely for display
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
    # Preferred: unexpected_count
    if r.get("unexpected_count") is not None:
        try:
            return int(r["unexpected_count"])
        except Exception:
            pass
    # Fallback: unexpected_index_list
    uil = r.get("unexpected_index_list")
    if isinstance(uil, list):
        return len(uil)
    return 0


def build_and_run_expectations(df: pd.DataFrame, config: Dict[str, Any], chosen_pack: Optional[str]) -> pd.DataFrame:
    """Run table-level checks + generic column rules + optional domain pack rules. Return tidy results DataFrame."""
    gdf = PandasDataset(df.copy())
    results = []
    eid = 0

    def add_result(ev, label, why, column=None):
        nonlocal eid
        if ev is None:
            return
        # GE's validation results are sometimes nested; standardize extraction
        evd = ev.to_json_dict() if hasattr(ev, "to_json_dict") else dict(ev)
        
        # Extract metadata
        expectation_config = evd.get("expectation_config", {}) or {}
        kwargs = expectation_config.get("kwargs", {}) or {}
        result_details = evd.get("result", {}) or {}

        # Calculate fail count
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
            "fail_count": fail_count # Added here to be available in the results_df
        })
        eid += 1

    # --- Table-level generic checks ---
    add_result(gdf.expect_table_row_count_to_be_between(min_value=1),
               "Table has at least 1 row", "Files should not be empty.")
    add_result(gdf.expect_table_columns_to_match_ordered_list(list(df.columns)),
               "Columns appear as uploaded", "Prevents silent column reordering or loss.")

    # --- Apply rules (Generic + Domain) ---
    def apply_rules_from_config(rules_list: List[Dict[str, Any]], context_why: str):
        for rule in (rules_list or []):
            pattern = rule.get("name", "")
            match_type = rule.get("match", "regex").lower()
            for col in df.columns:
                # Case-insensitive matching for regex
                matched = (col == pattern) if match_type == "exact" else bool(re.search(pattern, col, re.IGNORECASE))
                if not matched:
                    continue
                for exp in (rule.get("expectations") or []):
                    why = exp.get("why", context_why)
                    ev = _apply_expectation(gdf, df, col, exp)
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

    # The fail_count is already computed in add_result, but we ensure it's present
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
    # Optional form parameter for explicitly setting the domain pack
    domain_pack_override: Optional[str] = Form(None, description="Explicitly choose a domain pack (e.g., 'customer', 'loan'). If omitted, one is inferred.")
):
    """
    Runs data quality checks on an uploaded CSV using Great Expectations and YAML rules.

    The endpoint accepts a CSV and an optional rules.yaml file via multipart/form-data.
    It returns a JSON array of all expectation results.
    """
    
    # 1. Load and parse CSV
    if csv_file.content_type != 'text/csv':
         raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")
         
    try:
        # Read the uploaded file's content
        csv_bytes = await csv_file.read()
        csv_buffer = BytesIO(csv_bytes)
        df = pd.read_csv(csv_buffer)
        df = add_derived_columns(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read or process CSV: {e}")

    # 2. Load rules config
    if yaml_file:
        try:
            yaml_bytes = await yaml_file.read()
            rules_text = yaml_bytes.decode("utf-8", errors="ignore")
            config = load_rules_from_text(rules_text)
        except Exception as e:
            # load_rules_from_text raises HTTPException on failure
            raise
    else:
        config = load_rules_from_text(default_rules_yaml_text())
        
    # 3. Determine domain pack
    columns = list(df.columns)
    
    # If no override, try to infer
    if domain_pack_override is None:
        chosen_pack = infer_domain_pack(config, columns)
    else:
        # Validate override against available packs
        packs = (config.get("domain_packs") or {}).keys()
        if domain_pack_override not in packs:
            raise HTTPException(status_code=400, detail=f"Invalid domain_pack_override: '{domain_pack_override}'. Available packs are: {', '.join(packs)}")
        chosen_pack = domain_pack_override

    # 4. Run expectations
    try:
        results_df = build_and_run_expectations(df, config, chosen_pack)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running expectations: {e}")

    # 5. Prepare and return JSON response
    # Select and rename columns for the final response, dropping the complex 'result' dictionary
    response_df = results_df[[
        "id", "label", "why", "column", "success", "fail_count",
        "expectation_type", "kwargs"
    ]]

    # Convert the DataFrame to a list of dictionaries (JSON records)
    json_results = response_df.to_dict(orient="records")

    return JSONResponse(content=json_results)

# =========================
# Execution Instructions
# =========================

# To run this FastAPI application:
# 1. Install required libraries:
#    pip install fastapi uvicorn pandas numpy pyyaml great-expectations[spark] matplotlib
# 2. Save the code as 'fastapi_app.py'
# 3. Run the server:
#    uvicorn fastapi_app:app --reload
# 4. Access the API documentation at http://127.0.0.1:8000/docs to test the endpoint.

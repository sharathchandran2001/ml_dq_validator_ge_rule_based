import io
import re
from datetime import date
from typing import Dict, Any, List, Optional
from io import BytesIO

import pandas as pd
import numpy as np

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Corrected Great Expectations Imports for modern versions
from great_expectations.validator.validator import Validator
# 🚨 FIX: New import required for Validator instantiation 🚨
from great_expectations.execution_engine import PandasExecutionEngine 

# =========================
# Initialize FastAPI app
# =========================

app = FastAPI(
    title="Simple CSV Data Quality API (Hardcoded Rules)",
    description="Validates an uploaded CSV against hardcoded Great Expectations rules.",
    version="1.0.0"
)

# =========================
# Schemas for API Response
# =========================

class ExpectationResult(BaseModel):
    label: str
    column: Optional[str]
    success: bool
    fail_count: int
    expectation_type: str
    kwargs: Dict[str, Any]

# =========================
# Helper Functions
# =========================

def compute_age(dob_series: pd.Series) -> pd.Series:
    """Computes age from a DateOfBirth column after parsing."""
    today = date.today()
    
    def calculate(dob):
        if pd.isna(dob):
            return np.nan
        # Ensure DOB is a date object
        if isinstance(dob, str):
            dob = pd.to_datetime(dob, errors='coerce')
        if pd.isna(dob):
            return np.nan
        
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    
    return dob_series.apply(calculate)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'Age' column if 'DateOfBirth' exists."""
    df_copy = df.copy()
    
    # Standardize column name case for checking
    columns_lower = {c.lower(): c for c in df.columns}
    
    if "dateofbirth" in columns_lower and "age" not in columns_lower:
        dob_col = columns_lower["dateofbirth"]
        df_copy["Age"] = compute_age(df_copy[dob_col])
        
    return df_copy


def fail_count_from_result(r: Dict[str, Any]) -> int:
    """Extracts the number of failing rows from a GE result dictionary."""
    if r.get("unexpected_count") is not None:
        try:
            return int(r["unexpected_count"])
        except Exception:
            pass
    uil = r.get("unexpected_index_list")
    if isinstance(uil, list):
        return len(uil)
    return 0

# =========================
# Hardcoded Validation Logic (The Core Change)
# =========================

def hardcoded_expectations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defines and runs a fixed set of high-value expectations on the DataFrame.
    """
    
    # 1. Instantiate the Validator
    # 🚨 FIX APPLIED HERE: Added execution_engine argument 🚨
    validator = Validator(
        context=None, 
        batch={"data": df},
        execution_engine=PandasExecutionEngine(dialect=None)
    )
    
    results = []

    def add_result(ev, label, column=None):
        if ev is None:
            return
        
        evd = ev.to_json_dict() if hasattr(ev, "to_json_dict") else dict(ev)
        
        expectation_config = evd.get("expectation_config", {}) or {}
        result_details = evd.get("result", {}) or {}

        fail_count = fail_count_from_result(result_details)

        results.append({
            "label": label,
            "column": column,
            "success": bool(evd.get("success", False)),
            "expectation_type": expectation_config.get("expectation_type", ""),
            "kwargs": expectation_config.get("kwargs", {}),
            "fail_count": fail_count
        })

    # --- A. Table-Level Checks ---
    add_result(validator.expect_table_row_count_to_be_between(min_value=1),
               "Table must not be empty")

    # --- B. Column-Specific Checks ---
    columns_lower = {c.lower(): c for c in df.columns}
    
    # Rule 1: Email Validation 
    if any(re.match(r'.*email', col, re.IGNORECASE) for col in df.columns):
        email_col = next(col for col in df.columns if re.match(r'.*email', col, re.IGNORECASE))
        add_result(
            validator.expect_column_values_to_not_be_null(email_col, mostly=0.99),
            f"{email_col}: Not null (99%)",
            column=email_col
        )
        add_result(
            validator.expect_column_values_to_match_regex(
                email_col, 
                "^[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}$", 
                mostly=0.95
            ),
            f"{email_col}: Valid email format (95%)",
            column=email_col
        )

    # Rule 2: Age Range
    if "age" in columns_lower:
        age_col = columns_lower["age"]
        add_result(
            validator.expect_column_values_to_be_between(age_col, min_value=18, max_value=100, mostly=0.99),
            f"{age_col}: Must be between 18 and 100 (99%)",
            column=age_col
        )
        
    # Rule 3: Identifier Uniqueness
    if "customerid" in columns_lower:
        id_col = columns_lower["customerid"]
        add_result(
            validator.expect_column_values_to_be_unique(id_col),
            f"{id_col}: Must be unique",
            column=id_col
        )
        
    # Rule 4: CreditScore Bounds
    if "creditscore" in columns_lower:
        score_col = columns_lower["creditscore"]
        add_result(
            validator.expect_column_values_to_be_between(score_col, min_value=300, max_value=850, mostly=0.99),
            f"{score_col}: Must be standard FICO range (99%)",
            column=score_col
        )
        
    return pd.DataFrame(results)

# =========================
# FastAPI Endpoint
# =========================

@app.post("/validate_csv", response_model=List[ExpectationResult])
async def validate_csv(
    csv_file: UploadFile = File(..., description="The CSV file to validate.")
):
    """
    Runs hardcoded data quality checks on an uploaded CSV file.
    """
    
    # 1. Basic File Check
    if csv_file.content_type not in ('text/csv', 'application/octet-stream') and not csv_file.filename.lower().endswith(".csv"):
         raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")
         
    # 2. Load and Prepare CSV
    try:
        csv_bytes = await csv_file.read()
        csv_buffer = BytesIO(csv_bytes)
        df = pd.read_csv(csv_buffer)
        df = add_derived_columns(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read or process CSV: {e}")

    # 3. Run Hardcoded Expectations
    try:
        results_df = hardcoded_expectations(df)
    except Exception as e:
        print(f"Internal error during expectation run: {e}")
        # Reraise a clean 500 error for the API client
        raise HTTPException(status_code=500, detail="Internal error running expectations. Check server logs.")

    # 4. Prepare and return JSON response
    response_df = results_df[[
        "label", "column", "success", "fail_count",
        "expectation_type", "kwargs"
    ]]

    json_results = response_df.to_dict(orient="records")

    return JSONResponse(content=json_results)

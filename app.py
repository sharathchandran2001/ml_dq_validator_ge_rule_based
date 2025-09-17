import io
import json
import re
from datetime import date
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Great Expectations (dataset API)
from great_expectations.dataset import PandasDataset

st.set_page_config(page_title="Customer Data Quality (Great Expectations)", layout="wide")

st.title("Data Quality using Great Expectations - POC app")
st.caption("Upload a CSV of customer records. We'll run expectation checks and visualize any violations.")

# --- Sidebar ---
with st.sidebar:
    st.header("Options")
    sample_preview_rows = st.number_input("Preview rows", 5, 100, 15, step=5)
    show_pass_expectations = st.checkbox("Show passing expectations", value=False)
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- CSV must include headers (column names).")
    st.markdown("- Common columns recognized: CustomerID, Email, Phone, DateOfBirth, AccountOpenDate, "
                "AccountType, CreditScore, Balance, Status")

# --- File uploader ---
uploaded = st.file_uploader("Upload customer CSV", type=["csv"])


# Helpers
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


def add_derived_columns(df):
    if "DateOfBirth" in df.columns and "Age" not in df.columns:
        df = df.copy()
        df["Age"] = df["DateOfBirth"].apply(compute_age)
    return df


# --- Expectations ---
def run_expectations(df: pd.DataFrame) -> pd.DataFrame:
    gdf = PandasDataset(df.copy())
    results = []
    eid = 0

    def add_result(evresult, label, why, column=None):
        nonlocal eid
        ev = evresult.to_json_dict() if hasattr(evresult, "to_json_dict") else dict(evresult)
        results.append({
            "id": eid,
            "label": label,
            "why": why,
            "column": column,
            "success": bool(ev.get("success", False)),
            "result": ev.get("result", {}) or {},
            "expectation_type": (ev.get("expectation_config", {}) or {}).get("expectation_type", ""),
            "kwargs": (ev.get("expectation_config", {}) or {}).get("kwargs", {}),
        })
        eid += 1

    # Core checks
    add_result(gdf.expect_table_row_count_to_be_between(min_value=1),
               "Table has at least 1 row", "Files should not be empty.")
    add_result(gdf.expect_table_columns_to_match_ordered_list(list(df.columns)),
               "Columns appear as uploaded", "Prevents silent column reordering or loss.")

    # CustomerID
    if "CustomerID" in df.columns:
        add_result(gdf.expect_column_values_to_not_be_null("CustomerID"),
                   "CustomerID: not null", "Primary identifier cannot be blank.", "CustomerID")
        add_result(gdf.expect_column_values_to_be_unique("CustomerID"),
                   "CustomerID: unique", "Avoid duplicate identifiers.", "CustomerID")

    # Email
    if "Email" in df.columns:
        regex = r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
        add_result(gdf.expect_column_values_to_match_regex("Email", regex, mostly=0.98),
                   "Email: valid format (≈98%+)", "Ensures contactability.", "Email")

    # Phone
    if "Phone" in df.columns:
        regex = r"^[\d\-\+\s\(\)ext\.]{7,}$"
        add_result(gdf.expect_column_values_to_match_regex("Phone", regex, mostly=0.95),
                   "Phone: plausible (≈95%+)", "Catches invalid phone values.", "Phone")

    # Date checks
    if "AccountOpenDate" in df.columns:
        parsed = df["AccountOpenDate"].apply(parse_date_safe)
        temp = PandasDataset(df.copy())
        temp["__AccountOpenDate__parsed"] = parsed
        add_result(temp.expect_column_values_to_not_be_null("__AccountOpenDate__parsed", mostly=0.98),
                   "AccountOpenDate: parseable date (≈98%+)", "Detects malformed dates.", "AccountOpenDate")
        add_result(temp.expect_column_max_to_be_between("__AccountOpenDate__parsed", max_value=pd.Timestamp.today()),
                   "AccountOpenDate: not in the future", "Open date cannot be in the future.", "AccountOpenDate")

    if "DateOfBirth" in df.columns:
        parsed = df["DateOfBirth"].apply(parse_date_safe)
        temp = PandasDataset(df.copy())
        temp["__DateOfBirth__parsed"] = parsed
        add_result(temp.expect_column_values_to_not_be_null("__DateOfBirth__parsed", mostly=0.98),
                   "DateOfBirth: parseable date (≈98%+)", "Detects malformed dates.", "DateOfBirth")

    # Age
    if "Age" in df.columns:
        temp = PandasDataset(df.copy())
        add_result(temp.expect_column_values_to_be_between("Age", 18, 100, mostly=0.99),
                   "Age: between 18 and 100 (≈99%+)", "Regulatory/eligibility constraints.", "Age")

    # CreditScore
    if "CreditScore" in df.columns:
        add_result(gdf.expect_column_values_to_be_between("CreditScore", 300, 850, mostly=0.99),
                   "CreditScore: 300–850 (≈99%+)", "Standard FICO-like bounds.", "CreditScore")

    # Balance
    if "Balance" in df.columns:
        add_result(gdf.expect_column_values_to_be_between("Balance", min_value=0, mostly=0.995),
                   "Balance: non-negative (≈99.5%+)", "Balances should not be negative.", "Balance")

    # Status
    if "Status" in df.columns:
        allowed = ["Active", "Pending", "Closed", "Suspended"]
        add_result(gdf.expect_column_values_to_be_in_set("Status", allowed, mostly=0.98),
                   f"Status: in {allowed} (≈98%+)", "Enforces account states.", "Status")

    # AccountType
    if "AccountType" in df.columns:
        allowed = ["Checking", "Savings", "Credit", "Mortgage", "Loan"]
        add_result(gdf.expect_column_values_to_be_in_set("AccountType", allowed, mostly=0.98),
                   f"AccountType: in {allowed} (≈98%+)", "Keeps taxonomy consistent.", "AccountType")

    results_df = pd.DataFrame(results)

    # Count failing rows (row-level violations)
    def get_unexpected_count(r):
        if not isinstance(r, dict):
            return 0
        return int(r.get("unexpected_count", 0) or 0)

    results_df["fail_count"] = results_df["result"].apply(get_unexpected_count)
    return results_df


# --- Visuals ---
def render_violations_chart(results_df):
    counts = results_df.set_index("label")["fail_count"]
    counts = counts[counts > 0].sort_values(ascending=True)

    if counts.empty:
        st.success("No expectation violations 🎉")
        return

    fig = plt.figure()
    counts.plot(kind="barh")
    plt.xlabel("Number of failing rows")
    plt.ylabel("Expectation")
    plt.title("Expectation Violations")
    st.pyplot(fig)


def failed_examples(df, results_df, max_rows=25):
    samples = {}
    for _, row in results_df.iterrows():
        if row["fail_count"] <= 0:
            continue
        col = row.get("column")
        etype = row["expectation_type"]
        kwargs = row["kwargs"]

        bad_idx = None
        uil = (row["result"] or {}).get("unexpected_index_list")
        if uil:
            bad_idx = uil
        else:
            if etype == "expect_column_values_to_not_be_null" and col in df.columns:
                bad_idx = df[df[col].isna()].index
            elif etype == "expect_column_values_to_be_unique" and col in df.columns:
                bad_idx = df[df[col].duplicated(keep=False)].index
            elif etype == "expect_column_values_to_match_regex" and col in df.columns and "regex" in kwargs:
                regex = kwargs["regex"]
                bad_idx = df[~df[col].astype(str).str.match(regex, na=False)].index
            elif etype == "expect_column_values_to_be_between" and col in df.columns:
                min_v = kwargs.get("min_value", -np.inf)
                max_v = kwargs.get("max_value", np.inf)
                bad_idx = df[~df[col].between(min_v, max_v)].index
            elif etype == "expect_column_values_to_be_in_set" and col in df.columns:
                allowed = set(kwargs.get("value_set", []))
                bad_idx = df[~df[col].isin(allowed)].index

        if bad_idx is not None and len(bad_idx) > 0:
            samples[row["label"]] = df.loc[list(bad_idx)].head(max_rows)
    return samples


# --- Main display ---
def show_results(df):
    st.subheader("1) Data Preview")
    st.dataframe(df.head(sample_preview_rows))

    st.subheader("2) Expectations & Results")
    results_df = run_expectations(df)
    view = results_df[["label", "column", "fail_count"]].rename(columns={"fail_count": "failing_rows"})
    if not show_pass_expectations:
        view = view[view["failing_rows"] > 0]
    st.dataframe(view, use_container_width=True)

    st.subheader("3) Violations Dashboard")
    render_violations_chart(results_df if show_pass_expectations else results_df[results_df["fail_count"] > 0])

    st.subheader("4) Explanations")
    samples = failed_examples(df, results_df)
    for _, r in results_df.iterrows():
        if r["fail_count"] <= 0 and not show_pass_expectations:
            continue
        with st.expander(f'{r["label"]} — failing rows: {r["fail_count"]}'):
            st.markdown(f"**Why it matters:** {r['why']}")
            st.markdown("**Expectation type:** " + (r.get("expectation_type") or "n/a"))
            if r["label"] in samples:
                st.markdown("**Sample failing rows:**")
                st.dataframe(samples[r["label"]])

    st.subheader("5) Raw Validation JSON")
    json_payload = results_df.to_json(orient="records", indent=2)
    st.download_button("Download results.json", data=json_payload, file_name="ge_results.json", mime="application/json")


# --- Run ---
if uploaded is None:
    st.info("Upload a CSV to begin.")
else:
    try:
        df = pd.read_csv(uploaded)
        df = add_derived_columns(df)
        show_results(df)
    except Exception as e:
        st.error(f"Failed to read/validate file: {e}")

import io
import json
import re
from datetime import date
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import yaml

# Great Expectations (dataset API)
from great_expectations.dataset import PandasDataset

st.set_page_config(page_title="Universal Data Quality (Great Expectations + YAML)", layout="wide")
st.title("✅ Universal Data Quality — Great Expectations + YAML")
st.caption("Upload any CSV + an optional rules.yaml. The app builds expectations dynamically (generic + domain packs).")


# =========================
# Sidebar & Uploads
# =========================
with st.sidebar:
    st.header("Options")
    sample_preview_rows = st.number_input("Preview rows", min_value=5, max_value=200, value=15, step=5)
    show_pass_expectations = st.checkbox("Show passing expectations", value=False)
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- Upload **rules.yaml** or rely on built-in defaults.")
    st.markdown("- Domain packs: `customer`, `credit_card`, `loan` (editable in YAML).")
    st.markdown("- Rules support regex-based column matching (e.g., `Email|email|user_email`).")

csv_file = st.file_uploader("Upload CSV (any domain)", type=["csv"])
yaml_file = st.file_uploader("Upload rules.yaml (optional)", type=["yaml", "yml"])


# =========================
# Helpers
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
        st.error(f"Failed to parse rules.yaml: {e}")
        return {}


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
                if any(re.search(pattern, c, re.I) for c in columns):
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
        tmp = PandasDataset(df.copy())
        tmp[f"__{col}__parsed"] = pd.to_datetime(df[col], errors="coerce")
        return tmp.expect_column_values_to_not_be_null(f"__{col}__parsed", mostly=mostly)

    # You can add more types here: nonnegative, zscore_outliers, url/email/etc.
    return None


def build_and_run_expectations(df: pd.DataFrame, config: Dict[str, Any], chosen_pack: Optional[str]) -> pd.DataFrame:
    """Run table-level checks + generic column rules + optional domain pack rules. Return tidy results DataFrame."""
    gdf = PandasDataset(df.copy())
    results = []
    eid = 0

    def add_result(ev, label, why, column=None):
        nonlocal eid
        if ev is None:
            return
        evd = ev.to_json_dict() if hasattr(ev, "to_json_dict") else dict(ev)
        results.append({
            "id": eid,
            "label": label,
            "why": why,
            "column": column,
            "success": bool(evd.get("success", False)),
            "result": evd.get("result", {}) or {},
            "expectation_type": (evd.get("expectation_config", {}) or {}).get("expectation_type", ""),
            "kwargs": (evd.get("expectation_config", {}) or {}).get("kwargs", {}),
        })
        eid += 1

    # --- Table-level generic checks ---
    add_result(gdf.expect_table_row_count_to_be_between(min_value=1),
               "Table has at least 1 row", "Files should not be empty.")
    add_result(gdf.expect_table_columns_to_match_ordered_list(list(df.columns)),
               "Columns appear as uploaded", "Prevents silent column reordering or loss.")

    # --- Generic column rules from config["columns"] ---
    for rule in (config.get("columns") or []):
        pattern = rule.get("name", "")
        match_type = rule.get("match", "regex").lower()
        for col in df.columns:
            matched = (col == pattern) if match_type == "exact" else bool(re.search(pattern, col, re.I))
            if not matched:
                continue
            for exp in (rule.get("expectations") or []):
                why = exp.get("why", "Data quality constraint.")
                ev = _apply_expectation(gdf, df, col, exp)
                label = make_label(col, exp)
                add_result(ev, label, why, column=col)

    # --- Domain pack rules (optional) ---
    if chosen_pack:
        pack = (config.get("domain_packs") or {}).get(chosen_pack, {})
        for rule in (pack.get("columns") or []):
            pattern = rule.get("name", "")
            match_type = rule.get("match", "regex").lower()
            for col in df.columns:
                matched = (col == pattern) if match_type == "exact" else bool(re.search(pattern, col, re.I))
                if not matched:
                    continue
                for exp in (rule.get("expectations") or []):
                    why = exp.get("why", f"Domain rule from pack '{chosen_pack}'.")
                    ev = _apply_expectation(gdf, df, col, exp)
                    label = make_label(col, exp)
                    add_result(ev, label, why, column=col)

    results_df = pd.DataFrame(results)

    # --- Row-level fail counts ---
    def fail_count_from_result(r: Dict[str, Any]) -> int:
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
        return 0  # fallback

    results_df["fail_count"] = results_df["result"].apply(fail_count_from_result)
    return results_df


def make_label(col: str, exp: Dict[str, Any]) -> str:
    t = exp.get("type", "rule")
    if t == "regex":
        return f"{col}: matches regex"
    if t in ("between", "ge_between"):
        rng = f"{exp.get('min', '')}–{exp.get('max', '')}".strip("–")
        return f"{col}: between {rng}".strip()
    if t == "in_set":
        return f"{col}: in {exp.get('values', [])}"
    if t == "not_null":
        return f"{col}: not null"
    if t == "unique":
        return f"{col}: unique"
    if t == "date_parseable":
        return f"{col}: parseable date"
    return f"{col}: {t}"


def render_violations_chart(results_df: pd.DataFrame):
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


def failed_examples(df: pd.DataFrame, results_df: pd.DataFrame, max_rows=25) -> Dict[str, pd.DataFrame]:
    samples = {}
    for _, r in results_df.iterrows():
        if r["fail_count"] <= 0:
            continue
        label = r["label"]
        col = r.get("column")
        etype = r.get("expectation_type")
        kwargs = r.get("kwargs") or {}

        bad_idx = None
        # Preferred: unexpected_index_list from GE
        uil = (r["result"] or {}).get("unexpected_index_list")
        if uil:
            bad_idx = uil
        else:
            # Heuristics if indexes not captured
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
            samples[label] = df.loc[list(bad_idx)].head(max_rows)
    return samples


# =========================
# Main flow
# =========================
if csv_file is None:
    st.info("Upload a CSV to begin.")
else:
    try:
        df = pd.read_csv(csv_file)
        df = add_derived_columns(df)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Load rules
    if yaml_file is not None:
        rules_text = yaml_file.read().decode("utf-8", errors="ignore")
        config = load_rules_from_text(rules_text)
        st.success("Loaded rules.yaml from upload.")
    else:
        config = load_rules_from_text(default_rules_yaml_text())
        st.info("No rules.yaml uploaded — using built-in defaults.")

    # Domain inference + selector
    guess = infer_domain_pack(config, list(df.columns)) or "None"
    packs = list((config.get("domain_packs") or {}).keys())
    domain = st.selectbox("Domain pack (optional)", ["None"] + packs, index=(["None"] + packs).index(guess) if guess in (["None"] + packs) else 0)
    chosen_pack = None if domain == "None" else domain

    # Show preview
    st.subheader("1) Data Preview")
    st.dataframe(df.head(sample_preview_rows))

    # Run expectations
    st.subheader("2) Expectations & Results")
    results_df = build_and_run_expectations(df, config, chosen_pack)
    view = results_df[["label", "column", "fail_count"]].rename(columns={"fail_count": "failing_rows"})
    if not show_pass_expectations:
        view = view[view["failing_rows"] > 0]
    st.dataframe(view, use_container_width=True)

    # Chart
    st.subheader("3) Violations Summary")
    render_violations_chart(results_df if show_pass_expectations else results_df[results_df["fail_count"] > 0])

    # Why & Examples
    st.subheader("4) Why it matters & Sample failing rows")
    samples = failed_examples(df, results_df)
    for _, r in results_df.iterrows():
        if r["fail_count"] <= 0 and not show_pass_expectations:
            continue
        with st.expander(f'{r["label"]} — failing rows: {r["fail_count"]}'):
            why = r.get("why") or "Data quality constraint."
            st.markdown(f"**Why it matters:** {why}")
            st.markdown("**Expectation type:** " + (r.get("expectation_type") or "n/a"))
            if r["label"] in samples:
                st.markdown("**Sample failing rows:**")
                st.dataframe(samples[r["label"]])

    # Download JSON
    st.subheader("5) Raw Validation JSON")
    json_payload = results_df.to_json(orient="records", indent=2)
    st.download_button("Download results.json", data=json_payload, file_name="ge_results.json", mime="application/json")

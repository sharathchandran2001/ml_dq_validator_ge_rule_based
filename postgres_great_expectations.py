import os
from datetime import datetime
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# --- Great Expectations (GX) ---
import great_expectations as gx

# ------------------ ENV / CONFIG ------------------
ENV_DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "admin"),
    "database": os.getenv("DB_NAME", "MA"),
    "schema": os.getenv("DB_SCHEMA", "public"),
}

DEFAULT_TABLE = os.getenv("DB_TABLE", "customer_abc")  # change if your table name differs

# ------------------ HELPERS ------------------
def get_engine(env):
    url = (
        f"postgresql+psycopg2://{env['user']}:{env['password']}"
        f"@{env['host']}:{env['port']}/{env['database']}"
    )
    return create_engine(url, pool_pre_ping=True, future=True)

def load_table(env, table_name):
    engine = get_engine(env)
    schema = env["schema"]
    with engine.connect() as conn:
        # Confirm table exists
        conn.execute(text("SELECT 1"))
        df = pd.read_sql(
            f'SELECT * FROM "{schema}"."{table_name}"',
            con=conn,
        )
    return df

def build_and_run_expectations(df):
    """
    Uses the simple, code-only GX API:
      dfx = gx.from_pandas(df)
      ... add expectations ...
      return dfx.validate()
    """
    # Make sure types look reasonable
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    dfx = gx.from_pandas(df)

    # --- Expectations based on your screenshot/columns ---
    # Columns seen: customer_id, first_name, last_name, email, dob, address, phone_number, fax_number

    # customer_id
    if "customer_id" in dfx.columns:
        dfx.expect_column_values_to_not_be_null("customer_id")
        dfx.expect_column_values_to_be_unique("customer_id")

    # first_name / last_name
    for col in ["first_name", "last_name"]:
        if col in dfx.columns:
            dfx.expect_column_values_to_not_be_null(col)
            dfx.expect_column_value_lengths_to_be_between(col, min_value=1, max_value=50)

    # email
    if "email" in dfx.columns:
        dfx.expect_column_values_to_not_be_null("email")
        dfx.expect_column_values_to_match_regex(
            "email",
            r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        )
        # If your business rule allows duplicate emails, remove the next line.
        dfx.expect_column_values_to_be_unique("email")

    # dob
    if "dob" in dfx.columns:
        dfx.expect_column_values_to_not_be_null("dob")
        dfx.expect_column_values_to_be_between(
            "dob",
            min_value=pd.Timestamp("1900-01-01"),
            max_value=pd.Timestamp(datetime.utcnow().date()),
            parse_strings_as_datetimes=True,
        )

    # address / phone_number / fax_number are allowed to be null in your screenshot.
    # If present, add light-format checks (only when value exists).
    if "phone_number" in dfx.columns:
        dfx.expect_column_values_to_match_regex(
            "phone_number",
            r"^\+?[0-9\-\s\(\)]+$",
            mostly=0.95,  # allow some blanks/non-standard formats
            condition_parser="pandas"  # default; included for clarity
        )

    # Run validations
    results = dfx.validate()
    return results

def summarize_results(results):
    """
    Returns (total_tests, passed, failed), and a tidy list of failed detail rows.
    """
    stats = results["statistics"]
    total = stats.get("evaluated_expectations", 0)
    successful = stats.get("successful_expectations", 0)
    failed = total - successful

    failed_details = []
    for ev in results.get("results", []):
        if not ev.get("success", False):
            expect = ev.get("expectation_config", {}).get("expectation_type", "unknown")
            kwargs = ev.get("expectation_config", {}).get("kwargs", {})
            res = ev.get("result", {}) or {}

            # Try to pull unexpected rows/values when available
            # Newer GX often returns indices; fall back to values if present
            unexpected_indices = res.get("unexpected_index_list")
            unexpected_values = res.get("unexpected_list")

            failed_details.append({
                "expectation": expect,
                "column": kwargs.get("column"),
                "kwargs": {k: v for k, v in kwargs.items() if k != "column"},
                "unexpected_indices": unexpected_indices,
                "unexpected_values": unexpected_values,
                "observed_value": res.get("observed_value"),
                "partial_unexpected_counts": res.get("partial_unexpected_counts"),
            })
    return total, successful, failed, failed_details

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Great Expectations — Postgres Validation", layout="wide")
st.title("✅ Great Expectations — Validate Postgres Table")

with st.sidebar:
    st.subheader("Database")
    st.caption("Using environment defaults if left blank.")
    host = st.text_input("Host", ENV_DB["host"])
    port = st.text_input("Port", ENV_DB["port"])
    user = st.text_input("User", ENV_DB["user"])
    password = st.text_input("Password", ENV_DB["password"], type="password")
    database = st.text_input("Database", ENV_DB["database"])
    schema = st.text_input("Schema", ENV_DB["schema"])
    table = st.text_input("Table name", DEFAULT_TABLE)
    run_btn = st.button("Run Validation", type="primary")

# Reflect sidebar edits into ENV_DB used below
ENV_DB_RUNTIME = {
    "host": host, "port": port, "user": user, "password": password,
    "database": database, "schema": schema
}

if run_btn:
    with st.spinner("Connecting to Postgres and validating…"):
        try:
            df = load_table(ENV_DB_RUNTIME, table)
        except Exception as e:
            st.error(f"Database error: {e}")
        else:
            st.success(f"Loaded {len(df):,} rows from {schema}.{table}")
            with st.expander("Preview data", expanded=False):
                st.dataframe(df.head(50), use_container_width=True)

            try:
                results = build_and_run_expectations(df)
            except Exception as e:
                st.error(f"Validation error (GX): {e}")
            else:
                total, passed, failed, failed_details = summarize_results(results)

                st.subheader("Validation Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total expectations", total)
                c2.metric("Passed", passed)
                c3.metric("Failed", failed)

                # Overall JSON (optional)
                with st.expander("Raw GX result JSON", expanded=False):
                    st.json(results)

                # Failed details
                if failed == 0:
                    st.success("All expectations passed 🎉")
                else:
                    st.warning("Some expectations failed. See details below.")

                    # Flatten a view for tabular inspection
                    rows = []
                    for i, fd in enumerate(failed_details, 1):
                        rows.append({
                            "№": i,
                            "Expectation": fd["expectation"],
                            "Column": fd["column"],
                            "Observed (summary)": str(fd["observed_value"]),
                            "Unexpected indices": (
                                ", ".join(map(str, fd["unexpected_indices"]))[:200]
                                if fd["unexpected_indices"] else ""
                            ),
                            "Unexpected values (sample)": (
                                ", ".join(map(str, (fd["unexpected_values"] or [])[:10]))
                            ),
                            "Args": str(fd["kwargs"]),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    # Per-expectation deep dive
                    for i, fd in enumerate(failed_details, 1):
                        with st.expander(f"❌ {i}. {fd['expectation']} — column={fd['column']}", expanded=False):
                            st.write("**Arguments:**", fd["kwargs"])
                            if fd["observed_value"] is not None:
                                st.write("**Observed value (summary):**", fd["observed_value"])
                            if fd["unexpected_indices"]:
                                st.write("**Unexpected row indices:**", fd["unexpected_indices"])
                            if fd["unexpected_values"]:
                                st.write("**Unexpected values (sample):**", fd["unexpected_values"])
                            if fd["partial_unexpected_counts"]:
                                st.write("**Partial unexpected counts:**", fd["partial_unexpected_counts"])

else:
    st.info("Set your DB info in the sidebar and click **Run Validation** to start.")

st.caption(
    "Tips: adjust the expectations in `build_and_run_expectations()` to fit business rules. "
    "For example, remove email uniqueness if duplicates are allowed."
)

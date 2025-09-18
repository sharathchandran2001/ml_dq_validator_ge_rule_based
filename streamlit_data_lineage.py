import os
from typing import List, Tuple, Dict, Any
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from pyvis.network import Network
import networkx as nx
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# --------------------------
# ---- Constants  ----------
# --------------------------
DEFAULT_PG_HOST = os.getenv("PG_HOST", "localhost")
DEFAULT_PG_PORT = os.getenv("PG_PORT", "5432")
DEFAULT_PG_DB = os.getenv("PG_DB", "MA")
DEFAULT_PG_USER = os.getenv("PG_USER", "postgres")
DEFAULT_PG_PASS = os.getenv("PG_PASS", "admin")
DEFAULT_PG_SCHEMA = os.getenv("PG_SCHEMA", "public")
DEFAULT_TABLE_A = "customer_abc"
DEFAULT_TABLE_B = "customer_bcd"
DEFAULT_EXPECTED_EXTRA_IN_B = "fax_customer"

# --------------------------
# ---- Sidebar: Config  ----
# --------------------------
st.set_page_config(page_title="Postgres Lineage & DQ", layout="wide")

st.sidebar.header("Postgres Connection")
pg_host = st.sidebar.text_input("Host", value=DEFAULT_PG_HOST)
pg_port = st.sidebar.text_input("Port", value=DEFAULT_PG_PORT)
pg_db = st.sidebar.text_input("Database", value=DEFAULT_PG_DB)
pg_user = st.sidebar.text_input("User", value=DEFAULT_PG_USER)
pg_pass = st.sidebar.text_input("Password", type="password", value=DEFAULT_PG_PASS)
schema = st.sidebar.text_input("Schema", value=DEFAULT_PG_SCHEMA)

st.sidebar.header("Table Configuration")
table_a = st.sidebar.text_input("Source table (A)", value=DEFAULT_TABLE_A)
table_b = st.sidebar.text_input("Derived table (B)", value=DEFAULT_TABLE_B)
expected_extra_in_b_input = st.sidebar.text_input(
    "Expected extra columns in B (comma-separated)",
    value=DEFAULT_EXPECTED_EXTRA_IN_B
)
expected_extra_in_b = {col.strip() for col in expected_extra_in_b_input.split(',') if col.strip()}

st.sidebar.header("Optional Checks")
run_gx = st.sidebar.checkbox("Run Great Expectations validation", value=False)
analyze_btn = st.sidebar.button("Analyze Data Lineage & Quality")

# --------------------------
# ---- Helper functions ----
# --------------------------
@st.cache_resource
def make_engine(host, port, db, user, password) -> Tuple[bool, object, str]:
    """
    Creates and tests a SQLAlchemy engine for PostgreSQL.
    Caches the engine to prevent reconnecting on every rerun.
    """
    try:
        connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        eng = create_engine(connection_string)
        # Quick ping to verify connection
        with eng.connect() as conn: # Use .connect() then .close() or with statement
            conn.execute(text("SELECT 1"))
        return True, eng, ""
    except Exception as e:
        return False, None, str(e)

@st.cache_data(ttl=300) # Cache for 5 minutes
def get_columns(_engine, schema_name: str, tbl: str) -> List[str]: # Added underscore
    """Fetches column names for a given table and schema."""
    if not tbl:
        return []
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name AND table_name = :table_name
        ORDER BY ordinal_position
    """)
    with _engine.begin() as conn: # Use _engine
        rows = conn.execute(q, {"schema_name": schema_name, "table_name": tbl}).fetchall()
    return [r[0] for r in rows]

@st.cache_data(ttl=300) # Cache for 5 minutes
def get_row_count(_engine, schema_name: str, tbl: str) -> int: # Added underscore
    """Fetches the row count for a given table and schema."""
    if not tbl:
        return 0
    with _engine.begin() as conn: # Use _engine
        return conn.execute(text(f'SELECT COUNT(*) FROM "{schema_name}"."{tbl}"')).scalar()

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_df(_engine, schema_name: str, tbl: str, cols: List[str]) -> pd.DataFrame: # Added underscore
    """Fetches data from a table for specified columns."""
    if not tbl or not cols:
        return pd.DataFrame()
    quoted_cols = ','.join([f'"{c}"' for c in cols])
    return pd.read_sql(f'SELECT {quoted_cols} FROM "{schema_name}"."{tbl}"', _engine) # Use _engine

def lineage_graph_html(source_table: str, derived_table: str, issue_text: str, schema_ok: bool) -> str:
    """
    Builds a simple A -> B lineage graph and colors the edge based on schema_ok.
    """
    G = nx.DiGraph()
    G.add_node(source_table)
    G.add_node(derived_table)
    edge_color = "red" if not schema_ok else "green"
    G.add_edge(source_table, derived_table, title=issue_text or "Derived", color=edge_color)

    net = Network(height="420px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="black")
    net.from_nx(G)

    # Style nodes
    for node in net.nodes:
        node_id = node["id"]
        node["shape"] = "box"
        node["font"] = {"size": 14}
        if node_id == source_table:
            node["color"] = {"background": "#e8f4ff", "border": "#3182bd"}
            node["title"] = f"Source: {source_table}"
        elif node_id == derived_table:
            node["color"] = {"background": "#f9f6ff", "border": "#805ad5"}
            node["title"] = f"Derived: {derived_table}"

    net.toggle_physics(True)
    return net.generate_html()

def run_great_expectations_validation(_engine, connection_string: str, schema_name: str, table_names: List[str], pg_user: str, pg_pass: str, pg_host: str, pg_port: str, pg_db: str) -> None: # Added underscore
    """Runs Great Expectations validation for specified tables."""
    try:
        import great_expectations as gx
        from great_expectations.checkpoint import SimpleCheckpoint

        # Initialize DataContext (ephemeral mode for script-based usage)
        context = gx.get_context(mode="ephemeral")

        # Add PostgreSQL datasource using the modern API
        ds = context.data_sources.add_postgres(
            name="pg_ds",
            connection_string=connection_string
        )

        for tbl in table_names:
            if not tbl:
                st.warning(f"Skipping Great Expectations for an empty table name.")
                continue

            with st.spinner(f"Running GX validation for `{tbl}`..."):
                suite_name = f"{tbl}_suite"
                suite = context.suites.add(suite_name)

                # Add expectations
                # Pass _engine to get_columns
                current_columns = get_columns(_engine, schema_name, tbl)
                if current_columns:
                    suite.add_expectation("expect_table_columns_to_contain_set", {
                        "column_set": current_columns
                    })
                else:
                    st.warning(f"No columns found for {tbl}. Skipping column expectations.")

                # Pass _engine to get_row_count
                rc = get_row_count(_engine, schema_name, tbl)
                suite.add_expectation("expect_table_row_count_to_be_between", {
                    "min_value": max(0, int(rc * 0.9)), # Allow 0 for empty tables
                    "max_value": int(max(1, rc) * 1.1) + 1
                })

                # Example non-null expectations for common columns
                common_important_cols = ["id", "customer_id", "first_name", "last_name", "email"]
                for col in common_important_cols:
                    if col in current_columns:
                        suite.add_expectation("expect_column_values_to_not_be_null", {"column": col})

                # Create a table asset
                asset = ds.add_table_asset(name=f"{tbl}_asset", table_name=tbl, schema_name=schema_name)
                batch_request = asset.build_batch_request()

                # Create and run checkpoint
                checkpoint_name = f"{tbl}_checkpoint"
                checkpoint = SimpleCheckpoint(
                    name=checkpoint_name,
                    data_context=context,
                    validations=[
                        {
                            "batch_request": batch_request,
                            "expectation_suite_name": suite_name
                        }
                    ],
                )
                result = checkpoint.run()

                if result["success"]:
                    st.success(f"**GX validation for `{tbl}`: ✅ SUCCESS**")
                else:
                    st.error(f"**GX validation for `{tbl}`: ❌ FAILED**")
                    # Optionally display validation results or link to Data Docs
                    # For ephemeral context, Data Docs are not saved by default
                    # For more persistent usage, you'd configure a store for data docs
                st.write(f"See validation result details for `{tbl}` in the logs or by inspecting the `result` object.")

    except ImportError:
        st.warning("Great Expectations is not installed. Please `pip install great_expectations` to enable this feature.")
    except Exception as e:
        st.error(f"Great Expectations validation failed: {e}")
        st.info("Ensure Great Expectations, SQLAlchemy, and psycopg2 are installed (`pip install great_expectations sqlalchemy psycopg2-binary`).")


# --------------------------
# --------- Main UI --------
# --------------------------
st.title("🔗 Data Lineage & Quality (PostgreSQL)")

if not analyze_btn:
    st.info("Fill the connection details and table names on the left and click **Analyze Data Lineage & Quality**.")
    st.stop()

if not table_a or not table_b:
    st.error("Please provide both source (A) and derived (B) table names.")
    st.stop()

st.subheader("Database Connection")
with st.spinner("Attempting to connect to PostgreSQL..."):
    ok, engine, err = make_engine(pg_host, pg_port, pg_db, pg_user, pg_pass)

if not ok:
    st.error(f"Database connection failed: {err}")
    st.stop()
else:
    st.success("Successfully connected to PostgreSQL!")

# --- Fetch Schema + counts ---
with st.spinner(f"Fetching schema and row counts for `{table_a}` and `{table_b}`..."):
    try:
        # Pass the engine object to the cached functions
        cols_a = get_columns(engine, schema, table_a)
        cols_b = get_columns(engine, schema, table_b)
        count_a = get_row_count(engine, schema, table_a)
        count_b = get_row_count(engine, schema, table_b)
    except Exception as e:
        st.error(f"Failed to read schema or row counts: {e}")
        st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"{table_a} rows", count_a)
with col2:
    st.metric(f"{table_b} rows", count_b)

set_a, set_b = set(cols_a), set(cols_b)
shared = sorted(list(set_a.intersection(set_b)))
only_in_a = sorted(list(set_a - set_b))
only_in_b = sorted(list(set_b - set_a))

with col3:
    st.metric("Shared columns", len(shared))

st.subheader("Schema Comparison")
schema_df = pd.DataFrame({
    f'{table_a} Columns': pd.Series(cols_a),
    f'{table_b} Columns': pd.Series(cols_b)
}).fillna('')
st.dataframe(schema_df, use_container_width=True)

st.write(f"**Columns only in `{table_a}`:**", only_in_a if only_in_a else "—")
st.write(f"**Columns only in `{table_b}`:**", only_in_b if only_in_b else "—")

# --- Lineage rule: B should be A + expected_extra_in_b (and nothing else) ---
schema_ok = (len(only_in_a) == 0) and (set(only_in_b) == expected_extra_in_b)

st.subheader("Lineage Rule Check")
if schema_ok:
    st.success(f"Lineage check passed: `{table_b}` contains all columns from `{table_a}` and only the expected extra columns: `{', '.join(expected_extra_in_b) if expected_extra_in_b else 'none'}`.")
else:
    details = []
    if only_in_a:
        details.append(f"Missing in `{table_b}`: {only_in_a}")
    unexpected = sorted(list(set(only_in_b) - expected_extra_in_b))
    if unexpected:
        details.append(f"Unexpected in `{table_b}`: {unexpected}")
    missing_expected = sorted(list(expected_extra_in_b - set(only_in_b)))
    if missing_expected:
        details.append(f"Expected but not found in `{table_b}`: {missing_expected}")
    st.error("Lineage check failed: " + " | ".join(details))

# --- Value-level comparison on shared columns ---
st.subheader("Value Comparison (shared columns)")
PKEY = "customer_id" if ("customer_id" in shared) else None

with st.spinner(f"Fetching data for value comparison on shared columns..."):
    try:
        # Pass the engine object to the cached functions
        df_a = fetch_df(engine, schema, table_a, shared)
        df_b = fetch_df(engine, schema, table_b, shared)
    except Exception as e:
        st.error(f"Failed to fetch data for comparison: {e}")
        st.stop()

if df_a.empty and df_b.empty:
    st.info("No data to compare as both tables are empty or have no shared columns.")
elif df_a.empty:
    st.warning(f"Table `{table_a}` is empty. Cannot perform value comparison.")
elif df_b.empty:
    st.warning(f"Table `{table_b}` is empty. Cannot perform value comparison.")
else:
    if PKEY and PKEY in df_a.columns and PKEY in df_b.columns:
        df_a_keyed = df_a.set_index(PKEY).sort_index()
        df_b_keyed = df_b.set_index(PKEY).sort_index()

        missing_in_b = df_a_keyed.index.difference(df_b_keyed.index)
        missing_in_a = df_b_keyed.index.difference(df_a_keyed.index)
        compare_cols = [c for c in shared if c != PKEY]

        common_index = df_a_keyed.index.intersection(df_b_keyed.index)
        if not common_index.empty and compare_cols:
            # Align common rows and compare
            diffs = (df_a_keyed.loc[common_index, compare_cols] !=
                     df_b_keyed.loc[common_index, compare_cols])
            mismatched_idx = diffs.any(axis=1)
            mismatched_keys = common_index[mismatched_idx].tolist()
        else:
            mismatched_keys = []

        colA, colB, colC = st.columns(3)
        colA.metric(f"Missing in {table_b} (by PK)", len(missing_in_b))
        colB.metric(f"Missing in {table_a} (by PK)", len(missing_in_a))
        colC.metric("Mismatched rows (by PK)", len(mismatched_keys))

        if missing_in_b.any():
            st.write(f"**Sample primary keys missing in `{table_b}`:**", missing_in_b.tolist()[:10])
        if missing_in_a.any():
            st.write(f"**Sample primary keys missing in `{table_a}`:**", missing_in_a.tolist()[:10])
        if mismatched_keys:
            st.write(f"**Sample primary keys with differing values in shared columns:**", mismatched_keys[:10])
            if len(mismatched_keys) > 0:
                st.subheader("Sample Mismatched Rows (First 5)")
                sample_mismatch_keys = mismatched_keys[:5]
                for key in sample_mismatch_keys:
                    st.write(f"--- Primary Key: `{key}` ---")
                    st.json({
                        f"{table_a}": df_a_keyed.loc[key, compare_cols].to_dict(),
                        f"{table_b}": df_b_keyed.loc[key, compare_cols].to_dict()
                    })

    else:
        st.info(f"No primary key '{PKEY}' found in shared columns or not applicable. Using hash-based comparison.")
        # Hash-based comparison if no obvious PK
        def row_hash(df: pd.DataFrame) -> pd.Series:
            # Ensure all columns are treated as strings before hashing
            return (df.astype(str)
                      .apply(lambda r: "§".join(r.values.astype(str)), axis=1) # Ensure values are str
                      .apply(lambda s: hash(s))) # Use Python's built-in hash

        with st.spinner("Performing hash-based row comparison..."):
            hA = row_hash(df_a)
            hB = row_hash(df_b)
            vcA = hA.value_counts()
            vcB = hB.value_counts()

        missing_or_diff = []
        for h, cnt_a in vcA.items():
            cnt_b = vcB.get(h, 0)
            if cnt_b != cnt_a:
                missing_or_diff.append({"hash": h, f"count_in_{table_a}": cnt_a, f"count_in_{table_b}": cnt_b})

        if missing_or_diff:
            st.metric("Unique row-signature differences", len(missing_or_diff))
            st.write("Sample hash differences (hash, count_in_A, count_in_B):")
            st.dataframe(pd.DataFrame(missing_or_diff).head(10))
        else:
            st.success("No row-signature differences detected between shared columns.")

# --- Nulls overview (shared columns)
if shared:
    st.subheader("Nulls Overview (shared columns)")
    null_data = []
    with st.spinner("Checking for null values..."):
        with engine.begin() as conn: # This connection is outside a cached function, so `engine` is fine here
            for col in shared:
                nA = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{table_a}" WHERE "{col}" IS NULL')).scalar()
                nB = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{table_b}" WHERE "{col}" IS NULL')).scalar()
                null_data.append({"column": col, f"nulls_{table_a}": nA, f"nulls_{table_b}": nB})
    st.dataframe(pd.DataFrame(null_data), use_container_width=True)
else:
    st.info("No shared columns to check for null values.")

# --- Lineage graph (pyvis -> HTML -> component)
st.subheader("Lineage Graph")
issue_text = ""
if schema_ok:
    issue_text = f"Derived: {table_b} = {table_a} + expected extra column(s)"
else:
    msgs = []
    if only_in_a:
        msgs.append(f"Missing in {table_b}: {only_in_a}")
    unexpected = sorted(list(set(only_in_b) - expected_extra_in_b))
    if unexpected:
        msgs.append(f"Unexpected in {table_b}: {unexpected}")
    missing_expected = sorted(list(expected_extra_in_b - set(only_in_b)))
    if missing_expected:
        msgs.append(f"Expected but missing in {table_b}: {missing_expected}")
    issue_text = " | ".join(msgs) if msgs else "Schema mismatch"

html = lineage_graph_html(table_a, table_b, issue_text, schema_ok)
st.components.v1.html(html, height=460, scrolling=False)

# --- Optional: Great Expectations quick validation
if run_gx:
    st.subheader("Great Expectations Validation")
    conn_str = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    # Pass the engine object to the Great Expectations function
    run_great_expectations_validation(engine, conn_str, schema, [table_a, table_b], pg_user, pg_pass, pg_host, pg_port, pg_db)
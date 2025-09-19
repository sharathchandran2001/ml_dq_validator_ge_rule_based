# app_lineage_improved.py
# -------------------------------------------------------------
# Streamlit app: Postgres Data Lineage & Data Quality (GX)
#
# Features
# 1) Connects to Postgres (sidebar inputs, .env supported)
# 2) Compares schemas of A (source) and B (derived)
#    - B should be A + expected extra columns (e.g., fax_customer)
#    - Detects type drift (data_type/length/precision changes)
# 3) Compares values on shared columns
#    - Uses discovered PRIMARY KEY(S) if identical on both tables
#    - Otherwise uses deterministic (SHA-256) row-signature hashing
#    - Null-safe equality for accurate diffs
# 4) Shows nulls by column across A and B
# 5) Renders lineage graph (pyvis) with red/green edge + rich tooltip
# 6) (Optional) Great Expectations quick validation
#    - A: columns must match exactly cols(A)
#    - B: columns must match cols(A) ∪ expected_extras
#    - Rowcount equality checks
#    - Non-null checks for common important columns if present
#
# Run:
#   pip install streamlit pandas sqlalchemy psycopg2-binary networkx pyvis python-dotenv great-expectations
#   streamlit run app_lineage_improved.py
#
# -------------------------------------------------------------

import os
import re
import json
import hashlib
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from pyvis.network import Network
import networkx as nx
from dotenv import load_dotenv

# --------------------------
# ---- Load .env (optional)
# --------------------------
load_dotenv()

# --------------------------
# ---- Defaults via .env ---
# --------------------------
DEFAULT_PG_HOST = os.getenv("PG_HOST", "localhost")
DEFAULT_PG_PORT = os.getenv("PG_PORT", "5432")
DEFAULT_PG_DB = os.getenv("PG_DB", "MA")
DEFAULT_PG_USER = os.getenv("PG_USER", "postgres")
DEFAULT_PG_PASS = os.getenv("PG_PASS", "admin")
DEFAULT_PG_SCHEMA = os.getenv("PG_SCHEMA", "public")
DEFAULT_TABLE_A = os.getenv("TABLE_A", "customer_abc")
DEFAULT_TABLE_B = os.getenv("TABLE_B", "customer_bcd")
DEFAULT_EXPECTED_EXTRA_IN_B = os.getenv("EXPECTED_EXTRAS_B", "fax_customer")  # comma-separated ok

# --------------------------
# ---- Streamlit Setup -----
# --------------------------
st.set_page_config(page_title="Postgres Lineage & DQ", layout="wide")
st.title("🔗 Data Lineage & Quality (PostgreSQL)")

# Sidebar: connection
st.sidebar.header("Postgres Connection")
pg_host = st.sidebar.text_input("Host", value=DEFAULT_PG_HOST)
pg_port = st.sidebar.text_input("Port", value=DEFAULT_PG_PORT)
pg_db = st.sidebar.text_input("Database", value=DEFAULT_PG_DB)
pg_user = st.sidebar.text_input("User", value=DEFAULT_PG_USER)
pg_pass = st.sidebar.text_input("Password", type="password", value=DEFAULT_PG_PASS)
schema = st.sidebar.text_input("Schema", value=DEFAULT_PG_SCHEMA)

# Sidebar: tables & settings
st.sidebar.header("Table Configuration")
table_a = st.sidebar.text_input("Source table (A)", value=DEFAULT_TABLE_A)
table_b = st.sidebar.text_input("Derived table (B)", value=DEFAULT_TABLE_B)

expected_extra_in_b_input = st.sidebar.text_input(
    "Expected extra columns in B (comma-separated)",
    value=DEFAULT_EXPECTED_EXTRA_IN_B
)
expected_extra_in_b: set = {c.strip() for c in expected_extra_in_b_input.split(",") if c.strip()}

# Optional checks
st.sidebar.header("Options")
run_gx = st.sidebar.checkbox("Run Great Expectations validation", value=False)
show_download = st.sidebar.checkbox("Enable JSON report download", value=True)
analyze_btn = st.sidebar.button("Analyze Data Lineage & Quality")

# --------------------------
# ---- Utilities -----------
# --------------------------
def valid_ident(name: str) -> bool:
    """Very basic identifier validation: letters, digits, underscore; not empty; starts with letter/_."""
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))

def safe_ident_or_raise(name: str, label: str) -> str:
    if not valid_ident(name):
        raise ValueError(f"Invalid identifier for {label}: {name!r}")
    return name

def build_conn_str(user: str, pwd: str, host: str, port: str, db: str) -> str:
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

@st.cache_resource(show_spinner=False)
def get_engine(conn_str: str) -> Engine:
    eng = create_engine(conn_str)
    with eng.connect() as conn:
        conn.execute(text("SELECT 1"))
    return eng

@st.cache_data(ttl=300, show_spinner=False)
def get_columns(conn_str: str, schema_name: str, tbl: str) -> List[str]:
    if not tbl:
        return []
    safe_ident_or_raise(schema_name, "schema")
    safe_ident_or_raise(tbl, "table")

    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
    """)
    eng = get_engine(conn_str)
    with eng.begin() as conn:
        rows = conn.execute(q, {"schema": schema_name, "table": tbl}).fetchall()
    return [r[0] for r in rows]

@st.cache_data(ttl=300, show_spinner=False)
def get_columns_with_types(conn_str: str, schema_name: str, tbl: str) -> pd.DataFrame:
    if not tbl:
        return pd.DataFrame()
    safe_ident_or_raise(schema_name, "schema")
    safe_ident_or_raise(tbl, "table")

    q = text("""
        SELECT column_name, data_type, udt_name,
               character_maximum_length, numeric_precision, numeric_scale,
               is_nullable
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
    """)
    eng = get_engine(conn_str)
    with eng.begin() as conn:
        return pd.read_sql(q, conn, params={"schema": schema_name, "table": tbl})

@st.cache_data(ttl=300, show_spinner=False)
def get_row_count(conn_str: str, schema_name: str, tbl: str) -> int:
    if not tbl:
        return 0
    safe_ident_or_raise(schema_name, "schema")
    safe_ident_or_raise(tbl, "table")
    eng = get_engine(conn_str)
    with eng.begin() as conn:
        return conn.execute(text(f'SELECT COUNT(*) FROM "{schema_name}"."{tbl}"')).scalar()

def fetch_df(conn_str: str, schema_name: str, tbl: str, cols: List[str]) -> pd.DataFrame:
    if not tbl or not cols:
        return pd.DataFrame()
    safe_ident_or_raise(schema_name, "schema")
    safe_ident_or_raise(tbl, "table")
    # Validate column identifiers used in SELECT
    for c in cols:
        safe_ident_or_raise(c, f'column "{tbl}"')
    quoted_cols = ",".join([f'"{c}"' for c in cols])
    eng = get_engine(conn_str)
    return pd.read_sql(f'SELECT {quoted_cols} FROM "{schema_name}"."{tbl}"', eng)

@st.cache_data(ttl=300, show_spinner=False)
def get_primary_key_columns(conn_str: str, schema_name: str, tbl: str) -> List[str]:
    """Return PK columns in defined order; supports composite keys."""
    if not tbl:
        return []
    safe_ident_or_raise(schema_name, "schema")
    safe_ident_or_raise(tbl, "table")
    # Use regclass for schema-qualified lookup
    q = text("""
        SELECT a.attname
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indrelid
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relname = :table
          AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
    """)
    eng = get_engine(conn_str)
    with eng.begin() as conn:
        rows = conn.execute(q, {"schema": schema_name, "table": tbl}).fetchall()
    return [r[0] for r in rows]

def stable_row_hash(df: pd.DataFrame) -> pd.Series:
    """Deterministic hash across sorted columns, null-safe."""
    if df.empty:
        return pd.Series(dtype="object")
    cols = sorted(df.columns)
    normalized = (
        df[cols]
        .astype(str)
        .replace({"nan": "", "NaT": ""})
        .apply(lambda r: "§".join(r.values), axis=1)
    )
    return normalized.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())

def nullsafe_neq(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Null-aware not-equal: returns True where values differ, ignoring (NaN vs NaN)."""
    # Align columns
    cols = [c for c in a.columns if c in b.columns]
    a2, b2 = a[cols].copy(), b[cols].copy()
    # Simple normalization for object columns & whitespace
    for c in cols:
        if a2[c].dtype == "object" or b2[c].dtype == "object":
            a2[c] = a2[c].astype(str).str.strip().replace({"nan": None})
            b2[c] = b2[c].astype(str).str.strip().replace({"nan": None})
    return (a2 != b2) & ~(a2.isna() & b2.isna())

def lineage_graph_html(source_table: str, derived_table: str, issue_text: str, schema_ok: bool) -> str:
    """Build a simple A -> B lineage graph; color edge by status."""
    G = nx.DiGraph()
    G.add_node(source_table)
    G.add_node(derived_table)
    edge_color = "green" if schema_ok else "red"
    G.add_edge(source_table, derived_table, title=issue_text or "Derived", color=edge_color)

    net = Network(height="420px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="black")
    net.from_nx(G)

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

def run_great_expectations_validation(
    conn_str: str,
    schema_name: str,
    table_a: str,
    table_b: str,
    expected_extras_b: set,
    cols_a: List[str],
    cols_b: List[str],
    rowcount_a: int,
) -> None:
    """Run a compact GX validation with graceful fallback between modern/legacy APIs."""
    try:
        import great_expectations as gx
        from great_expectations.checkpoint import SimpleCheckpoint

        context = gx.get_context(mode="ephemeral")

        # Some GX versions expose `data_sources.add_postgres`; others use `sources.add_sqlalchemy`.
        try:
            ds = context.data_sources.add_postgres(name="pg_ds", connection_string=conn_str)
            using_modern_api = True
        except Exception:
            ds = context.sources.add_sqlalchemy(name="pg_ds", connection_string=conn_str)
            using_modern_api = False

        # A suite: exact columns = cols_a; rowcount equals rowcount_a
        suiteA = context.suites.add(f"{table_a}_suite")
        suiteA.add_expectation("expect_table_columns_to_match_set", {"column_set": cols_a})
        suiteA.add_expectation("expect_table_row_count_to_equal", {"value": rowcount_a})
        for col in ["id", "customer_id", "first_name", "last_name", "email"]:
            if col in cols_a:
                suiteA.add_expectation("expect_column_values_to_not_be_null", {"column": col})
        if "email" in cols_a:
            suiteA.add_expectation(
                "expect_column_values_to_match_regex",
                {"column": "email", "regex": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
            )

        # B suite: columns = cols_a ∪ expected_extras; rowcount equals rowcount_a
        expected_b_cols = sorted(list(set(cols_a) | set(expected_extras_b)))
        suiteB = context.suites.add(f"{table_b}_suite")
        suiteB.add_expectation("expect_table_columns_to_match_set", {"column_set": expected_b_cols})
        suiteB.add_expectation("expect_table_row_count_to_equal", {"value": rowcount_a})
        for col in ["id", "customer_id", "first_name", "last_name", "email"]:
            if col in cols_b:
                suiteB.add_expectation("expect_column_values_to_not_be_null", {"column": col})
        if "email" in cols_b:
            suiteB.add_expectation(
                "expect_column_values_to_match_regex",
                {"column": "email", "regex": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
            )

        if using_modern_api:
            assetA = ds.add_table_asset(name=f"{table_a}_asset", table_name=table_a, schema_name=schema_name)
            assetB = ds.add_table_asset(name=f"{table_b}_asset", table_name=table_b, schema_name=schema_name)
        else:
            assetA = ds.add_table_asset(name=f"{table_a}_asset", table_name=table_a, schema_name=schema_name)
            assetB = ds.add_table_asset(name=f"{table_b}_asset", table_name=table_b, schema_name=schema_name)

        cpA = SimpleCheckpoint(
            name=f"{table_a}_checkpoint",
            data_context=context,
            validations=[{"batch_request": assetA.build_batch_request(), "expectation_suite_name": suiteA.name}],
        )
        resA = cpA.run()
        st.write(f"**GX validation for `{table_a}`:** {'✅ SUCCESS' if resA['success'] else '❌ FAILED'}")

        cpB = SimpleCheckpoint(
            name=f"{table_b}_checkpoint",
            data_context=context,
            validations=[{"batch_request": assetB.build_batch_request(), "expectation_suite_name": suiteB.name}],
        )
        resB = cpB.run()
        st.write(f"**GX validation for `{table_b}`:** {'✅ SUCCESS' if resB['success'] else '❌ FAILED'}")

        st.caption("Tip: Configure GX stores & Data Docs for persistent history instead of ephemeral mode.")

    except ImportError:
        st.warning("Great Expectations is not installed. `pip install great-expectations` to enable this feature.")
    except Exception as e:
        st.error(f"Great Expectations validation failed: {e}")

# --------------------------
# --------- MAIN -----------
# --------------------------
if not analyze_btn:
    st.info("Fill the connection details and table names on the left and click **Analyze Data Lineage & Quality**.")
    st.stop()

# Basic validation
try:
    safe_ident_or_raise(schema, "schema")
    safe_ident_or_raise(table_a, "table A")
    safe_ident_or_raise(table_b, "table B")
except ValueError as ve:
    st.error(str(ve))
    st.stop()

conn_str = build_conn_str(pg_user, pg_pass, pg_host, pg_port, pg_db)

# Connection check
with st.spinner("Connecting to PostgreSQL..."):
    try:
        engine = get_engine(conn_str)
        st.success("Connected to PostgreSQL.")
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

# Fetch schema + counts
with st.spinner(f"Fetching schema and row counts for `{table_a}` and `{table_b}`..."):
    try:
        cols_a = get_columns(conn_str, schema, table_a)
        cols_b = get_columns(conn_str, schema, table_b)
        dtypes_a = get_columns_with_types(conn_str, schema, table_a)
        dtypes_b = get_columns_with_types(conn_str, schema, table_b)
        count_a = get_row_count(conn_str, schema, table_a)
        count_b = get_row_count(conn_str, schema, table_b)
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
schema_df = pd.DataFrame({f"{table_a} Columns": pd.Series(cols_a), f"{table_b} Columns": pd.Series(cols_b)}).fillna("")
st.dataframe(schema_df, use_container_width=True)
st.write(f"**Columns only in `{table_a}`:**", only_in_a if only_in_a else "—")
st.write(f"**Columns only in `{table_b}`:**", only_in_b if only_in_b else "—")

# Type drift detection on shared columns
type_issues: List[str] = []
if not dtypes_a.empty and not dtypes_b.empty:
    a_map = dtypes_a.set_index("column_name")
    b_map = dtypes_b.set_index("column_name")
    for col in shared:
        ra = a_map.loc[col]
        rb = b_map.loc[col]
        sig_a = (str(ra["data_type"]), str(ra["character_maximum_length"]), str(ra["numeric_precision"]), str(ra["numeric_scale"]))
        sig_b = (str(rb["data_type"]), str(rb["character_maximum_length"]), str(rb["numeric_precision"]), str(rb["numeric_scale"]))
        if sig_a != sig_b:
            type_issues.append(col)

# Lineage rule: B should be A + expected_extras (and nothing else), and no type drift
schema_ok = (len(only_in_a) == 0) and (set(only_in_b) == expected_extra_in_b) and (len(type_issues) == 0)

st.subheader("Lineage Rule Check")
if schema_ok:
    st.success(
        f"Lineage check passed: `{table_b}` contains all columns from `{table_a}` "
        f"and only the expected extra column(s): `{', '.join(sorted(expected_extra_in_b)) if expected_extra_in_b else 'none'}`. "
        "No type drift detected."
    )
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
    if type_issues:
        details.append(f"Type drift on shared columns: {type_issues}")
    st.error("Lineage check failed: " + " | ".join(details))

# Value comparison on shared columns
st.subheader("Value Comparison (shared columns)")

if not shared:
    st.info("No shared columns between A and B; skipping value comparison.")
    df_a = pd.DataFrame()
    df_b = pd.DataFrame()
else:
    with st.spinner("Fetching data for value comparison on shared columns..."):
        try:
            df_a = fetch_df(conn_str, schema, table_a, shared)
            df_b = fetch_df(conn_str, schema, table_b, shared)
        except Exception as e:
            st.error(f"Failed to fetch data for comparison: {e}")
            df_a = pd.DataFrame()
            df_b = pd.DataFrame()

if df_a.empty and df_b.empty and shared:
    st.warning("Both tables appear empty over shared columns. Nothing to compare.")
elif not df_a.empty and not df_b.empty and shared:
    # Discover PKs on both tables; only use if identical.
    pk_a = get_primary_key_columns(conn_str, schema, table_a)
    pk_b = get_primary_key_columns(conn_str, schema, table_b)
    use_pks = pk_a if pk_a and (pk_a == pk_b) and all(p in shared for p in pk_a) else []

    if use_pks:
        st.write(f"Primary key(s) used for comparison: `{use_pks}`")
        df_a_keyed = df_a.set_index(use_pks).sort_index()
        df_b_keyed = df_b.set_index(use_pks).sort_index()

        # Missing keys
        missing_in_b = df_a_keyed.index.difference(df_b_keyed.index)
        missing_in_a = df_b_keyed.index.difference(df_a_keyed.index)

        # Compare only on non-PK shared columns
        compare_cols = [c for c in shared if c not in use_pks]
        common_index = df_a_keyed.index.intersection(df_b_keyed.index)
        if len(common_index) and compare_cols:
            neq_mask = nullsafe_neq(
                df_a_keyed.loc[common_index, compare_cols],
                df_b_keyed.loc[common_index, compare_cols]
            )
            mismatched_keys = list(common_index[neq_mask.any(axis=1)])
        else:
            mismatched_keys = []

        colA, colB, colC = st.columns(3)
        colA.metric(f"Missing in {table_b} (by PK)", len(missing_in_b))
        colB.metric(f"Missing in {table_a} (by PK)", len(missing_in_a))
        colC.metric("Mismatched rows (by PK)", len(mismatched_keys))

        if len(missing_in_b) > 0:
            st.write(f"**Sample keys missing in `{table_b}`:**", list(missing_in_b)[:10])
        if len(missing_in_a) > 0:
            st.write(f"**Sample keys missing in `{table_a}`:**", list(missing_in_a)[:10])
        if mismatched_keys:
            st.write(f"**Sample keys with differing values in shared columns:**", mismatched_keys[:10])
            st.subheader("Sample Mismatched Rows (First 5)")
            for key in mismatched_keys[:5]:
                a_row = df_a_keyed.loc[key, compare_cols]
                b_row = df_b_keyed.loc[key, compare_cols]
                st.json({f"{table_a}": a_row.to_dict(), f"{table_b}": b_row.to_dict()})

    else:
        st.info("No identical primary key(s) detected on both tables; using deterministic row-signature comparison.")
        with st.spinner("Computing row signatures (SHA-256) over shared columns..."):
            sig_a = stable_row_hash(df_a)
            sig_b = stable_row_hash(df_b)
            vc_a = sig_a.value_counts()
            vc_b = sig_b.value_counts()

        diffs = []
        for h, cnt_a in vc_a.items():
            cnt_b = int(vc_b.get(h, 0))
            if cnt_b != cnt_a:
                diffs.append({"hash": h, f"count_in_{table_a}": int(cnt_a), f"count_in_{table_b}": cnt_b})

        if diffs:
            st.metric("Unique row-signature differences", len(diffs))
            st.dataframe(pd.DataFrame(diffs).head(10), use_container_width=True)
        else:
            st.success("No row-signature differences detected between shared columns.")

# Nulls overview (shared columns)
if shared:
    st.subheader("Nulls Overview (shared columns)")
    null_rows: List[Dict[str, Any]] = []
    with st.spinner("Calculating null counts..."):
        eng = get_engine(conn_str)
        with eng.begin() as conn:
            for col in shared:
                safe_ident_or_raise(col, f'column "{table_a}"')
                nA = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{table_a}" WHERE "{col}" IS NULL')).scalar()
                nB = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{table_b}" WHERE "{col}" IS NULL')).scalar()
                rateA = (nA / count_a) if count_a else 0
                rateB = (nB / count_b) if count_b else 0
                null_rows.append({
                    "column": col,
                    f"nulls_{table_a}": nA,
                    f"rate_{table_a}": f"{rateA:.2%}",
                    f"nulls_{table_b}": nB,
                    f"rate_{table_b}": f"{rateB:.2%}",
                })
    st.dataframe(pd.DataFrame(null_rows), use_container_width=True)

# Lineage graph
st.subheader("Lineage Graph")
tooltip_parts = []
if only_in_a:
    tooltip_parts.append(f"Missing in {table_b}: {only_in_a}")
if set(only_in_b) - expected_extra_in_b:
    tooltip_parts.append(f"Unexpected in {table_b}: {sorted(list(set(only_in_b) - expected_extra_in_b))}")
if expected_extra_in_b - set(only_in_b):
    tooltip_parts.append(f"Expected but missing in {table_b}: {sorted(list(expected_extra_in_b - set(only_in_b)))}")
if type_issues:
    tooltip_parts.append(f"Type drift: {type_issues}")
if count_a != count_b:
    tooltip_parts.append(f"Rowcount A={count_a} vs B={count_b}")

issue_text = " | ".join(tooltip_parts) if tooltip_parts else "Derived: B = A + expected extras"
html = lineage_graph_html(table_a, table_b, issue_text, schema_ok)
st.components.v1.html(html, height=460, scrolling=False)

# Great Expectations (optional)
if run_gx:
    st.subheader("Great Expectations Validation")
    with st.spinner("Running Great Expectations..."):
        run_great_expectations_validation(
            conn_str=conn_str,
            schema_name=schema,
            table_a=table_a,
            table_b=table_b,
            expected_extras_b=expected_extra_in_b,
            cols_a=cols_a,
            cols_b=cols_b,
            rowcount_a=count_a,
        )

# Download JSON report (optional)
if show_download:
    report = {
        "tables": {"A": table_a, "B": table_b, "schema": schema},
        "row_counts": {"A": count_a, "B": count_b},
        "schema": {
            "A_columns": cols_a,
            "B_columns": cols_b,
            "shared": shared,
            "only_in_A": only_in_a,
            "only_in_B": only_in_b,
            "type_drift_columns": type_issues,
            "expected_extras_in_B": sorted(list(expected_extra_in_b)),
            "schema_ok": schema_ok,
        },
        "notes": issue_text,
    }
    st.download_button(
        "⬇️ Download lineage report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="lineage_report.json",
        mime="application/json",
    )

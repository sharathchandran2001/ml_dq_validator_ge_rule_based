import os
import json
import psycopg2
import streamlit as st
import tempfile
from pyvis.network import Network
from dotenv import load_dotenv
import streamlit.components.v1 as components
from typing import List, Dict, Set, Any, Tuple

# ------------------------------------------------------
# Load environment variables
# ------------------------------------------------------
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "admin")
DB_NAME = os.getenv("DB_NAME", "MA")

# ------------------------------------------------------
# Utility to get table columns
# ------------------------------------------------------
@st.cache_data(ttl=300) # Cache column fetches for 5 minutes
def get_table_columns(conn_params: Dict[str, str], table_name: str) -> List[str]:
    """Fetch column names for a table from PostgreSQL."""
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                """, (table_name,))
                return [row[0] for row in cur.fetchall()]
    except psycopg2.Error as e:
        st.error(f"Database error fetching columns for {table_name}: {e}")
        return []

# ------------------------------------------------------
# Build Lineage Graph
# ------------------------------------------------------
def build_lineage_graph(
    source_table: str,
    cols_a: List[str],
    target_table: str,
    cols_b: List[str],
    expected_extra_in_b: Set[str],
    ignored_source_columns: Set[str],
    column_mappings: Dict[str, str]
) -> Tuple[Network, List[str], List[str], List[str]]:
    """
    Builds an interactive lineage graph using pyvis, highlighting column relationships and issues.
    """
    net = Network(height="750px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="black", notebook=False)
    net.barnes_hut() # Use Barnes-Hut algorithm for physics layout

    # Add main table nodes
    net.add_node(source_table, label=f"Source: {source_table}", shape="box", color={"background": "#ADD8E6", "border": "#3182bd"}, title=f"Source Table: {source_table}")
    net.add_node(target_table, label=f"Target: {target_table}", shape="box", color={"background": "#90EE90", "border": "#2D6A4F"}, title=f"Target Table: {target_table}")

    # Add an overall lineage edge (can be styled based on overall success/failure)
    # This edge represents the overall data flow
    net.add_edge(source_table, target_table, title="Overall Data Flow", color={"color": "#6B46C1", "highlight": "#805ad5"}, width=2, arrows="to")

    # Prepare column sets after applying mappings and considering ignored columns
    mapped_source_cols = {column_mappings.get(col, col) for col in cols_a if col not in ignored_source_columns}
    target_cols_set = set(cols_b)

    shared = sorted(list(mapped_source_cols.intersection(target_cols_set)))
    only_in_source = sorted(list(mapped_source_cols - target_cols_set))
    only_in_target = sorted(list(target_cols_set - mapped_source_cols))

    # Add nodes and edges for columns
    for col_a_original in cols_a:
        if col_a_original in ignored_source_columns:
            # Add ignored columns as distinct nodes, not participating in lineage flow
            net.add_node(f"{source_table}.{col_a_original}",
                         label=f"{col_a_original} (Ignored)",
                         shape="dot",
                         color={"background": "#C0C0C0", "border": "#808080"}, # Grey for ignored
                         title=f"Column '{col_a_original}' in {source_table} is ignored.")
            continue # Skip further processing for ignored columns

        mapped_col_a = column_mappings.get(col_a_original, col_a_original)
        source_col_node_id = f"{source_table}.{col_a_original}"
        target_col_node_id = f"{target_table}.{mapped_col_a}"

        # Add node for source column
        net.add_node(source_col_node_id,
                     label=col_a_original,
                     shape="dot",
                     color={"background": "#E0FFFF", "border": "#3182bd"}, # Lighter blue for source columns
                     title=f"Source column: {col_a_original} in {source_table}")

        # If source column maps to a shared column in target
        if mapped_col_a in shared:
            # Ensure target column node exists (it might be added implicitly later, but explicit is clear)
            net.add_node(target_col_node_id,
                         label=mapped_col_a,
                         shape="dot",
                         color={"background": "#E0FFE0", "border": "#2D6A4F"}, # Lighter green for target columns
                         title=f"Target column: {mapped_col_a} in {target_table}")

            net.add_edge(source_col_node_id, target_col_node_id,
                         label="maps_to",
                         color={"color": "#4CAF50", "highlight": "#66BB6A"}, # Green for mapping
                         title=f"'{col_a_original}' from {source_table} maps to '{mapped_col_a}' in {target_table}")
        elif mapped_col_a in only_in_source: # If after mapping, it's still only in source
            # Highlight as missing in target, no direct edge to target column
            net.add_node(source_col_node_id,
                         color={"background": "#FFDDC1", "border": "#E65100"}, # Orange for missing in target
                         title=f"Column '{col_a_original}' (mapped to '{mapped_col_a}') in {source_table} is MISSING in {target_table}")
            # Optional: Add an edge to the target table, but make it a dashed "missing" edge
            net.add_edge(source_col_node_id, target_table,
                         label="missing_in_target",
                         color={"color": "#FF8F00", "highlight": "#FFB300"},
                         dashes=True,
                         title=f"Column '{col_a_original}' (mapped to '{mapped_col_a}') from {source_table} is MISSING in {target_table}")


    # Columns only in target (extras)
    for col_b in only_in_target:
        target_col_node_id = f"{target_table}.{col_b}"
        color_info = {"background": "#F0F0F0", "border": "#A9A9A9"} # Default grey
        label_text = "Expected Extra"
        title_text = f"Column '{col_b}' in {target_table} is an expected extra."

        if col_b not in expected_extra_in_b:
            color_info = {"background": "#FFCCCC", "border": "#D32F2F"} # Red for unexpected
            label_text = "Unexpected Extra"
            title_text = f"Column '{col_b}' in {target_table} is an UNEXPECTED extra!"

        net.add_node(target_col_node_id,
                     label=col_b,
                     shape="dot",
                     color=color_info,
                     title=title_text)

        # Optional: Add an edge from the target table to the extra column
        net.add_edge(target_table, target_col_node_id,
                     label=label_text,
                     color={"color": color_info["border"], "highlight": color_info["background"]},
                     dashes=True if col_b not in expected_extra_in_b else False, # Dashed for unexpected
                     width=1,
                     arrows="to",
                     title=title_text)

    # Enable physics for better layout, but make it stable
    net.toggle_physics(True)
    return net, shared, only_in_source, only_in_target

# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.set_page_config(page_title="Data Lineage Visualizer", layout="wide")
st.title("🔗 Data Lineage Visualizer (PostgreSQL)")

# Sidebar – Mapping JSON
st.sidebar.header("Lineage Configuration (JSON)")
mapping_file = st.sidebar.file_uploader("Upload Lineage Config (JSON)", type=["json"])

lineage_jobs_config: List[Dict[str, Any]] = []
selected_job_index = 0

if mapping_file:
    try:
        config = json.load(mapping_file)
        if "lineage_jobs" in config and isinstance(config["lineage_jobs"], list):
            lineage_jobs_config = config["lineage_jobs"]
            st.sidebar.success(f"✅ Loaded {len(lineage_jobs_config)} lineage job(s).")

            job_options = [f"{job.get('id', f'Job {i+1}')} ({job.get('source_table', '?')} -> {job.get('target_table', '?')})"
                           for i, job in enumerate(lineage_jobs_config)]
            selected_job_index = st.sidebar.selectbox("Select Lineage Job", options=range(len(job_options)), format_func=lambda x: job_options[x])

            lineage_cfg = lineage_jobs_config[selected_job_index]
            table_a = lineage_cfg.get("source_table", "")
            table_b = lineage_cfg.get("target_table", "")
            expected_extra_in_b = set(lineage_cfg.get("expected_extras", []))
            ignored_source_columns = set(lineage_cfg.get("ignored_source_columns", []))
            column_mappings = lineage_cfg.get("column_mappings", {})
            st.sidebar.info(f"Viewing job: **{lineage_cfg.get('id', 'N/A')}**\n\n_{lineage_cfg.get('description', 'No description.')}_")

        else:
            st.sidebar.error("❌ Invalid JSON format. Expected a 'lineage_jobs' list.")
            st.stop()
    except json.JSONDecodeError:
        st.sidebar.error("❌ Invalid JSON file. Please upload a valid JSON.")
        st.stop()
else:
    st.sidebar.warning("⚠️ No mapping config uploaded. Using manual inputs.")
    st.sidebar.subheader("Manual Lineage Setup")
    table_a = st.sidebar.text_input("Source Table", "customer_abc")
    table_b = st.sidebar.text_input("Target Table", "customer_bcd")
    expected_extra_in_b = set(st.sidebar.text_area("Expected extra columns in Target (comma-separated)", "fax_customer").split(","))
    expected_extra_in_b.discard('') # Remove empty string if present
    ignored_source_columns = set(st.sidebar.text_area("Ignored Source Columns (comma-separated)", "").split(","))
    ignored_source_columns.discard('') # Remove empty string if present

    mapping_input = st.sidebar.text_area("Column Mappings (JSON format, e.g., {\"old_name\": \"new_name\"})", "{}")
    try:
        column_mappings = json.loads(mapping_input)
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON for column mappings. Please correct it.")
        column_mappings = {}


if not table_a or not table_b:
    st.warning("Please specify both Source and Target table names.")
    st.stop()

# ------------------------------------------------------
# Fetch Schema
# ------------------------------------------------------
st.subheader("📑 Schema & Connectivity")
conn_params = {
    "host": DB_HOST, "port": DB_PORT, "user": DB_USER, "password": DB_PASS, "dbname": DB_NAME
}

try:
    with st.spinner(f"Connecting to database and fetching schema for {table_a} and {table_b}..."):
        cols_a = get_table_columns(conn_params, table_a)
        cols_b = get_table_columns(conn_params, table_b)

    if not cols_a and not cols_b:
        st.warning(f"Could not retrieve columns for both '{table_a}' and '{table_b}'. Check table names and database connection.")
        st.stop()
    elif not cols_a:
        st.warning(f"Could not retrieve columns for source table '{table_a}'. Please check the table name.")
        st.stop()
    elif not cols_b:
        st.warning(f"Could not retrieve columns for target table '{table_b}'. Please check the table name.")
        st.stop()

    st.success("✅ Successfully connected and fetched schemas.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{table_a} Columns**")
        st.json(cols_a)
    with col2:
        st.markdown(f"**{table_b} Columns**")
        st.json(cols_b)

    # ------------------------------------------------------
    # Build & Show Graph
    # ------------------------------------------------------
    st.subheader("📊 Lineage Visualization")

    net, shared, only_in_source, only_in_target = build_lineage_graph(
        table_a, cols_a, table_b, cols_b, expected_extra_in_b, ignored_source_columns, column_mappings
    )

    # Save and render inside Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        html_content = open(tmp.name, "r", encoding="utf-8").read()
        components.html(html_content, height=600, scrolling=True)
    os.remove(tmp.name) # Clean up temp file

    # ------------------------------------------------------
    # Show Results
    # ------------------------------------------------------
    st.subheader("✅ Lineage Comparison Results")
    st.info("Hover over nodes and edges in the graph for more details!")

    st.markdown("---")
    st.markdown("**Column Categories**")
    st.write(f"**🟢 Mapped & Shared Columns:** {shared if shared else '—'}")
    st.write(f"**🟠 Columns only in Source ('{table_a}') (missing in Target):** {only_in_source if only_in_source else '—'}")
    st.write(f"**🔴 Unexpected Extra Columns in Target ('{table_b}'):** {[col for col in only_in_target if col not in expected_extra_in_b] if [col for col in only_in_target if col not in expected_extra_in_b] else '—'}")
    st.write(f"**⚪ Expected Extra Columns in Target ('{table_b}'):** {[col for col in only_in_target if col in expected_extra_in_b] if [col for col in only_in_target if col in expected_extra_in_b] else '—'}")
    st.write(f"**⚫ Ignored Source Columns in ('{table_a}'):** {list(ignored_source_columns.intersection(cols_a)) if ignored_source_columns.intersection(cols_a) else '—'}")
    st.markdown("---")


except psycopg2.Error as e:
    st.error(f"❌ Database connection error: {e}")
    st.info("Please check your `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, and `DB_NAME` environment variables or database accessibility.")
except Exception as e:
    st.error(f"❌ An unexpected error occurred: {e}")
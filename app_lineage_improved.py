# app_lineage_improved.py

import os
import json
import psycopg2
import streamlit as st
import tempfile
from pyvis.network import Network
from dotenv import load_dotenv
import streamlit.components.v1 as components

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
def get_table_columns(conn, table_name):
    """Fetch column names for a table from PostgreSQL"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (table_name,))
        return [row[0] for row in cur.fetchall()]

# ------------------------------------------------------
# Build Lineage Graph
# ------------------------------------------------------
def build_lineage_graph(table_a, cols_a, table_b, cols_b, expected_extra_in_b, column_mappings):
    net = Network(notebook=False, directed=True)
    net.barnes_hut()

    # Source & Target Nodes
    net.add_node(table_a, shape="box", color="lightblue")
    net.add_node(table_b, shape="box", color="lightgreen")

    # Remap A’s columns if mapping exists
    mapped_cols_a = [column_mappings.get(c, c) for c in cols_a]

    set_a = set(mapped_cols_a)
    set_b = set(cols_b)

    shared = sorted(list(set_a.intersection(set_b)))
    only_in_a = sorted(list(set_a - set_b))
    only_in_b = sorted(list(set_b - set_a))

    # Draw shared columns
    for col in shared:
        net.add_node(f"{table_a}.{col}", color="lightblue")
        net.add_node(f"{table_b}.{col}", color="lightgreen")
        net.add_edge(f"{table_a}.{col}", f"{table_b}.{col}", label="maps_to")

    # Draw only_in_a (no mapping found)
    for col in only_in_a:
        net.add_node(f"{table_a}.{col}", color="orange")
        net.add_edge(f"{table_a}.{col}", table_b, label="missing_in_B")

    # Draw only_in_b (extra in target)
    for col in only_in_b:
        color = "lightgrey" if col in expected_extra_in_b else "red"
        label = "expected_extra" if col in expected_extra_in_b else "unexpected_extra"
        net.add_node(f"{table_b}.{col}", color=color)
        net.add_edge(table_a, f"{table_b}.{col}", label=label)

    return net, shared, only_in_a, only_in_b

# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.set_page_config(page_title="Data Lineage Visualizer", layout="wide")
st.title("🔗 Data Lineage Visualizer (PostgreSQL)")

# Sidebar – Mapping JSON
st.sidebar.header("Mapping Config (JSON)")
mapping_file = st.sidebar.file_uploader("Upload mapping config (JSON)", type=["json"])

if mapping_file:
    config = json.load(mapping_file)
    lineage_cfg = config["lineage"][0]  # assume one mapping for now
    table_a = lineage_cfg["source_table"]
    table_b = lineage_cfg["target_table"]
    expected_extra_in_b = set(lineage_cfg.get("expected_extras", []))
    column_mappings = lineage_cfg.get("column_mappings", {})
    st.success(f"✅ Loaded mapping config for {table_a} → {table_b}")
else:
    st.warning("⚠️ No mapping config uploaded. Using defaults.")
    table_a = st.sidebar.text_input("Source Table", "customer_abc")
    table_b = st.sidebar.text_input("Target Table", "customer_bcd")
    expected_extra_in_b = set(st.sidebar.text_area("Expected extras (comma-separated)", "").split(","))
    column_mappings = {}

# ------------------------------------------------------
# Fetch Schema
# ------------------------------------------------------
try:
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, dbname=DB_NAME
    )

    cols_a = get_table_columns(conn, table_a)
    cols_b = get_table_columns(conn, table_b)
    conn.close()

    st.subheader("📑 Schema")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{table_a}**", cols_a)
    with col2:
        st.write(f"**{table_b}**", cols_b)

    # ------------------------------------------------------
    # Build & Show Graph
    # ------------------------------------------------------
    net, shared, only_in_a, only_in_b = build_lineage_graph(
        table_a, cols_a, table_b, cols_b, expected_extra_in_b, column_mappings
    )

    st.subheader("📊 Lineage Visualization")

    # Save and render inside Streamlit (Fix for .render error)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html_content = open(tmp.name, "r", encoding="utf-8").read()
        components.html(html_content, height=600, scrolling=True)

    # ------------------------------------------------------
    # Show Results
    # ------------------------------------------------------
    st.subheader("✅ Comparison Results")
    st.write("**Shared Columns:**", shared)
    st.write("**Only in Source:**", only_in_a)
    st.write("**Only in Target:**", only_in_b)

except Exception as e:
    st.error(f"❌ Error: {e}")

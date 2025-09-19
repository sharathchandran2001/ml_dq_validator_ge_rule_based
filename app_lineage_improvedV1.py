# app_lineage_pro_v3.py
# ------------------------------------------------------
# Streamlit: PostgreSQL Data Lineage Visualizer (flexible JSON + fixed pyvis options)
#
# ✅ Accepts BOTH JSON formats:
#   A) Rich/new:
#      {
#        "connection": {...},             # optional
#        "lineage": [{
#          "name": "Customer ABC -> BCD",
#          "source": {"schema":"public","table":"customer_abc"},
#          "target": {"schema":"public","table":"customer_bcd"},
#          "primary_key": {"source":["customer_id"], "target":["customer_id"]},
#          "filters": {"source_where":"", "target_where":""},
#          "expected_extras": ["fax_customer"],
#          "column_mappings": [
#             {"source":"first_name","target":"first_name","type":"copy"},
#             {"source":"phone","target":"phone_number","type":"rename"},
#             {"source":["addr1","addr2"],"target":"address","type":"concat","expression":"addr1 || ' ' || addr2"},
#             {"source":"email","target":"email","type":"transform","expression":"lower(email)"},
#             {"source": null, "target": "fax_customer", "type": "expected_extra"}
#          ]
#        }]}
#
#   B) Legacy/compact (like your file):
#      {
#        "lineage": [{
#          "source_table": "customer_abc",
#          "target_table": "customer_bcd",
#          "expected_extras": ["fax_customer"],
#          "column_mappings": {
#            "id":"id","first_name":"first_name","last_name":"last_name",
#            "email":"email","phone":"phone_number","dob":"date_of_birth"
#          }
#        }]}
#
# Run:
#   pip install streamlit psycopg2-binary pyvis python-dotenv
#   streamlit run app_lineage_pro_v3.py
# ------------------------------------------------------

import os
import json
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Set

import psycopg2
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pyvis.network import Network

# ---------------------------
# Env defaults (overrideable)
# ---------------------------
load_dotenv()
ENV_DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "admin"),
    "database": os.getenv("DB_NAME", "MA"),
    "schema": os.getenv("DB_SCHEMA", "public"),
}

# ---------------------------
# Streamlit page
# ---------------------------
st.set_page_config(page_title="Data Lineage Visualizer", layout="wide")
st.title("🔗 Data Lineage Visualizer (PostgreSQL)")

# ---------------------------
# Sidebar: config upload
# ---------------------------
st.sidebar.header("Mapping Config (JSON)")
cfg_file = st.sidebar.file_uploader("Upload mapping config (.json)", type=["json"])
use_manual = st.sidebar.checkbox("No JSON? Use manual inputs instead", value=False)

# ---------------------------
# Helpers
# ---------------------------
def parse_schema_table(value: str, default_schema: str) -> Tuple[str, str]:
    """Accepts 'table' or 'schema.table'; returns (schema, table)."""
    if "." in value:
        s, t = value.split(".", 1)
        return (s.strip() or default_schema), t.strip()
    return default_schema, value.strip()

def safe_list_str(lst: Optional[List[Optional[str]]]) -> List[str]:
    """Normalize lists, stripping whitespace and removing empties/None."""
    if not lst:
        return []
    return [x.strip() for x in lst if isinstance(x, str) and x.strip()]

def connect_pg(conn_dict: Dict[str, str]):
    return psycopg2.connect(
        host=conn_dict["host"],
        port=conn_dict["port"],
        user=conn_dict["user"],
        password=conn_dict["password"],
        dbname=conn_dict["database"],
    )

def get_columns(conn, schema: str, table: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (schema, table),
        )
        return [r[0] for r in cur.fetchall()]

def get_row_count(conn, schema: str, table: str, where: str = "") -> int:
    where_clause = f" WHERE {where} " if where and where.strip() else ""
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"{where_clause}')
        return cur.fetchone()[0]

def normalize_lineage_entries(raw_lineage: list, default_schema: str) -> list:
    """
    Accept both formats and normalize to:
    {
      "name": str|None,
      "source": {"schema":..., "table":...},
      "target": {"schema":..., "table":...},
      "primary_key": {...},
      "filters": {...},
      "expected_extras": [..],
      "column_mappings": [ {"source":[...], "target": "...", "type":"copy|rename|transform|concat|expected_extra", "expression":""}, ... ]
    }
    """
    def _safe_list_str(x):
        return [s.strip() for s in x if isinstance(s, str) and s.strip()]

    normalized = []
    for entry in raw_lineage:
        # New format
        if isinstance(entry, dict) and ("source" in entry or "target" in entry):
            e = {
                "name": entry.get("name"),
                "source": entry.get("source", {}) or {},
                "target": entry.get("target", {}) or {},
                "primary_key": entry.get("primary_key", {}) or {},
                "filters": entry.get("filters", {}) or {},
                "expected_extras": _safe_list_str(entry.get("expected_extras", [])),
                "column_mappings": entry.get("column_mappings", []) or []
            }
            e["source"]["schema"] = (e["source"].get("schema") or default_schema)
            e["target"]["schema"] = (e["target"].get("schema") or default_schema)
            normalized.append(e)
            continue

        # Legacy format
        src_tbl = entry.get("source_table", "")
        tgt_tbl = entry.get("target_table", "")
        exp_extras = entry.get("expected_extras", []) or []
        cm = entry.get("column_mappings", {}) or {}

        # Convert dict column_mappings -> list
        mappings_list = []
        if isinstance(cm, dict):
            for src, tgt in cm.items():
                if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
                    mappings_list.append({"source": src.strip(), "target": tgt.strip(), "type": "copy"})
        elif isinstance(cm, list):
            mappings_list = cm  # already a list
        else:
            mappings_list = []

        e = {
            "name": entry.get("name") or f"{default_schema}.{src_tbl} -> {default_schema}.{tgt_tbl}",
            "source": {"schema": default_schema, "table": src_tbl},
            "target": {"schema": default_schema, "table": tgt_tbl},
            "primary_key": entry.get("primary_key", {}) or {},
            "filters": entry.get("filters", {}) or {},
            "expected_extras": _safe_list_str(exp_extras),
            "column_mappings": mappings_list
        }
        normalized.append(e)

    # Normalize each mapping's source to a list
    for e in normalized:
        nmaps = []
        for m in e["column_mappings"]:
            src = m.get("source", None)
            if isinstance(src, list):
                src_list = safe_list_str(src)
            elif src is None:
                src_list = []
            else:
                src_list = safe_list_str([src])
            nmaps.append({
                "source": src_list,
                "target": (m.get("target") or "").strip(),
                "type": (m.get("type") or "copy").lower(),
                "expression": m.get("expression", "")
            })
        e["column_mappings"] = nmaps

    return normalized

# ---------------------------
# Visualization helpers
# ---------------------------
COLOR = {
    "table": "#DCEBFF",
    "source_col": "#E6F7FF",
    "target_col": "#F0FFF4",
    "edge_copy": "#2F855A",
    "edge_rename": "#3182CE",
    "edge_transform": "#805AD5",
    "edge_concat": "#DD6B20",
    "edge_expected_extra": "#718096",
    "edge_unexpected_extra": "#E53E3E",
    "edge_missing_in_target": "#E53E3E",
}

def add_legend(net: Network):
    net.add_node("LEGEND", label="Legend", shape="box", color="#FFF5F5")
    legend_items = [
        ("Copy/Rename", COLOR["edge_copy"]),
        ("Transform/Derived", COLOR["edge_transform"]),
        ("Concat", COLOR["edge_concat"]),
        ("Expected extra (target-only)", COLOR["edge_expected_extra"]),
        ("Unexpected extra (target-only)", COLOR["edge_unexpected_extra"]),
        ("Missing in target (source-only)", COLOR["edge_missing_in_target"]),
    ]
    for i, (label, color) in enumerate(legend_items, start=1):
        nid = f"LEGEND_{i}"
        net.add_node(nid, label=label, shape="dot", color=color)
        net.add_edge("LEGEND", nid, color=color)

def build_lineage_graph(
    source_fq: Tuple[str, str],
    target_fq: Tuple[str, str],
    cols_a: List[str],
    cols_b: List[str],
    expected_extras_b: Set[str],
    column_mappings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the pyvis graph and return it plus comparison details."""
    src_schema, table_a = source_fq
    tgt_schema, table_b = target_fq

    net = Network(height="650px", width="100%", directed=True, notebook=False)
    # IMPORTANT: pyvis expects a JSON string — not JS — so we use json.dumps
    net.set_options(json.dumps({
        "layout": {
            "hierarchical": {
                "enabled": True,
                "levelSeparation": 120,
                "direction": "LR",
                "sortMethod": "directed"
            }
        },
        "physics": {"enabled": False},
        "nodes": {"font": {"size": 14}}
    }))

    src_node = f"{src_schema}.{table_a}"
    tgt_node = f"{tgt_schema}.{table_b}"
    net.add_node(src_node, label=src_node, shape="box", color=COLOR["table"])
    net.add_node(tgt_node, label=tgt_node, shape="box", color=COLOR["table"])

    set_a = set(cols_a)
    set_b = set(cols_b)

    # Column nodes
    for c in cols_a:
        net.add_node(f"{src_node}.{c}", label=c, shape="box", color=COLOR["source_col"])
        net.add_edge(src_node, f"{src_node}.{c}", color="#A0AEC0")
    for c in cols_b:
        net.add_node(f"{tgt_node}.{c}", label=c, shape="box", color=COLOR["target_col"])
        net.add_edge(f"{tgt_node}.{c}", tgt_node, color="#A0AEC0")

    # Normalize mappings (already normalized by normalize_lineage_entries)
    mapped_targets = set()
    mapped_sources = set()
    for m in (column_mappings or []):
        tgt = m["target"]
        mtype = m["type"]
        expr = m["expression"]
        srcs = m["source"]
        if not tgt:
            continue
        if tgt not in set_b:
            # mapping points to a non-existent target col
            net.add_edge(src_node, tgt_node, color=COLOR["edge_unexpected_extra"],
                         label=f"mapping to missing target '{tgt}'", arrows="to", dashes=True)
            continue

        mapped_targets.add(tgt)
        for s in srcs:
            mapped_sources.add(s)

        edge_color = {
            "copy": COLOR["edge_copy"],
            "rename": COLOR["edge_rename"],
            "transform": COLOR["edge_transform"],
            "concat": COLOR["edge_concat"],
            "derived": COLOR["edge_transform"],
            "expected_extra": COLOR["edge_expected_extra"],
        }.get(mtype, COLOR["edge_copy"])

        label = mtype + (f" ({expr})" if expr else "")
        if srcs:
            for s in srcs:
                if s in set_a:
                    net.add_edge(f"{src_node}.{s}", f"{tgt_node}.{tgt}", color=edge_color, label=label, arrows="to")
                else:
                    net.add_edge(src_node, f"{tgt_node}.{tgt}", color=COLOR["edge_unexpected_extra"],
                                 label=f"{mtype}: missing src {s}", arrows="to", dashes=True)
        else:
            # constant/derived without specific source column
            net.add_edge(src_node, f"{tgt_node}.{tgt}", color=edge_color, label=label, arrows="to", dashes=(mtype != "expected_extra"))

    # Physical-only comparisons (by raw column names)
    shared_physical = sorted(list(set_a.intersection(set_b)))
    only_in_a_physical = sorted(list(set_a - set_b))
    only_in_b_physical = sorted(list(set_b - set_a))

    only_in_b_expected = sorted([c for c in only_in_b_physical if c in expected_extras_b])
    only_in_b_unexpected = sorted([c for c in only_in_b_physical if c not in expected_extras_b])

    # Source-only columns without mapping => likely missing in target
    for c in only_in_a_physical:
        if c not in mapped_sources:
            net.add_edge(f"{src_node}.{c}", tgt_node, color=COLOR["edge_missing_in_target"],
                         label="missing_in_target", arrows="to", dashes=True)

    # Target-only columns without mapping
    for c in only_in_b_physical:
        if c in mapped_targets:
            continue
        if c in expected_extras_b:
            net.add_edge(src_node, f"{tgt_node}.{c}", color=COLOR["edge_expected_extra"],
                         label="expected_extra", arrows="to")
        else:
            net.add_edge(src_node, f"{tgt_node}.{c}", color=COLOR["edge_unexpected_extra"],
                         label="unexpected_extra", arrows="to", dashes=True)

    add_legend(net)

    return {
        "net": net,
        "shared_physical": shared_physical,
        "only_in_a_physical": only_in_a_physical,
        "only_in_b_expected": only_in_b_expected,
        "only_in_b_unexpected": only_in_b_unexpected,
        "mapped_targets": sorted(list(mapped_targets)),
    }

# ---------------------------
# Load / Merge config
# ---------------------------
effective = {
    "connection": dict(ENV_DB),  # start from env
    "lineage": []
}

if cfg_file and not use_manual:
    try:
        cfg_json = json.load(cfg_file)
        # Merge connection (optional)
        if isinstance(cfg_json.get("connection"), dict):
            effective["connection"].update({k: v for k, v in cfg_json["connection"].items() if v not in [None, ""]})

        # Normalize lineage entries from either format
        raw_lineage = cfg_json.get("lineage")
        if isinstance(raw_lineage, list) and raw_lineage:
            effective["lineage"] = normalize_lineage_entries(raw_lineage, default_schema=effective["connection"]["schema"])
        else:
            st.warning("`lineage` is missing or empty in the JSON. Switch to manual inputs below.")
            use_manual = True
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        use_manual = True

# Manual fallback
if use_manual:
    st.sidebar.header("Manual Setup")
    manual_src = st.sidebar.text_input("Source table (schema.table or table)", "customer_abc")
    manual_tgt = st.sidebar.text_input("Target table (schema.table or table)", "customer_bcd")
    manual_expected = st.sidebar.text_input("Expected extras in target (comma-separated)", "fax_customer")

    src_schema, src_table = parse_schema_table(manual_src, ENV_DB["schema"])
    tgt_schema, tgt_table = parse_schema_table(manual_tgt, ENV_DB["schema"])
    expected_extras = set([x.strip() for x in manual_expected.split(",") if x.strip()])

    # Minimal lineage entry
    lineage_entry = {
        "name": f"{src_schema}.{src_table} -> {tgt_schema}.{tgt_table}",
        "source": {"schema": src_schema, "table": src_table},
        "target": {"schema": tgt_schema, "table": tgt_table},
        "expected_extras": sorted(list(expected_extras)),
        "column_mappings": []  # add detailed mappings in JSON for richer edges
    }
    effective["lineage"] = [lineage_entry]

# UI: choose which lineage entry to display (if multiple)
if len(effective["lineage"]) > 1:
    names = [
        x.get("name") or f"{x['source'].get('schema','public')}.{x['source']['table']} -> "
                         f"{x['target'].get('schema','public')}.{x['target']['table']}"
        for x in effective["lineage"]
    ]
    idx = st.sidebar.selectbox("Choose a lineage to visualize", list(range(len(names))), format_func=lambda i: names[i])
else:
    idx = 0

if not effective["lineage"]:
    st.stop()

entry = effective["lineage"][idx]

# Resolve connection (sidebar allows quick override)
st.sidebar.header("Connection Override (optional)")
ov_host = st.sidebar.text_input("Host", value=effective["connection"]["host"])
ov_port = st.sidebar.text_input("Port", value=effective["connection"]["port"])
ov_user = st.sidebar.text_input("User", value=effective["connection"]["user"])
ov_pass = st.sidebar.text_input("Password", type="password", value=effective["connection"]["password"])
ov_db   = st.sidebar.text_input("Database", value=effective["connection"]["database"])
ov_schema_default = st.sidebar.text_input("Default Schema", value=effective["connection"]["schema"])

CONN = {
    "host": ov_host, "port": ov_port, "user": ov_user,
    "password": ov_pass, "database": ov_db, "schema": ov_schema_default
}

# Resolve source/target FQNs
src_schema = (entry.get("source", {}) or {}).get("schema", CONN["schema"])
src_table = (entry.get("source", {}) or {}).get("table", "")
tgt_schema = (entry.get("target", {}) or {}).get("schema", CONN["schema"])
tgt_table = (entry.get("target", {}) or {}).get("table", "")

if not src_table or not tgt_table:
    st.error("Source/Target table names are required (check your JSON or manual inputs).")
    st.stop()

expected_extras_b = set(safe_list_str(entry.get("expected_extras")))
column_mappings = entry.get("column_mappings", [])
filters = entry.get("filters", {}) or {}
src_where = (filters.get("source_where") or "").strip()
tgt_where = (filters.get("target_where") or "").strip()

# Preview effective config
with st.expander("🔧 Effective config (read-only preview)"):
    st.json({
        "connection": {**CONN, "password": "******"},
        "lineage_entry": {
            "name": entry.get("name"),
            "source": {"schema": src_schema, "table": src_table},
            "target": {"schema": tgt_schema, "table": tgt_table},
            "filters": {"source_where": src_where, "target_where": tgt_where},
            "expected_extras": sorted(list(expected_extras_b)),
            "column_mappings": column_mappings
        }
    })

# ---------------------------
# Main: connect, fetch, build
# ---------------------------
try:
    conn = connect_pg(CONN)
    with conn:
        cols_a = get_columns(conn, src_schema, src_table)
        cols_b = get_columns(conn, tgt_schema, tgt_table)
        row_a = get_row_count(conn, src_schema, src_table, where=src_where)
        row_b = get_row_count(conn, tgt_schema, tgt_table, where=tgt_where)
except Exception as e:
    st.error(f"❌ Connection or metadata fetch error: {e}")
    st.stop()

# Metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.metric(f"Rows in {src_schema}.{src_table}", row_a)
with m2:
    st.metric(f"Rows in {tgt_schema}.{tgt_table}", row_b)
with m3:
    st.metric("Shared physical columns", len(set(cols_a).intersection(cols_b)))

# Show schemas
st.subheader("📑 Physical Schemas")
c1, c2 = st.columns(2)
with c1:
    st.write(f"**{src_schema}.{src_table}**")
    st.code(cols_a, language="text")
with c2:
    st.write(f"**{tgt_schema}.{tgt_table}**")
    st.code(cols_b, language="text")

# Build & render graph
graph_out = build_lineage_graph(
    (src_schema, src_table),
    (tgt_schema, tgt_table),
    cols_a, cols_b,
    expected_extras_b,
    column_mappings
)

st.subheader("🕸️ Lineage Visualization")
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
    graph_out["net"].save_graph(tmp.name)
    with open(tmp.name, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=700, scrolling=True)

# Results panels
st.subheader("✅ Comparison Results")
left, right = st.columns(2)
with left:
    st.write("**Shared physical columns**", graph_out["shared_physical"] or "—")
    st.write("**Only in source (physical)**", graph_out["only_in_a_physical"] or "—")
with right:
    st.write("**Only in target (expected extras)**", graph_out["only_in_b_expected"] or "—")
    st.write("**Only in target (UNEXPECTED extras)**", graph_out["only_in_b_unexpected"] or "—")

# Mapping coverage
mapped_targets = set(graph_out["mapped_targets"])
unmapped_targets = set(cols_b) - mapped_targets
with st.expander("📈 Mapping coverage"):
    st.write(f"Mapped target columns: {sorted(list(mapped_targets)) or '—'}")
    st.write(f"Unmapped target columns: {sorted(list(unmapped_targets)) or '—'}")
    st.caption("Tip: Add entries to `column_mappings` to explicitly document/visualize how each target column is produced.")

st.success("Analysis complete. Adjust the JSON config or manual inputs in the sidebar to refine lineage.")

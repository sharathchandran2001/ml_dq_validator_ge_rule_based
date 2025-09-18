# pip install sqlalchemy psycopg2-binary pandas great_expectations
import os
from typing import List
import pandas as pd
from sqlalchemy import create_engine, text

# ----------------------------
# 1) CONNECTION
# ----------------------------
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DB", "MA")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "admin")
SCHEMA  = "public"
T_A     = "customer_abc"
T_B     = "customer_bcd"
EXTRA_IN_B = "fax_customer"  # the known extra column in B

engine = create_engine(
    f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

def get_columns(table: str) -> List[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
    """)
    with engine.begin() as conn:
        rows = conn.execute(q, {"schema": SCHEMA, "table": table}).fetchall()
    return [r[0] for r in rows]

def get_row_count(table: str) -> int:
    with engine.begin() as conn:
        return conn.execute(text(f'SELECT COUNT(*) FROM "{SCHEMA}"."{table}"')).scalar()

def fetch_df(table: str, cols: List[str]) -> pd.DataFrame:
    col_list = ','.join([f'"{c}"' for c in cols])
    return pd.read_sql(f'SELECT {col_list} FROM "{SCHEMA}"."{table}"', engine)

# ----------------------------
# 2) SCHEMA COMPARISON
# ----------------------------
cols_a = get_columns(T_A)
cols_b = get_columns(T_B)

set_a, set_b = set(cols_a), set(cols_b)
shared = [c for c in cols_a if c in set_b]            # preserve A’s order for readability
only_in_a = list(set_a - set_b)
only_in_b = list(set_b - set_a)

print("=== SCHEMA DIFF ===")
print(f"{T_A} columns: {cols_a}")
print(f"{T_B} columns: {cols_b}")
print(f"Shared columns: {shared}")
print(f"Only in {T_A}: {only_in_a}")
print(f"Only in {T_B}: {only_in_b}")
print()

# Basic rule we expect:
# - All columns of A should be present in B EXCEPT the known extra column in B.
expected_only_in_b = {EXTRA_IN_B}
schema_ok = (len(only_in_a) == 0) and (set(only_in_b) == expected_only_in_b)
print(f"Schema conformity (B == A + '{EXTRA_IN_B}'): {schema_ok}")
print()

# ----------------------------
# 3) ROW COUNT COMPARISON
# ----------------------------
count_a = get_row_count(T_A)
count_b = get_row_count(T_B)
print("=== ROW COUNTS ===")
print(f"{T_A}: {count_a}")
print(f"{T_B}: {count_b}")
rowcount_ok = (count_a == count_b)
print(f"Row counts equal: {rowcount_ok}")
print()

# ----------------------------
# 4) VALUE-LEVEL COMPARISON ON SHARED COLUMNS
# Strategy:
# - Prefer a PK called 'customer_id' if it exists in both tables.
# - If no single PK, create a row hash across shared columns and compare.
# ----------------------------
PKEY = "customer_id" if ("customer_id" in shared) else None

df_a = fetch_df(T_A, shared)
df_b = fetch_df(T_B, shared)

if PKEY and PKEY in df_a.columns and PKEY in df_b.columns:
    df_a_keyed = df_a.set_index(PKEY).sort_index()
    df_b_keyed = df_b.set_index(PKEY).sort_index()

    # Align on keys
    missing_in_b = df_a_keyed.index.difference(df_b_keyed.index)
    missing_in_a = df_b_keyed.index.difference(df_a_keyed.index)

    mismatched_rows = []
    common_index = df_a_keyed.index.intersection(df_b_keyed.index)
    # Compare row-by-row for shared columns (excluding the PK itself)
    compare_cols = [c for c in shared if c != PKEY]
    diffs = (df_a_keyed.loc[common_index, compare_cols] !=
             df_b_keyed.loc[common_index, compare_cols])
    # Any row with at least one differing column:
    mismatched_idx = diffs.any(axis=1)
    if mismatched_idx.any():
        mismatched_rows = common_index[mismatched_idx].tolist()

    print("=== VALUE COMPARISON (by primary key) ===")
    print(f"Rows present in {T_A} but missing in {T_B}: {len(missing_in_b)}")
    print(f"Rows present in {T_B} but missing in {T_A}: {len(missing_in_a)}")
    print(f"Rows with differing values on shared columns: {len(mismatched_rows)}")

    if len(mismatched_rows) > 0:
        sample = mismatched_rows[:10]
        print(f"Sample differing keys: {sample}")
else:
    # Fallback: hash of shared columns to compare sets (works when no PK)
    def row_hash(df: pd.DataFrame) -> pd.Series:
        return (df.astype(str)
                  .apply(lambda r: "§".join(r.values), axis=1)
                  .pipe(lambda s: s.str.hash()))  # pandas hash

    hash_a = row_hash(df_a)
    hash_b = row_hash(df_b)

    # Count matching hashes; report differences
    # Using value_counts because duplicates can exist
    vc_a = hash_a.value_counts()
    vc_b = hash_b.value_counts()

    # Hashes in A not in B or with different frequencies
    missing_or_diff = []
    for h, cnt in vc_a.items():
        if h not in vc_b or vc_b[h] != cnt:
            missing_or_diff.append((h, cnt, vc_b.get(h, 0)))

    print("=== VALUE COMPARISON (by hash of shared columns) ===")
    print(f"Distinct row-signatures in {T_A}: {len(vc_a)}")
    print(f"Distinct row-signatures in {T_B}: {len(vc_b)}")
    print(f"Row-signatures that differ (A vs B): {len(missing_or_diff)}")
    if missing_or_diff:
        print("Sample differences (hash, count_in_A, count_in_B):")
        print(missing_or_diff[:10])

print()

# ----------------------------
# 5) BASIC DATA QUALITY CHECKS (nulls in key columns, fax format presence)
# ----------------------------
critical_cols = [c for c in shared if c != EXTRA_IN_B]
with engine.begin() as conn:
    for col in critical_cols:
        nulls_a = conn.execute(text(
            f'SELECT COUNT(*) FROM "{SCHEMA}"."{T_A}" WHERE "{col}" IS NULL'
        )).scalar()
        nulls_b = conn.execute(text(
            f'SELECT COUNT(*) FROM "{SCHEMA}"."{T_B}" WHERE "{col}" IS NULL'
        )).scalar

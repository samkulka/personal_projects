#!/usr/bin/env python3
"""
q2_migrate_lite.py
A self-contained demo of a legacy -> cloud (SQLite) data migration with transforms and QA.
No external libraries required (uses Python stdlib).
"""

import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Iterable, Tuple

DATA_DIR = Path("data_legacy")
TARGET_DB = Path("target.db")

# -----------------------------
# 0) Seed some legacy CSV data
# -----------------------------
def seed_legacy_data():
    DATA_DIR.mkdir(exist_ok=True)

    customers = [
        {"id": "1001", "name": "Alice Johnson", "dob": "1988-05-04", "address": "101 Main St"},
        {"id": "1002", "name": "Bob Smith",    "dob": "1979-11-20", "address": "202 Oak Ave"},
        {"id": "1003", "name": "Carol White",  "dob": "1990-03-15", "address": "303 Pine Rd"},
    ]
    accounts = [
        {"id": "5001", "customer_id": "1001", "balance": "1234.56", "open_date": "2018-02-01"},
        {"id": "5002", "customer_id": "1001", "balance": "250.00",  "open_date": "2020-01-15"},
        {"id": "5003", "customer_id": "1002", "balance": "999.99",  "open_date": "2017-08-30"},
    ]
    transactions = [
        {"id": "900001", "account_id": "5001", "type": "DEP", "amount": "100.00", "timestamp": "2024-07-03T10:20:00"},
        {"id": "900002", "account_id": "5001", "type": "WDR", "amount": "25.21",  "timestamp": "2024-07-05T14:33:00"},
        {"id": "900003", "account_id": "5002", "type": "FEE", "amount": "5.00",   "timestamp": "2024-07-10T09:00:00"},
        {"id": "900004", "account_id": "5003", "type": "DEP", "amount": "300.00", "timestamp": "2024-07-11T16:45:00"},
    ]

    def write_csv(path: Path, rows: List[Dict[str, str]]):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(DATA_DIR / "customers.csv", customers)
    write_csv(DATA_DIR / "accounts.csv", accounts)
    write_csv(DATA_DIR / "transactions.csv", transactions)


# -----------------------------------------
# 1) Minimal config (mappings & validations)
# -----------------------------------------
MAPPINGS = {
    "customers": {
        "source": str(DATA_DIR / "customers.csv"),
        "target": "dim_customer",
        "key": "customer_id",
        "columns": {
            "id": "customer_id",
            "name": "full_name",
            "dob": "birth_date",
            "address": "address",
        },
        "casts": {"birth_date": "date"},
    },
    "accounts": {
        "source": str(DATA_DIR / "accounts.csv"),
        "target": "dim_account",
        "key": "account_id",
        "columns": {
            "id": "account_id",
            "customer_id": "customer_id",
            "balance": "balance_cents",
            "open_date": "open_date",
        },
        "transforms": {
            # compute cents from legacy float string
            "balance_cents": lambda r: int(round(float(r["balance"]) * 100)),
        },
        "casts": {"open_date": "date"},
    },
    "transactions": {
        "source": str(DATA_DIR / "transactions.csv"),
        "target": "fct_transaction",
        "key": "txn_id",
        "columns": {
            "id": "txn_id",
            "account_id": "account_id",
            "type": "txn_type",
            "amount": "amount_cents",
            "timestamp": "txn_ts",
        },
        "enums": {
            "txn_type": {"DEP": "DEPOSIT", "WDR": "WITHDRAWAL", "FEE": "FEE"},
        },
        "transforms": {
            "amount_cents": lambda r: int(round(float(r["amount"]) * 100)),
        },
        "casts": {"txn_ts": "timestamptz"},
    },
}

VALIDATION_RULES = [
    # row counts
    {"name": "customers_count", "type": "row_count_equals_source", "table": "dim_customer", "source": "customers"},
    {"name": "accounts_count",  "type": "row_count_equals_source", "table": "dim_account",  "source": "accounts"},
    {"name": "transactions_count", "type": "row_count_equals_source", "table": "fct_transaction", "source": "transactions"},
    # sums
    {"name": "sum_balance_cents", "type": "target_sql",
     "sql": "SELECT COALESCE(SUM(balance_cents),0) FROM dim_account", "expect": None},  # we’ll compute expected
    {"name": "sum_amount_cents", "type": "target_sql",
     "sql": "SELECT COALESCE(SUM(amount_cents),0) FROM fct_transaction", "expect": None},
    # uniqueness & referential integrity
    {"name": "unique_customer_id", "type": "unique", "table": "dim_customer", "column": "customer_id"},
    {"name": "unique_account_id",  "type": "unique", "table": "dim_account",  "column": "account_id"},
    {"name": "unique_txn_id",      "type": "unique", "table": "fct_transaction", "column": "txn_id"},
    {"name": "accounts_fk_customer", "type": "fk_check",
     "child_table": "dim_account", "child_col": "customer_id",
     "parent_table": "dim_customer", "parent_col": "customer_id"},
    {"name": "txn_fk_account", "type": "fk_check",
     "child_table": "fct_transaction", "child_col": "account_id",
     "parent_table": "dim_account", "parent_col": "account_id"},
]


# -----------------
# 2) Extract (CSV)
# -----------------
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


# --------------------------
# 3) Transform (mappings)
# --------------------------
def cast_value(val: str, cast_type: str):
    if val is None:
        return None
    if cast_type == "date":
        # standardize YYYY-MM-DD
        return datetime.strptime(val[:10], "%Y-%m-%d").date().isoformat()
    if cast_type == "timestamptz":
        # store ISO string; SQLite will keep TEXT
        # ensure format like 2024-07-03T10:20:00Z or local; here just normalize
        return val.replace(" ", "T")
    return val

def apply_mappings(rows: List[Dict[str, str]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out_rows = []
    colmap: Dict[str, str] = cfg.get("columns", {})
    enums: Dict[str, Dict[str, str]] = cfg.get("enums", {})
    transforms: Dict[str, Any] = cfg.get("transforms", {})
    casts: Dict[str, str] = cfg.get("casts", {})

    for r in rows:
        out = {}
        # column renames
        for src, tgt in colmap.items():
            out[tgt] = r.get(src)

        # enum mapping
        for col, mapping in enums.items():
            if col in out and out[col] in mapping:
                out[col] = mapping[out[col]]

        # transforms (callables)
        for col, func in transforms.items():
            out[col] = func(r)

        # casts
        for col, ctype in casts.items():
            if col in out:
                out[col] = cast_value(out[col], ctype)

        out_rows.append(out)
    return out_rows


# --------------------------
# 4) Load (SQLite target)
# --------------------------
DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dim_customer (
  customer_id  INTEGER PRIMARY KEY,
  full_name    TEXT NOT NULL,
  birth_date   TEXT,
  address      TEXT
);

CREATE TABLE IF NOT EXISTS dim_account (
  account_id     INTEGER PRIMARY KEY,
  customer_id    INTEGER NOT NULL,
  balance_cents  INTEGER NOT NULL,
  open_date      TEXT,
  FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id)
);

CREATE TABLE IF NOT EXISTS fct_transaction (
  txn_id        INTEGER PRIMARY KEY,
  account_id    INTEGER NOT NULL,
  txn_type      TEXT CHECK (txn_type IN ('DEPOSIT','WITHDRAWAL','FEE')),
  amount_cents  INTEGER NOT NULL,
  txn_ts        TEXT,
  FOREIGN KEY (account_id) REFERENCES dim_account(account_id)
);
"""

def connect_db() -> sqlite3.Connection:
    if TARGET_DB.exists():
        TARGET_DB.unlink()  # clean start for demo
    conn = sqlite3.connect(TARGET_DB.as_posix())
    conn.executescript(DDL)
    return conn

def upsert_naive(conn: sqlite3.Connection, table: str, rows: List[Dict[str, Any]], key_col: str):
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))

    # delete existing keys (naive idempotency)
    key_vals = [r[key_col] for r in rows]
    q_marks = ",".join(["?"] * len(key_vals))
    conn.execute(f"DELETE FROM {table} WHERE {key_col} IN ({q_marks})", key_vals)

    # insert
    conn.executemany(
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
        [tuple(r[c] for c in cols) for r in rows]
    )
    conn.commit()


# --------------------------
# 5) Validation / QA checks
# --------------------------
def count_csv_rows(name: str) -> int:
    path = MAPPINGS[name]["source"]
    return sum(1 for _ in open(path, encoding="utf-8")) - 1  # minus header

def run_validations(conn: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    results: List[Tuple[str, str, str]] = []

    # Precompute expected sums from legacy to make the test meaningful
    acc_rows = read_csv_rows(MAPPINGS["accounts"]["source"])
    txn_rows = read_csv_rows(MAPPINGS["transactions"]["source"])
    expected_balance_cents = sum(int(round(float(r["balance"]) * 100)) for r in acc_rows)
    expected_amount_cents = sum(int(round(float(r["amount"]) * 100)) for r in txn_rows)

    for rule in VALIDATION_RULES:
        name = rule["name"]
        rtype = rule["type"]

        if rtype == "row_count_equals_source":
            table = rule["table"]
            source_name = rule["source"]
            target_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            source_count = count_csv_rows(source_name)
            ok = target_count == source_count
            results.append((name, "PASS" if ok else "FAIL", f"src={source_count}, tgt={target_count}"))

        elif rtype == "target_sql":
            sql = rule["sql"]
            val = conn.execute(sql).fetchone()[0]
            expect = expected_balance_cents if "balance" in name else expected_amount_cents
            ok = int(val or 0) == int(expect)
            results.append((name, "PASS" if ok else "FAIL", f"expect={expect}, got={val}"))

        elif rtype == "unique":
            table, col = rule["table"], rule["column"]
            dups = conn.execute(
                f"SELECT {col}, COUNT(*) c FROM {table} GROUP BY 1 HAVING COUNT(*)>1"
            ).fetchall()
            ok = len(dups) == 0
            results.append((name, "PASS" if ok else "FAIL", f"dups={len(dups)}"))

        elif rtype == "fk_check":
            ct, cc, pt, pc = rule["child_table"], rule["child_col"], rule["parent_table"], rule["parent_col"]
            missing = conn.execute(f"""
                SELECT c.{cc} FROM {ct} c
                LEFT JOIN {pt} p ON c.{cc} = p.{pc}
                WHERE p.{pc} IS NULL
            """).fetchall()
            ok = len(missing) == 0
            results.append((name, "PASS" if ok else "FAIL", f"orphans={len(missing)}"))

        else:
            results.append((name, "INFO", "unknown rule"))

    return results

def print_report(rows: Iterable[Tuple[str, str, str]]):
    print("\nValidation Report")
    print("-" * 72)
    print(f"{'Rule':30} | {'Status':7} | Detail")
    print("-" * 72)
    for name, status, detail in rows:
        print(f"{name:30} | {status:7} | {detail}")
    print("-" * 72)


# --------------------------
# 6) Orchestrate end to end
# --------------------------
def run():
    print("Seeding legacy CSVs...")
    seed_legacy_data()

    print("Connecting to target SQLite and creating schema...")
    conn = connect_db()

    for src_name, cfg in MAPPINGS.items():
        print(f"\nProcessing table: {src_name} -> {cfg['target']}")
        legacy_rows = read_csv_rows(cfg["source"])
        print(f"  Extracted rows: {len(legacy_rows)}")

        transformed = apply_mappings(legacy_rows, cfg)
        print(f"  Transformed rows: {len(transformed)}")

        upsert_naive(conn, cfg["target"], transformed, cfg["key"])
        tgt_count = conn.execute(f"SELECT COUNT(*) FROM {cfg['target']}").fetchone()[0]
        print(f"  Loaded rows into {cfg['target']}: {tgt_count}")

    results = run_validations(conn)
    print_report(results)

    print(f"\nDone. Target DB written to: {TARGET_DB.resolve()}")


if __name__ == "__main__":
    run()
import duckdb

def ensure_schema(conn, ddl_sql):
    conn.execute(ddl_sql)

def upsert(conn, table, rows, key):
    if not rows:
        return
    cols = list(rows[0].keys())
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM (SELECT 1 AS {cols[0]} WHERE FALSE)")
    # naive: DELETE+INSERT (fine for demo)
    keys = ",".join(str(r[key]) for r in rows)
    conn.execute(f"DELETE FROM {table} WHERE {key} IN ({keys})")
    placeholders = ",".join(["?"] * len(cols))
    conn.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
                     [tuple(r[c] for c in cols) for r in rows])

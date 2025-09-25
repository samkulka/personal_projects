from tabulate import tabulate

def run_rules(conn, rules):
    results = []
    for rule in rules:
        if rule["type"] == "unique":
            col = rule["column"]
            tbl = rule["table"]
            dup = conn.execute(
                f"SELECT {col}, COUNT(*) c FROM {tbl} GROUP BY 1 HAVING COUNT(*)>1"
            ).fetchall()
            ok = len(dup) == 0
            results.append((rule["name"], "PASS" if ok else "FAIL", "" if ok else f"{len(dup)} dups"))
        elif rule["type"] == "row_count_equals_source":
            # compare count(target) vs len(source CSV)
            tbl = rule["table"]
            tgt = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            # you can stash source counts earlier; here we just display target
            results.append((rule["name"], "INFO", f"target_count={tgt}"))
        else:
            # sql vs sql comparison
            src = conn.execute(rule["sql_source"]).fetchone()[0]
            tgt = conn.execute(rule["sql_target"]).fetchone()[0]
            results.append((rule["name"], "PASS" if src == tgt else "FAIL", f"src={src} tgt={tgt}"))
    print(tabulate(results, headers=["Rule", "Status", "Detail"]))

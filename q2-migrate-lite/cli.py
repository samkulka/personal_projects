import yaml, duckdb
from etl.extract import read_csv
from etl.transform import apply_mappings
from etl.load import upsert
from etl.validate import run_rules

def run_all():
    cfg = yaml.safe_load(open("config/mappings.yml"))
    rules = yaml.safe_load(open("config/validation.yml"))["rules"]
    conn = duckdb.connect("target.duckdb")
    # create schema from file once:
    conn.execute(open("sql/target_schema.sql").read())

    for src_table, tcfg in cfg["tables"].items():
        rows = read_csv(f"data_legacy/{tcfg['source']}")
        rows_t = apply_mappings(rows, tcfg)
        upsert(conn, tcfg["target"], rows_t, tcfg["key"])

    run_rules(conn, rules)

if __name__ == "__main__":
    run_all()

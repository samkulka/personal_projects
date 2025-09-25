import csv
from pathlib import Path

def read_csv(path, dtypes=None):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    # light dtype coercion if desired
    return rows

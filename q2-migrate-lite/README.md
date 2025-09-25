# Summary 
This script simulates a real data migration. It creates a small “legacy” banking dataset (CSV files), transforms it with Python (renames columns, converts balances to cents, maps enum codes, normalizes dates), loads it into a “cloud” target database (SQLite for the demo), then runs validation checks to prove the migration is correct.

# What it does step by step
1. Seed legacy data
    Generates three CSVs in data_legacy/: customers.csv, accounts.csv, transactions.csv.

2. Define mappings
    In code, sets rules for how to reshape the data:
        Column renames (e.g., id → customer_id)
        Type conversions (dates, timestamps)
        Business transforms (e.g., balance 12.34 → balance_cents 1234)
        Enum mapping (e.g., DEP → DEPOSIT)

3. Create target schema
    Builds three target tables in target.db (SQLite): dim_customer, dim_account, fct_transaction, with primary keys and foreign keys.

4. Extract → Transform → Load
    Reads the CSVs
    Applies the mappings and transforms
    Performs a simple idempotent load (delete by key, then insert) into the target tables

5. Validate the migration
    Runs a lightweight QA suite and prints a report:
        Row counts match source
        Sum of balances and transaction amounts match
        Uniqueness checks on keys
        Referential integrity checks (no orphan accounts or transactions)

6. Output
    Writes target.db and prints a formatted “Validation Report” with PASS or FAIL for each rule.

# How to run

python q2_migrate_lite.py


# Expected console output includes:
    Extracted, transformed, and loaded row counts per table
    A validation report, for example:

Validation Report
----------------------------------------------------------------
Rule                          | Status  | Detail
customers_count               | PASS    | src=3, tgt=3
accounts_count                | PASS    | src=3, tgt=3
transactions_count            | PASS    | src=4, tgt=4
sum_balance_cents             | PASS    | expect=248455, got=248455
sum_amount_cents              | PASS    | expect=43021,  got=43021
unique_customer_id            | PASS    | dups=0
unique_account_id             | PASS    | dups=0
unique_txn_id                 | PASS    | dups=0
accounts_fk_customer          | PASS    | orphans=0
txn_fk_account                | PASS    | orphans=0
----------------------------------------------------------------
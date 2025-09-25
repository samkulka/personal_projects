CREATE TABLE dim_customer (
  customer_id BIGINT PRIMARY KEY,
  full_name   TEXT NOT NULL,
  birth_date  DATE,
  address     TEXT
);

CREATE TABLE dim_account (
  account_id    BIGINT PRIMARY KEY,
  customer_id   BIGINT NOT NULL REFERENCES dim_customer(customer_id),
  balance_cents BIGINT NOT NULL,
  open_date     DATE
);

CREATE TABLE fct_transaction (
  txn_id       BIGINT PRIMARY KEY,
  account_id   BIGINT NOT NULL REFERENCES dim_account(account_id),
  txn_type     TEXT CHECK (txn_type IN ('DEPOSIT','WITHDRAWAL','FEE')),
  amount_cents BIGINT NOT NULL,
  txn_ts       TIMESTAMPTZ
);

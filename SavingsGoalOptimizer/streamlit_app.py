# streamlit_app.py
import json
from datetime import datetime, date
import pandas as pd
import streamlit as st

from datetime import date

# define DATE_FMT locally so we don't rely on optimizer
DATE_FMT = "%Y-%m-%d"


# Import your optimizer logic
import optimizer as opt

st.set_page_config(page_title="Savings Goal Optimizer", page_icon="💸", layout="wide")

def parse_cfg_from_editor(as_of_str, monthly_income, fixed_expenses, flexible_budget, emergency_buffer, goals_df):
    """Build the dict expected by opt.build_plan() using the optimizer.Goal class."""
    as_of = datetime.strptime(as_of_str, DATE_FMT).date()
    goals = []
    for _, row in goals_df.iterrows():
        # Skip completely blank rows
        if not str(row.get("name", "")).strip():
            continue
        goals.append(
            opt.Goal(
                name=str(row["name"]),
                target=float(row["target"]),
                current=float(row.get("current", 0) or 0),
                deadline=datetime.strptime(str(row["deadline"]), opt.DATE_FMT).date(),
                priority=float(row.get("priority", 1.0) or 1.0),
            )
        )
    return {
        "as_of": as_of,
        "goals": goals,
        "monthly_income": float(monthly_income),
        "fixed_expenses": float(fixed_expenses),
        "flexible_budget": float(flexible_budget),
        "emergency_buffer": float(emergency_buffer),
    }

st.title("💸 Savings Goal Optimizer")
st.caption("Deadline-aware allocation of your monthly surplus across multiple goals.")

# --- Sidebar controls
st.sidebar.header("Global Settings")
as_of = st.sidebar.text_input("As of (YYYY-MM-DD)", value=date.today().strftime(DATE_FMT))
monthly_income = st.sidebar.number_input("Monthly income", min_value=0.0, value=6000.0, step=100.0)
fixed_expenses = st.sidebar.number_input("Fixed expenses", min_value=0.0, value=3200.0, step=50.0)
flexible_budget = st.sidebar.number_input("Flexible budget", min_value=0.0, value=1000.0, step=50.0)
emergency_buffer = st.sidebar.number_input("Emergency buffer", min_value=0.0, value=200.0, step=50.0)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Load goals from JSON (optional)", type=["json"])

# --- Default or uploaded goals
if "goals_df" not in st.session_state:
    default_goals = [
        {"name": "Emergency Fund", "target": 15000, "current": 6000, "deadline": "2026-03-01", "priority": 1.2},
        {"name": "Hawaii Trip", "target": 4000, "current": 500, "deadline": "2026-01-15", "priority": 0.8},
        {"name": "Debt Paydown", "target": 5000, "current": 1500, "deadline": "2026-06-01", "priority": 1.0},
    ]
    st.session_state.goals_df = pd.DataFrame(default_goals)

if uploaded is not None:
    try:
        cfg = json.load(uploaded)
        st.session_state.goals_df = pd.DataFrame(cfg["goals"])
        if cfg.get("as_of"):
            as_of = cfg["as_of"]
        monthly_income = float(cfg["monthly_income"])
        fixed_expenses = float(cfg["fixed_expenses"])
        flexible_budget = float(cfg["flexible_budget"])
        emergency_buffer = float(cfg.get("emergency_buffer", 0))
        st.sidebar.success("Config loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not parse JSON: {e}")

st.subheader("Goals")
edited_df = st.data_editor(
    st.session_state.goals_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "name": "Name",
        "target": st.column_config.NumberColumn("Target", min_value=0.0, step=100.0),
        "current": st.column_config.NumberColumn("Current", min_value=0.0, step=50.0),
        "deadline": st.column_config.TextColumn("Deadline (YYYY-MM-DD)", help="e.g., 2026-03-01"),
        "priority": st.column_config.NumberColumn("Priority", min_value=0.0, step=0.1, help="Relative weight"),
    },
)
st.session_state.goals_df = edited_df

# --- Build plan
try:
    cfg = parse_cfg_from_editor(as_of, monthly_income, fixed_expenses, flexible_budget, emergency_buffer, edited_df)
    plan = opt.build_plan(cfg)
except Exception as e:
    st.error(f"Problem building plan: {e}")
    st.stop()

# --- Summary
cols = st.columns(5)
cols[0].metric("Surplus", f"${plan['budget_surplus']:.2f}")
cols[1].metric("Required this month", f"${plan['total_required']:.2f}")
cols[2].metric("Allocated", f"${plan['total_allocated']:.2f}")
cols[3].metric("On track", "Yes" if plan["on_track"] else "No")
cols[4].metric("Shortfall", f"${plan['shortfall']:.2f}")

# --- Per-goal table
table_rows = []
for g in plan["goals"]:
    name = g["goal"]
    table_rows.append({
        "Goal": name,
        "Required / Month": g["required_monthly"],
        "Allocated / Month": plan["allocations"].get(name, 0.0),
        "Remaining": g["remaining"],
        "Months Left": g["months_left"],
        "Deadline": g["deadline"],
        "Priority": g["priority"],
    })
table_df = pd.DataFrame(table_rows)

st.subheader("Per-goal allocations")
st.dataframe(table_df, use_container_width=True)

# --- Chart
st.subheader("Allocation chart")
chart_df = table_df[["Goal", "Allocated / Month"]].set_index("Goal")
st.bar_chart(chart_df)

# --- Download buttons
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download plan JSON",
        data=json.dumps(plan, indent=2),
        file_name="plan_output.json",
        mime="application/json",
    )
with col2:
    out_csv = table_df.to_csv(index=False)
    st.download_button(
        "Download allocations CSV",
        data=out_csv,
        file_name="allocations.csv",
        mime="text/csv",
    )

# --- Save config
st.markdown("### Save current config")
if st.button("Download config JSON for later"):
    config_like = {
        "as_of": as_of,
        "monthly_income": monthly_income,
        "fixed_expenses": fixed_expenses,
        "flexible_budget": flexible_budget,
        "emergency_buffer": emergency_buffer,
        "goals": edited_df.to_dict(orient="records"),
    }
    st.download_button(
        "Save config.json",
        data=json.dumps(config_like, indent=2),
        file_name="config.json",
        mime="application/json",
    )

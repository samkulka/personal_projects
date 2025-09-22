#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Optional

DATE_FMT = "%Y-%m-%d"

@dataclass
class Goal:
    name: str
    target: float
    current: float
    deadline: date
    priority: float = 1.0
    @property
    def remaining(self) -> float:
        return max(0.0, self.target - self.current)
    def months_left(self, as_of: date) -> int:
        days = (self.deadline - as_of).days
        return max(1, math.ceil(days / 30))
    def required_monthly(self, as_of: date) -> float:
        ml = self.months_left(as_of)
        return 0.0 if self.remaining <= 0 else self.remaining / ml

def parse_config(path: str, override_as_of: Optional[str]) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    if override_as_of:
        cfg["as_of"] = override_as_of
    as_of = datetime.strptime(cfg.get("as_of") or datetime.today().strftime(DATE_FMT), DATE_FMT).date()
    goals: List[Goal] = []
    for g in cfg["goals"]:
        goals.append(Goal(
            name=g["name"],
            target=float(g["target"]),
            current=float(g.get("current", 0)),
            deadline=datetime.strptime(g["deadline"], DATE_FMT).date(),
            priority=float(g.get("priority", 1.0)),
        ))
    return {
        "as_of": as_of,
        "goals": goals,
        "monthly_income": float(cfg["monthly_income"]),
        "fixed_expenses": float(cfg["fixed_expenses"]),
        "flexible_budget": float(cfg["flexible_budget"]),
        "emergency_buffer": float(cfg.get("emergency_buffer", 0)),
    }

def compute_surplus(monthly_income: float, fixed_expenses: float, flexible_budget: float, emergency_buffer: float) -> float:
    return max(0.0, monthly_income - fixed_expenses - flexible_budget - emergency_buffer)

def proportional_allocation(goals: List[Goal], as_of: date, budget: float) -> Dict[str, float]:
    req = {g.name: g.required_monthly(as_of) for g in goals}
    weights = {g.name: req[g.name] * max(0.0, g.priority) for g in goals}
    allocations = {g.name: 0.0 for g in goals}
    sum_required = sum(req.values())
    if budget <= 0 or sum(weights.values()) == 0:
        return allocations
    if budget <= sum_required:
        remaining_budget = budget
        remaining_names = set(req.keys())
        while remaining_budget > 1e-8 and remaining_names:
            total_w = sum(weights[n] for n in remaining_names)
            if total_w <= 0:
                break
            give = {n: remaining_budget * (weights[n] / total_w) for n in remaining_names}
            saturated = []
            for n in list(remaining_names):
                room = req[n] - allocations[n]
                add = min(room, give[n])
                allocations[n] += add
                remaining_budget -= add
                if allocations[n] >= req[n] - 1e-9:
                    saturated.append(n)
            for n in saturated:
                remaining_names.remove(n)
        return allocations
    for n in req:
        allocations[n] = req[n]
    remaining_budget = budget - sum_required
    residual_need = {g.name: max(0.0, g.remaining - allocations[g.name]) for g in goals}
    order = sorted(goals, key=lambda g: (weights[g.name], -g.months_left(as_of)), reverse=True)
    i = 0
    while remaining_budget > 1e-8 and any(residual_need[n] > 1e-8 for n in residual_need):
        g = order[i % len(order)]
        n = g.name
        if residual_need[n] <= 1e-8:
            i += 1; continue
        add = min(remaining_budget, residual_need[n] / g.months_left(as_of))
        allocations[n] += add
        residual_need[n] -= add
        remaining_budget -= add
        i += 1
    return allocations

def build_plan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    as_of = cfg["as_of"]
    goals: List[Goal] = cfg["goals"]
    budget = compute_surplus(cfg["monthly_income"], cfg["fixed_expenses"], cfg["flexible_budget"], cfg["emergency_buffer"])
    req_table = [{
        "goal": g.name,
        "remaining": round(g.remaining, 2),
        "months_left": g.months_left(as_of),
        "required_monthly": round(g.required_monthly(as_of), 2),
        "priority": g.priority,
        "deadline": g.deadline.strftime(DATE_FMT),
    } for g in goals]
    allocations = proportional_allocation(goals, as_of, budget)
    total_required = round(sum(r["required_monthly"] for r in req_table), 2)
    total_alloc = round(sum(allocations.values()), 2)
    on_track = budget >= total_required
    return {
        "as_of": as_of.strftime(DATE_FMT),
        "budget_surplus": round(budget, 2),
        "total_required": total_required,
        "on_track": on_track,
        "goals": req_table,
        "allocations": {k: round(v, 2) for k, v in allocations.items()},
        "total_allocated": total_alloc,
        "shortfall": round(max(0.0, total_required - budget), 2),
    }

def print_plan(plan: Dict[str, Any]) -> None:
    print(f"As of: {plan['as_of']}")
    print(f"Monthly surplus: ${plan['budget_surplus']:.2f}")
    print(f"Required to stay on track: ${plan['total_required']:.2f}")
    print("Status:", "✅ On track" if plan["on_track"] else f"⚠️ Shortfall of ${plan['shortfall']:.2f}")
    print("\nPer-goal details:")
    print(f"{'Goal':30} {'Req/Mo':>10} {'Alloc':>10} {'Months':>8} {'Deadline':>12} {'Priority':>9}")
    for g in plan["goals"]:
        name = g["goal"]
        print(f"{name:30} "
              f"{g['required_monthly']:>10.2f} "
              f"{plan['allocations'].get(name, 0.0):>10.2f} "
              f"{g['months_left']:>8d} "
              f"{g['deadline']:>12} "
              f"{g['priority']:>9.2f}")
    print(f"\nTotal allocated: ${plan['total_allocated']:.2f}")

def export_csv(plan: Dict[str, Any], path: str) -> None:
    import csv
    rows = []
    for g in plan["goals"]:
        name = g["goal"]
        rows.append({
            "as_of": plan["as_of"],
            "goal": name,
            "required_monthly": g["required_monthly"],
            "allocation": plan["allocations"].get(name, 0.0),
            "months_left": g["months_left"],
            "deadline": g["deadline"],
            "priority": g["priority"],
            "on_track": plan["on_track"],
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Savings Goal Optimizer")
    ap.add_argument("--config", required=True, help="Path to config JSON")
    ap.add_argument("--as_of", help="Override as_of date YYYY-MM-DD")
    ap.add_argument("--export", help="CSV path to export allocations")
    args = ap.parse_args()
    cfg = parse_config(args.config, args.as_of)
    plan = build_plan(cfg)
    print_plan(plan)
    if args.export: export_csv(plan, args.export)

if __name__ == "__main__":
    main()

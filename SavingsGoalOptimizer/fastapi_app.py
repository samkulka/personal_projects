from typing import List, Optional, Dict
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
import optimizer as opt

DATE_FMT = "%Y-%m-%d"

class GoalIn(BaseModel):
    name: str
    target: float
    current: float = 0
    deadline: str
    priority: float = 1.0
    @validator("deadline")
    def check_deadline(cls, v):
        datetime.strptime(v, DATE_FMT)
        return v

class PlanRequest(BaseModel):
    as_of: Optional[str] = None
    monthly_income: float = Field(ge=0)
    fixed_expenses: float = Field(ge=0)
    flexible_budget: float = Field(ge=0)
    emergency_buffer: float = 0
    goals: List[GoalIn]

class PlanResponse(BaseModel):
    plan: Dict

app = FastAPI(title="Savings Goal Optimizer API", version="0.1.0")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/optimize", response_model=PlanResponse)
def optimize(req: PlanRequest):
    as_of = datetime.strptime(req.as_of or datetime.today().strftime(DATE_FMT), DATE_FMT).date()
    goals = [opt.Goal(
        name=g.name, target=g.target, current=g.current,
        deadline=datetime.strptime(g.deadline, DATE_FMT).date(),
        priority=g.priority
    ) for g in req.goals]
    cfg = {
        "as_of": as_of,
        "goals": goals,
        "monthly_income": req.monthly_income,
        "fixed_expenses": req.fixed_expenses,
        "flexible_budget": req.flexible_budget,
        "emergency_buffer": req.emergency_buffer,
    }
    plan = opt.build_plan(cfg)
    return {"plan": plan}
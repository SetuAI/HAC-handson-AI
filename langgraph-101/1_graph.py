'''
Scenario: 
A portfolio arrives. Three independent risk checks run on it — concentration, liquidity, credit. 
Their results get combined into one score. If the score is above the desk's tolerance, 
it goes to the investment committee. 
If not, it's approved automatically. Everything that happened is logged.
That's it. That's a real workflow you'd find on a risk desk, in about 200 lines.


    START
      |
      v
 [load_portfolio]
      |
  +---+---+--------------+          THREE CHECKS IN PARALLEL
  v       v              v
[concentration] [liquidity] [credit]
  |       |              |
  +---+---+--------------+
      v
  [aggregate]                       fan-in: combine into one score
      |
   +--+--+
 high   ok                          conditional edge
   |     |
   v     v
[escalate] [auto_approve]
   |     |
   +--+--+
      v
     END

'''

from typing import Annotated, TypedDict, Optional
from operator import add # add reducer
from langgraph.graph import StateGraph, START , END

class PortfolioState(TypedDict):
    portfolio_id : str
    escalation_threshold : float
    holdings : Optional[list[dict]]
    risk_score : Optional[float]
    decision : Optional[str]
    findings : Annotated[list[dict],add]
    audit_log : Annotated[list[str],add]
    
HOLDINGS = [
    {"name": "Reliance Industries", "weight_pct": 34.0, "liquidity": "high", "rating": "AAA"},
    {"name": "HDFC Bank",           "weight_pct": 22.0, "liquidity": "high", "rating": "AAA"},
    {"name": "Unlisted RE Fund II", "weight_pct": 18.0, "liquidity": "low",  "rating": "BBB"},
    {"name": "Infosys",             "weight_pct": 14.0, "liquidity": "high", "rating": "AA"},
    {"name": "SME Credit Pool",     "weight_pct": 12.0, "liquidity": "low",  "rating": "BB"},
]

# first node : 
def load_portfolio(state:PortfolioState):
    "fetch the holdings"
    return { # return the state with the holdings and an audit log entry
        "holdings" : HOLDINGS,
        "audit_log" : [f"Loaded : {len(HOLDINGS)}"]
    }
    
# 3 parallel checks will work on this state : 
def concentration_check(state: PortfolioState):
    """Flag any single holding above 30%."""
    heavy = [h for h in state["holdings"] if h["weight_pct"] > 30.0]
    return {
        "findings": [{
            "check": "concentration",
            "severity": 8.0 if heavy else 2.0,
            "detail": f"{len(heavy)} holding(s) over 30%: {[h['name'] for h in heavy]}"
                      if heavy else "No holding exceeds 30%",
        }],
        "audit_log": ["concentration_check complete"],
    }


def liquidity_check(state: PortfolioState):
    """Flag more than 25% in low-liquidity assets."""
    illiquid = sum(h["weight_pct"] for h in state["holdings"] if h["liquidity"] == "low")
    return {
        "findings": [{
            "check": "liquidity",
            "severity": 7.0 if illiquid > 25.0 else 3.0,
            "detail": f"{illiquid:.1f}% is low-liquidity",
        }],
        "audit_log": ["liquidity_check complete"],
    }


def credit_check(state: PortfolioState):
    """Flag sub-investment-grade holdings."""
    weak = [h for h in state["holdings"] if h["rating"] in ("BB", "B", "CCC")]
    return {
        "findings": [{
            "check": "credit",
            "severity": 6.0 if weak else 1.0,
            "detail": f"Sub-investment-grade: {[h['name'] for h in weak]}"
                      if weak else "All investment grade",
        }],
        "audit_log": ["credit_check complete"],
    }


# aggregate node : fan in 

def aggregate(state: PortfolioState):
    "fan -in , average of 3 scores into one risk score"
    severities = [f["severity"] for f in state["findings"]]
    score = sum(severities) / len(severities) 
    return {
        "risk score" : round(score,2),
        "audit_log": [f"Aggregated {len(severities)} findings -> {score:.2f}"],
    }
    
def escalate(state: PortfolioState):
    return {
        "decision" : "Escalated to investment committee",
        "audit_log" : [f"Escalated for review"]
    }

def auto_approve(state: PortfolioState):
    return {
        "decision" : "Auto approved",
        "audit_log" : [f"Auto approved , no need for review committee"]
    }
    
# router function : conditional edge logic
# this is applied after the aggregate node, to decide which path to take next

def risk_router(state: PortfolioState):
    "route to escalate or auto_approve based on risk score"
    # if the risk score is above the threshold, go to escalate node, else auto approve
    if state["risk_score"] > state["escalation_threshold"]:
        return "escalate"
    else:
        return "auto_approve"
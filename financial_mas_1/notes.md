mkdir financial_mas
mkdir financial_mas/core
mkdir financial_mas/tools
mkdir financial_mas/agents
mkdir financial_mas/graph
mkdir financial_mas/api
mkdir financial_mas/ui

touch financial_mas/__init__.py
touch financial_mas/core/__init__.py
touch financial_mas/tools/__init__.py
touch financial_mas/agents/__init__.py
touch financial_mas/graph/__init__.py
touch financial_mas/api/__init__.py
touch financial_mas/ui/__init__.py

touch financial_mas/.env
touch financial_mas/main.py
touch financial_mas/requirements.txt
touch financial_mas/core/config.py
touch financial_mas/tools/financial_tools.py
touch financial_mas/agents/supervisor.py
touch financial_mas/agents/market_data_agent.py
touch financial_mas/agents/risk_analyst_agent.py
touch financial_mas/agents/fundamental_analyst_agent.py
touch financial_mas/agents/report_writer_agent.py
touch financial_mas/graph/pipeline.py
touch financial_mas/api/server.py
touch financial_mas/ui/dashboard.py

## File 1 of 11 — `core/config.py`

**What it does:**
This is the foundation of the entire system. It defines two things:

1. **`Settings`** — reads your `OPENAI_API_KEY` from the `.env` file so no API key is ever hardcoded anywhere
2. **`FinancialAnalysisState`** — this is the most important concept in LangGraph. It's the **shared memory dictionary** (the "baton") that gets passed between every agent. Each agent reads from it and writes their results back into it. Think of it like a shared Google Doc — every agent opens it, adds their section, and passes it to the next person.
3. **`AgentNames`** — just string constants so we never have typos like `"market_data_agnt"` buried somewhere

**Do we run this file?** No. It's a configuration module — other files import from it. You'll never run it directly.

## File 2 of 11 — `tools/financial_tools.py`

**What it does:**
These are the **"hands"** of the agents. Agents are smart decision-makers but they can't do anything on their own — they need tools to interact with the outside world.

This file defines 4 tools:

1. **`fetch_stock_price`** — calls yfinance to get current price, volume, 52-week range, 30 days of historical prices
2. **`fetch_company_info`** — gets company name, sector, industry, description
3. **`calculate_risk_metrics`** — pure Python math: calculates volatility, beta, Sharpe ratio, Value at Risk, max drawdown
4. **`fetch_fundamental_data`** — gets P/E ratio, EPS, revenue growth, margins, debt-to-equity

The `@tool` decorator is what makes these LangChain-compatible — it's what allows an LLM agent to "see" these functions and decide to call them.

**Do we run this file?** No. It's a module — agents import from it.

## File 3 of 11 — `agents/supervisor.py`

**What it does:**
The Supervisor is the **brain/orchestrator** of the entire system. It doesn't do any analysis itself — it just decides  **who runs next** .

Think of it like a senior manager at an investment bank. When a client request comes in, the manager doesn't do the work — they look at what's been done so far and say "okay, go talk to the market data team first, then the risk team, then the fundamentals team, then get the report written."

Two functions in this file:

1. **`supervisor_node`** — looks at the current state and sets `next_agent` to whoever should run next. Simple waterfall logic: no market data? → go get it. No risk metrics? → calculate them. And so on.
2. **`route_to_next_agent`** — LangGraph calls this after every single agent finishes. It reads `next_agent` from state and returns the node name to jump to. This is what LangGraph calls a  **conditional edge** .

**Do we run this file?** No. The graph imports these two functions.

## File 4 of 11 — `agents/market_data_agent.py`

**What it does:**
This is the **first specialist agent** that runs in the pipeline. Its only job is to fetch raw data — price, volume, company info — and store it in the shared state.

This file is the best one to teach **how a tool-calling agent works** because the pattern is very clear:

```
LLM sees the tools available
    → decides to call fetch_stock_price("AAPL")
    → tool runs, returns {price: 185.20, ...}
    → result is fed back to the LLM
    → LLM decides to call fetch_company_info("AAPL")
    → tool runs, returns {sector: "Technology", ...}
    → LLM sees both results, writes a short summary
    → agent stores the data in state and exits
```

Two key LangChain concepts introduced here:

* **`bind_tools()`** — registers the available tools with the LLM so it knows what it can call
* **`ToolMessage`** — how you feed a tool's result back into the conversation so the LLM can read it

**Do we run this file?** No. The graph imports `market_data_agent_node`.

## File 5 of 11 — `agents/risk_analyst_agent.py`

**What it does:**
This is the  **second specialist agent** . It calculates and — more importantly — **interprets** the risk metrics.

This is where the real value of using an LLM agent becomes clear. Without the LLM, you'd just get raw numbers like `annualized_volatility_pct: 45.2`. With the LLM agent, you get:

> *"TSLA's 45% annualized volatility is roughly 2.5x the S&P 500's typical 18% — this classifies it as HIGH risk, suitable only for investors with aggressive risk tolerance"*

That translation from number → human insight is exactly what the LLM adds on top of the tool.

One important difference from the Market Data Agent —  **this agent uses the context from the previous agent** . It injects the current price and market cap from `state["market_data"]` into its prompt. This is the key MAS pattern:  **agents share context through shared state** .

**Do we run this file?** No. The graph imports `risk_analyst_agent_node`.

## File 6 of 11 — `agents/fundamental_analyst_agent.py`

**What it does:**
This is the  **third specialist agent** . It evaluates whether the stock is  **financially healthy and priced fairly** .

Two types of analysis:

1. **Valuation** — is the stock cheap or expensive? (P/E, PEG, Price/Book)
2. **Financial health** — is the business strong? (margins, debt, cash flow, ROE)

This agent also demonstrates something important for teaching — it injects context from **both** previous agents, not just one. It knows the current price (from Market Data Agent) AND the risk level (from Risk Analyst Agent). This shows how each agent in a MAS builds on the work of all prior agents.

The pattern is identical to the Risk Analyst — `bind_tools` → ReAct loop → interpret → parse insights → return state update.

**Do we run this file?** No. The graph imports `fundamental_analyst_agent_node`.

## File 7 of 11 — `agents/report_writer_agent.py`

**What it does:**
This is the **final specialist agent** — the one that synthesizes everything into a polished investment brief.

Three things make this agent different from all the others:

1. **No tool calls** — it doesn't call any tools. It only reads the state and writes. Pure LLM synthesis.
2. **Sees the complete state** — it's the most context-rich agent. It reads market data, risk metrics, risk insights, fundamental data, and fundamental insights all at once and formats them into one structured prompt before calling the LLM.
3. **Higher temperature (0.4)** — the analyst agents use 0.1 (near-deterministic) for consistent numbers. The report writer uses 0.4 so the prose reads naturally, not like a robot.

It also extracts a structured `recommendation` field (BUY/HOLD/SELL + confidence) from the report text using regex, so the API and UI can display it as a badge without parsing the full report.

**Do we run this file?** No. The graph imports `report_writer_agent_node`.

## File 8 of 11 — `graph/pipeline.py`

**What it does:**
This is the  **most important file in the entire project** . This is where LangGraph actually comes to life.

Everything we've built so far — the state, the tools, the 5 agents — none of it is connected yet. This file connects it all by building a  **directed graph** :

* **Nodes** = the agent functions
* **Edges** = connections between agents (who can call whom)
* **Conditional edges** = smart routing via the supervisor

The graph we're building looks exactly like this:

```
START → supervisor → market_data_agent → supervisor
                   → risk_analyst_agent → supervisor
                   → fundamental_analyst_agent → supervisor
                   → report_writer_agent → supervisor → END
```

Notice the supervisor is in the middle of everything. Every agent flows back to the supervisor before the next one runs. This is the  **hub-and-spoke pattern** .

Three key LangGraph methods to understand:

* **`add_node`** — registers an agent function as a node
* **`add_edge`** — hardwired connection (always goes here)
* **`add_conditional_edges`** — smart routing (goes here based on state)

This file also has `run_financial_analysis()` and `stream_financial_analysis()` — the two functions the API and UI will call.

**Do we run this file?** No directly, but this is the file that `main.py`, `api/server.py`, and `ui/dashboard.py` all import from.

## File 9 of 11 — `api/server.py`

**What it does:**
This is the  **FastAPI REST gateway** . It wraps the entire LangGraph pipeline behind clean HTTP endpoints so any client — the Streamlit UI, a mobile app, a Postman request, another service — can trigger an analysis without knowing anything about LangGraph or agents.

Three endpoints:

1. **`POST /analyze`** — runs the full pipeline, waits for completion, returns structured JSON with the report and all metrics
2. **`GET /analyze/stream`** — same pipeline but streams results back as  **Server-Sent Events (SSE)** , one event per agent completing. This is what the UI uses so users see progress in real time instead of a blank screen for 60 seconds
3. **`GET /metrics/{ticker}`** — fast endpoint (2-5 seconds) that calls yfinance directly with  **no LLM involved** . Useful for quick data checks

Two things to highlight for teaching:

* **Pydantic models** (`AnalysisRequest`, `AnalysisResponse`) — FastAPI uses these to automatically validate requests and generate the Swagger docs at `/docs`
* **`/docs`** — once the server is running, go to `http://localhost:8000/docs` and you get a fully interactive API explorer. Great for live demos.

**Do we run this file?** Yes — but via uvicorn, not directly. Command is at the bottom.

to execute :

cd financial_mas
uvicorn api.server:app --reload --port 8000

Go to your browser and open:

```
http://localhost:8000/docs
```

You will see the Swagger UI. Here is exactly what to do for each endpoint:

---

**1. Health check (simplest — test this first)**

* Click `GET /health`
* Click **Try it out**
* Click **Execute**
* Should return `{"status": "healthy", ...}`

---

**2. Raw metrics (no LLM, fast — test this second)**

* Click `GET /metrics/{ticker}`
* Click **Try it out**
* In the `ticker` field type: `AAPL`
* Click **Execute**
* Should return price, risk, fundamentals in 3-5 seconds

---

**3. Full AI analysis (uses OpenAI — test this last)**

* Click `POST /analyze`
* Click **Try it out**
* Replace the request body with:

json

```json
{
"ticker":"AAPL",
"analysis_request":"Provide a comprehensive investment analysis with buy/hold/sell recommendation"
}
```

* Click **Execute**
* This takes **30-60 seconds** — all 4 agents are running
* Returns the full report + recommendation

---

Start with `/health` first, then `/metrics/AAPL`, then `/analyze`
-------------------------------

## File 10 of 11 — `ui/dashboard.py`

**What it does:**
This is the  **Streamlit interactive dashboard** . It gives a visual interface to the entire pipeline so your class can see everything happening in real time instead of reading raw JSON.

Key things it does:

1. **Input panel** — enter any ticker, quick buttons for AAPL/MSFT/TSLA etc, custom analysis request
2. **Real-time progress** — as each agent completes, its status updates live on screen. Users see the pipeline executing step by step
3. **Results tabs** — four tabs after analysis completes:
   * 📝 Full investment report (Markdown rendered)
   * ⚠️ Risk metrics with all calculated numbers
   * 📈 Fundamental metrics laid out cleanly
   * 🔍 Agent execution timeline (the audit trail)

**Do we run this file?** Yes — with `streamlit run`, in a **new terminal** while the uvicorn server is either running or stopped. The Streamlit UI talks directly to the pipeline, not through FastAPI.

## File 11 of 11 — `main.py`

**What it does:**
This is the  **command-line entry point** . It lets you run the full pipeline directly from the terminal without needing the FastAPI server or Streamlit UI.

This is the simplest way to test the pipeline during development — just run it and watch all 4 agents execute in the terminal with clean formatted output.

It is also great for teaching because the terminal output shows the entire pipeline executing sequentially — you can see each agent's output printed one after another.

**Do we run this file?** Yes — directly with `python main.py`

**python main.py --ticker TSLA --request "Focus on EV market growth"**

if shoes open api key not found then : export OPENAI_API_KEY="sk-your-actual-key-here"

and then execute the above command

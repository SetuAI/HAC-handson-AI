"""
graph/pipeline.py — LangGraph Multi-Agent Pipeline

This is where all agents get wired together into an executable graph.

CORE CONCEPTS:

NODES
    Every agent function becomes a node.
    A node takes the full state as input and returns a partial update.

EDGES
    Normal edge:      A → B  (always go from A to B)
    Conditional edge: A → ?  (call a function to decide where to go)

STATE FLOW
    t0: state = {ticker: "AAPL", market_data: {}, ...}  ← initial
    t1: supervisor    → sets next_agent = "market_data_agent"
    t2: market_data   → fills market_data, company_name
    t3: supervisor    → sets next_agent = "risk_analyst_agent"
    t4: risk_analyst  → fills risk_metrics, risk_insights
    t5: supervisor    → sets next_agent = "fundamental_analyst_agent"
    t6: fundamental   → fills fundamental_data, fundamental_insights
    t7: supervisor    → sets next_agent = "report_writer_agent"
    t8: report_writer → fills final_report, recommendation
    t9: supervisor    → sees analysis_complete=True → next_agent = "END"
    t10: graph terminates, returns final state
"""

import logging

from langgraph.graph import StateGraph, START, END
# StateGraph — main class for building agent graphs
# START      — built-in constant: the entry point of the graph
# END        — built-in constant: terminates the graph

from core.config import FinancialAnalysisState, AgentNames

from agents.supervisor import supervisor_node, route_to_next_agent
from agents.market_data_agent import market_data_agent_node
from agents.risk_analyst_agent import risk_analyst_agent_node
from agents.fundamental_analyst_agent import fundamental_analyst_agent_node
from agents.report_writer_agent import report_writer_agent_node

logger = logging.getLogger(__name__)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_financial_analysis_graph():
    """
    Construct and compile the LangGraph multi-agent pipeline.

    Steps:
        1. Create StateGraph with our state schema
        2. Add all agent nodes
        3. Set the entry point (START → supervisor)
        4. Add return edges (every agent → supervisor)
        5. Add conditional edges (supervisor → next agent)
        6. Compile

    Returns:
        A compiled graph ready to run with .invoke() or .stream()
    """
    logger.info("[GRAPH] Building graph...")

    # Step 1 — Create the StateGraph
    # Passing FinancialAnalysisState tells LangGraph:
    #   - what fields exist
    #   - how to merge updates (operator.add for list fields)
    graph = StateGraph(FinancialAnalysisState)

    # Step 2 — Add nodes
    # add_node(name, function)
    # The name must match what route_to_next_agent() returns
    graph.add_node(AgentNames.SUPERVISOR,          supervisor_node)
    graph.add_node(AgentNames.MARKET_DATA_AGENT,   market_data_agent_node)
    graph.add_node(AgentNames.RISK_ANALYST_AGENT,  risk_analyst_agent_node)
    graph.add_node(AgentNames.FUNDAMENTAL_ANALYST, fundamental_analyst_agent_node)
    graph.add_node(AgentNames.REPORT_WRITER,       report_writer_agent_node)

    # Step 3 — Entry point
    # The graph always starts at the supervisor.
    # The supervisor checks state and routes to whoever should go first.
    graph.add_edge(START, AgentNames.SUPERVISOR)

    # Step 4 — Return edges (agent → supervisor)
    # After EVERY specialist agent finishes, control returns to the supervisor.
    # The supervisor then decides what happens next.
    # This is the hub-and-spoke pattern.
    graph.add_edge(AgentNames.MARKET_DATA_AGENT,   AgentNames.SUPERVISOR)
    graph.add_edge(AgentNames.RISK_ANALYST_AGENT,  AgentNames.SUPERVISOR)
    graph.add_edge(AgentNames.FUNDAMENTAL_ANALYST, AgentNames.SUPERVISOR)
    graph.add_edge(AgentNames.REPORT_WRITER,       AgentNames.SUPERVISOR)

    # Step 5 — Conditional edges (supervisor → next agent)
    # add_conditional_edges(source, condition_fn, mapping)
    #
    # After supervisor runs, LangGraph calls route_to_next_agent(state).
    # That function returns a string. The mapping dict translates that
    # string into the node to visit next.
    #
    # "__end__" is LangGraph's internal name for the END constant.
    graph.add_conditional_edges(
        AgentNames.SUPERVISOR,       # source node
        route_to_next_agent,         # function that reads state and returns a string
        {
            AgentNames.MARKET_DATA_AGENT   : AgentNames.MARKET_DATA_AGENT,
            AgentNames.RISK_ANALYST_AGENT  : AgentNames.RISK_ANALYST_AGENT,
            AgentNames.FUNDAMENTAL_ANALYST : AgentNames.FUNDAMENTAL_ANALYST,
            AgentNames.REPORT_WRITER       : AgentNames.REPORT_WRITER,
            "__end__"                      : END,
        }
    )

    # Step 6 — Compile
    # Validates the graph structure and returns an executable object.
    # After compile() you can call:
    #   graph.invoke(state)   → synchronous, returns final state
    #   graph.stream(state)   → yields each node's output as it completes
    compiled = graph.compile()

    logger.info("[GRAPH] Graph compiled successfully")
    return compiled


# =============================================================================
# PIPELINE RUNNER — synchronous
# =============================================================================

def run_financial_analysis(
    ticker: str,
    analysis_request: str = "Provide a comprehensive investment analysis",
) -> dict:
    """
    Run the full pipeline synchronously for a given ticker.

    Blocks until all agents complete (~30-60 seconds).
    Returns the final state with all outputs filled in.

    Args:
        ticker:           stock symbol, e.g. "AAPL"
        analysis_request: optional custom instruction

    Returns:
        Final state dict containing:
            market_data, risk_metrics, risk_insights,
            fundamental_data, fundamental_insights,
            final_report, recommendation, agent_steps
    """
    logger.info(f"[PIPELINE] Starting analysis for {ticker}")

    graph = build_financial_analysis_graph()

    # Initial state — only input fields are set.
    # All output fields start empty. Agents fill them in as the pipeline runs.
    initial_state = {
        # Input
        "ticker"              : ticker.upper(),
        "analysis_request"    : analysis_request,
        # Agent outputs (all empty — filled by agents)
        "company_name"        : "",
        "market_data"         : {},
        "risk_metrics"        : {},
        "risk_insights"       : [],
        "fundamental_data"    : {},
        "fundamental_insights": [],
        "next_agent"          : "",
        "analysis_complete"   : False,
        "final_report"        : "",
        "recommendation"      : "",
        # Audit trail
        "agent_steps"         : [],
        "errors"              : [],
        # Required by MessagesState base class
        "messages"            : [],
    }

    # .invoke() runs the full graph to completion and returns final state
    final_state = graph.invoke(initial_state)

    logger.info(
        f"[PIPELINE] Complete. "
        f"Recommendation: {final_state.get('recommendation', 'N/A')}"
    )
    return final_state


# =============================================================================
# PIPELINE RUNNER — streaming
# =============================================================================

def stream_financial_analysis(ticker: str, analysis_request: str = ""):
    """
    Stream pipeline results node-by-node.

    Instead of waiting 30-60 seconds for the full result,
    the caller receives each agent's output as it completes.
    Used by the Streamlit UI to show real-time progress.

    Yields:
        dict: {node_name: partial_state_update} for each completed node
    """
    logger.info(f"[PIPELINE STREAM] Starting for {ticker}")

    graph = build_financial_analysis_graph()

    initial_state = {
        "ticker"              : ticker.upper(),
        "analysis_request"    : analysis_request or "Provide a comprehensive investment analysis",
        "company_name"        : "",
        "market_data"         : {},
        "risk_metrics"        : {},
        "risk_insights"       : [],
        "fundamental_data"    : {},
        "fundamental_insights": [],
        "next_agent"          : "",
        "analysis_complete"   : False,
        "final_report"        : "",
        "recommendation"      : "",
        "agent_steps"         : [],
        "errors"              : [],
        "messages"            : [],
    }

    # .stream() yields {node_name: state_update} as each node completes
    for chunk in graph.stream(initial_state):
        yield chunk
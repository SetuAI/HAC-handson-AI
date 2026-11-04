"""
The web backend for the pipeline UI.

Run from the project root:

    uvicorn api.main:app --reload

then open http://localhost:8000

WHAT THIS FILE DOES
-------------------
It runs exactly the same LangGraph pipeline as run_query.py, but instead of
printing to a terminal it streams each node's output to the browser the moment
that node finishes (Server-Sent Events). The UI in ui/ draws those updates as
they arrive, so you watch the question travel through the graph.

Two endpoints matter:

    GET  /api/run      starts a run and streams one event per node
    POST /api/review   a human's decision, which resumes a paused graph

The pause is real: when the output guardrail routes to human_review, the graph
stops (interrupt_before) and the stream ends with a "paused" event carrying the
thread id. The graph's state sits in the MemorySaver checkpointer until the
reviewer answers -- which is the same mechanism run_query.py uses, just with a
browser in place of input().
"""

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))

import config
from graph.build_graph import build_graph
from graph.state import create_initial_state
from guardrails.topic import check_by_keywords

UI_DIR = Path(__file__).parent.parent / "ui"

app = FastAPI(title="Annual Report RAG")

# One compiled graph for the whole process. Its MemorySaver holds the state of
# every paused run, keyed by thread id, so /api/review can resume any of them.
graph = build_graph()


SAMPLE_QUESTIONS = [
    "How did the company's profitability perform across both FYs?",
    "What new risks appeared in FY2026 that were not mentioned in FY2025?",
    "By what percentage did revenue grow between the two years?",
    "Did management sound more cautious about hiring in FY2026?",
    "How did the company describe its AI strategy?",
    "What is a good recipe for paneer butter masala?",
]


# ------------------------------------------------------------------------------
# Serialising node output for the browser
# ------------------------------------------------------------------------------

def serialise_candidates(candidates: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Copy each year's candidates into plain dicts with a stable id and rank.

    This must run as soon as the retrieve node's update arrives -- before the
    rerank node runs -- because rerank() writes rerank_score onto these same
    dict objects. Copying here captures the "before" picture faithfully.
    """
    out = {}
    for fiscal_year, results in candidates.items():
        out[fiscal_year] = [
            {
                "id": f"{fiscal_year}-{r['position']}",
                "rank": i + 1,
                "text": r["text"],
                "position": r["position"],
                "year_label": r["year_label"],
                "score": round(float(r["score"]), 4),
            }
            for i, r in enumerate(results)
        ]
    return out


def serialise_reranked(reranked: Dict[str, List[Dict]],
                       candidates: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    The kept chunks, each tagged with where it sat before reranking, plus the
    rerank score of every candidate so the UI can show what was discarded.
    """
    out = {}
    for fiscal_year, kept in reranked.items():
        found = candidates.get(fiscal_year, [])
        original_rank = {r["position"]: i + 1 for i, r in enumerate(found)}
        out[fiscal_year] = {
            "kept": [
                {
                    "id": f"{fiscal_year}-{r['position']}",
                    "rank": i + 1,
                    "was": original_rank.get(r["position"]),
                    "text": r["text"],
                    "position": r["position"],
                    "score": round(float(r["score"]), 4),
                    "rerank_score": round(float(r["rerank_score"]), 4),
                }
                for i, r in enumerate(kept)
            ],
            "all_scores": {
                f"{fiscal_year}-{r['position']}": round(float(r["rerank_score"]), 4)
                for r in found if "rerank_score" in r
            },
        }
    return out


def describe_update(node: str, update: dict, state: dict, question: str) -> dict:
    """Turn one node's raw state update into the payload the UI renders."""

    if node == "input_guardrail":
        if update.get("input_allowed"):
            keyword = check_by_keywords(question)  # local, costs nothing
            reason = (keyword.reason if keyword
                      else "passed the semantic similarity check against domain examples")
            return {"allowed": True, "check": "keyword" if keyword else "similarity",
                    "reason": reason}
        return {"allowed": False, "reason": update.get("input_block_reason"),
                "final_answer": update.get("final_answer")}

    if node == "retrieve":
        return {"candidates": serialise_candidates(update["candidate_chunks"]),
                "top_k": config.VECTOR_SEARCH_TOP_K}

    if node == "rerank":
        return {"reranked": serialise_reranked(update["reranked_chunks"],
                                               state.get("candidate_chunks") or {}),
                "top_k": config.RERANK_TOP_K, "model": config.RERANK_MODEL}

    if node == "generate":
        return {"draft_answer": update["draft_answer"], "model": config.GENERATION_MODEL}

    if node == "output_guardrail":
        return {
            "verdict": update.get("grounding_verdict"),
            "score": update.get("grounding_score"),
            "unsupported_items": update.get("unsupported_items") or [],
            "review_required": update.get("review_required"),
            "threshold": config.GROUNDING_THRESHOLD,
        }

    if node in ("finalise", "human_review"):
        return {"final_answer": update.get("final_answer")}

    return {}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_graph(graph_input, thread: dict, question: str):
    """
    Run (or resume) the graph, yielding one SSE event per finished node.

    stream_mode="updates" gives us each node's return value as soon as the node
    completes, and the generator does not advance to the next node until we
    have handled it -- which is what lets serialise_candidates() capture the
    pre-rerank state.
    """
    state: dict = {}
    started = time.perf_counter()
    try:
        for chunk in graph.stream(graph_input, thread, stream_mode="updates"):
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            for node, update in chunk.items():
                if node.startswith("__"):          # e.g. __interrupt__
                    continue
                update = update or {}
                state.update(update)
                yield sse("stage", {
                    "node": node,
                    "ms": elapsed_ms,
                    "data": describe_update(node, update, state, question),
                })
            started = time.perf_counter()

        snapshot = graph.get_state(thread)
        if snapshot.next:
            yield sse("paused", {"thread_id": thread["configurable"]["thread_id"],
                                 "next": list(snapshot.next)})
        else:
            yield sse("done", {"final_answer": snapshot.values.get("final_answer")})
    except Exception as error:  # surface the failure in the UI rather than hang
        yield sse("error", {"message": f"{type(error).__name__}: {error}"})


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    return {
        "companies": config.COMPANIES,
        "fiscal_years": config.FISCAL_YEARS,
        "documents": config.DOCUMENTS,
        "vector_top_k": config.VECTOR_SEARCH_TOP_K,
        "rerank_top_k": config.RERANK_TOP_K,
        "models": {
            "generation": config.GENERATION_MODEL,
            "guardrail": config.GUARDRAIL_MODEL,
            "embedding": config.EMBEDDING_MODEL,
            "rerank": config.RERANK_MODEL,
        },
        "sample_questions": SAMPLE_QUESTIONS,
    }


@app.get("/api/run")
def run(question: str, company: str):
    if company not in config.COMPANIES:
        raise HTTPException(400, f"company must be one of {config.COMPANIES}")
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial = create_initial_state(question=question, company=company)
    return StreamingResponse(
        stream_graph(initial, thread, question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class Review(BaseModel):
    thread_id: str
    decision: str              # "approved" or "rejected"
    note: Optional[str] = None  # an edited answer, when approving


@app.post("/api/review")
def review(body: Review):
    if body.decision not in ("approved", "rejected"):
        raise HTTPException(400, "decision must be 'approved' or 'rejected'")
    thread = {"configurable": {"thread_id": body.thread_id}}
    snapshot = graph.get_state(thread)
    if not snapshot.next:
        raise HTTPException(404, "no paused run with that thread id")

    graph.update_state(thread, {"review_decision": body.decision,
                                "review_note": (body.note or "").strip() or None})
    events = list(stream_graph(None, thread, snapshot.values.get("question", "")))
    # The resumed run is a single short node, so return it as plain JSON.
    parsed = []
    for raw in events:
        lines = raw.strip().split("\n")
        parsed.append({"event": lines[0][7:], "data": json.loads(lines[1][6:])})
    return {"events": parsed}


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")


app.mount("/", StaticFiles(directory=UI_DIR), name="ui")

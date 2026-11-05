# Assignment 2 — Agentic RAG with LangGraph

Build an internal assistant for **FlowDesk**, a fictional AI helpdesk product made by
the fictional company Lumio Labs. The assistant answers product-management questions from five
internal documents. An LLM research agent decides what to search and where. A
cross-encoder reranker picks the best evidence, a grader triggers a retry when that
evidence falls short, guardrails check both the question and the answer, and the graph
pauses for a human before it gives an answer that commits the company to anything.

## What is in this folder

| File                                       | What it is                                                                                                                       |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `Assignment_2_Agentic_RAG_LangGraph.pdf` | The step-by-step guide: architecture, setup, the starter code for every file, the TODOs, tasks and grading.**Start here.** |
| `Questions.pdf`                          | The questions your assistant must answer, and what a good answer must contain.                                                   |
| `data/`                                  | The five product documents your assistant searches.                                                                              |

### The documents

| File                                           | doc_type        | Contents                                                         | Dated       |
| ---------------------------------------------- | --------------- | ---------------------------------------------------------------- | ----------- |
| `01_FlowDesk_PRD_v3.0.pdf`                   | `prd`         | Personas, 3.0 features, requirements, non-goals, success metrics | 22 Jul 2026 |
| `02_FlowDesk_Pricing_and_Packaging_2026.pdf` | `pricing`     | Plans, limits, add-ons, discount policy, the 1 July price change | 1 Jul 2026  |
| `03_FlowDesk_Release_Notes_and_Roadmap.pdf`  | `releases`    | Releases 2.4 to 3.0.1, roadmap, deprecations                     | 2 Oct 2026  |
| `04_Competitive_Analysis_Q1_2026.pdf`        | `competitive` | FlowDesk vs HelpNest, Zentrik, ClearDesk; win/loss               | 10 Mar 2026 |
| `05_Customer_Insights_Q3_2026.pdf`           | `insights`    | ARR, NPS, churn, feature requests, interviews                    | 5 Oct 2026  |

The documents are designed to be tricky:

- **Facts are spread across documents.** Many questions need more than one search.
- **Some sources are out of date.** The competitive analysis still lists the old prices.
- **One document contains personal data.** The customer interview notes include names, emails and phone numbers.
- **Some questions ask for promises.** Discounts, refunds and roadmap dates must be checked by a human.

## What you build

```
input_guardrail -> research_agent (tool calls) -> rerank -> grade_context --retry--> research_agent
                                                                  |
                                                               generate -> output_guardrail -> finalise
                                                                                    |
                                                                         [pause] human_review
```

| Concept                                                                          | Where it lives                     |
| -------------------------------------------------------------------------------- | ---------------------------------- |
| Agentic retrieval (ReAct tool calling, metadata filters)                         | `tools.py`, `research_agent`   |
| Reranking (cross-encoder, before/after comparison)                               | `rerank.py`                      |
| Self-correcting loop with a hard limit                                           | `grade_context`, `after_grade` |
| Guardrails (topic, injection, PII, number grounding)                             | `guardrails.py`                  |
| Human-in-the-loop (`interrupt_before`, checkpointer, `update_state`, resume) | `pipeline.py`, `main.py`       |

## Prerequisites

- Assignment 1 (the LangGraph stock research desk with multiple agents)
- The class demo in `../annual-report-rag/` (reranking, guardrails and human-in-the-loop). Every TODO in this assignment has a close equivalent there.

## Quick start

```bash
mkdir product_rag && cd product_rag
cp -r ../assignment_2/data ./data
python3 -m venv .venv && source .venv/bin/activate
pip install langgraph langchain-openai chromadb pymupdf sentence-transformers python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env
```

Then follow the guide PDF from section 4.

## Submit

Submit your `product_rag/` folder **without** `.env`, `.venv` or `chroma_db`, plus an
`answers.md` that answers every question in `Questions.pdf`.

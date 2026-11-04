/* =====================================================================
   Annual Report RAG — pipeline UI
   Streams each LangGraph node's output from /api/run and renders it.
   ===================================================================== */

const STEPS = [
  { id: "input_guardrail",  label: "Input guardrail", panel: "panel-input_guardrail" },
  { id: "retrieve",         label: "Vector search",   panel: "panel-retrieve" },
  { id: "rerank",           label: "Rerank",          panel: "panel-rerank" },
  { id: "generate",         label: "Generate",        panel: "panel-generate" },
  { id: "output_guardrail", label: "Grounding check", panel: "panel-output_guardrail" },
  { id: "finalise",         label: "Finalise",        panel: "panel-final",        branch: true },
  { id: "human_review",     label: "Human review",    panel: "panel-human_review", branch: true },
];

const STOPWORDS = new Set(("the a an and or of to in on for with by is are was were be been did does do how what which " +
  "who whom when where why that this these those it its from as at between across both two year years fy " +
  "company company's their there than then not no any all about into over under more most less much many " +
  "has have had can could would should will shall may might our your they them we you i me").split(" "));

const state = {
  cfg: null,
  run: null,          // data for the current run
  sort: "rank",
  filter: "all",
  highlight: true,
  busy: false,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */
init();

async function init() {
  initTheme();
  renderSteps();
  try {
    state.cfg = await (await fetch("/api/config")).json();
  } catch (e) {
    showError("Could not reach the API. Start it with:  uvicorn api.main:app --reload");
    return;
  }
  const sel = $("#company");
  sel.innerHTML = state.cfg.companies.map((c) => `<option>${esc(c)}</option>`).join("");
  sel.value = state.cfg.companies.includes("Eternal") ? "Eternal" : state.cfg.companies[0];
  $(".k-vec").textContent = state.cfg.vector_top_k;
  $(".k-rr").textContent = state.cfg.rerank_top_k;

  $("#samples").innerHTML = state.cfg.sample_questions
    .map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
  $$("#samples .chip").forEach((b) => b.addEventListener("click", () => {
    $("#question").value = b.textContent;
    $("#question").focus();
  }));
  $("#question").value = state.cfg.sample_questions[0];

  $("#queryForm").addEventListener("submit", (e) => { e.preventDefault(); startRun(); });
  bindToolbar();
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#drawer").addEventListener("click", (e) => { if (e.target.id === "drawer") closeDrawer(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("rag-theme"); } catch {}
  if (saved) document.documentElement.dataset.theme = saved;
  $("#themeToggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("rag-theme", next); } catch {}
  });
}

/* ------------------------------------------------------------------ */
/* Stepper                                                             */
/* ------------------------------------------------------------------ */
function renderSteps() {
  $("#steps").innerHTML = STEPS.map((s, i) => `
    <li class="step ${s.branch ? "branch" : ""}" data-step="${s.id}" title="Jump to ${esc(s.label)}">
      <span class="label"><span class="dot"></span>${i + 1}. ${esc(s.label)}</span>
      <span class="meta"></span>
    </li>`).join("");
  $$(".step").forEach((el) => el.addEventListener("click", () => {
    const step = STEPS.find((s) => s.id === el.dataset.step);
    const panel = document.getElementById(step.panel);
    if (!panel || panel.classList.contains("hidden")) return;
    panel.scrollIntoView();
    flash(panel);
  }));
}

function setStep(id, status, meta) {
  const el = $(`.step[data-step="${id}"]`);
  if (!el) return;
  el.classList.remove("running", "done", "blocked", "paused", "skipped");
  if (status) el.classList.add(status);
  if (meta !== undefined) $(".meta", el).textContent = meta;
}

function flash(panel) {
  panel.classList.add("active");
  setTimeout(() => panel.classList.remove("active"), 1200);
}

/* ------------------------------------------------------------------ */
/* Run                                                                 */
/* ------------------------------------------------------------------ */
async function startRun() {
  if (state.busy) return;
  const question = $("#question").value.trim();
  const company = $("#company").value;
  if (!question) { $("#question").focus(); return; }

  resetPanels(question, company);
  state.busy = true;
  $("#runBtn").disabled = true;
  $("#runBtn").textContent = "Running…";

  try {
    const url = `/api/run?question=${encodeURIComponent(question)}&company=${encodeURIComponent(company)}`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
    await readSSE(resp, handleEvent);
  } catch (e) {
    showError(e.message);
  } finally {
    state.busy = false;
    $("#runBtn").disabled = false;
    $("#runBtn").textContent = "Run pipeline ▸";
  }
}

/* Parse a text/event-stream response body without EventSource
   (EventSource would auto-reconnect and silently re-run the pipeline). */
async function readSSE(resp, onEvent) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}

function resetPanels(question, company) {
  state.run = { question, company, terms: queryTerms(question), candidates: null, reranked: null,
                grounding: null, draft: null, threadId: null };
  state.sort = "rank"; state.filter = "all";
  hideError();
  $("#stages").classList.remove("hidden");
  $("#echoQuestion").textContent = question;
  $("#echoCompany").textContent = company;

  for (const s of STEPS) setStep(s.id, null, "");
  setStep("input_guardrail", "running", "checking…");

  const waiting = `<div class="placeholder">Waiting…</div>`;
  for (const id of ["panel-input_guardrail", "panel-generate", "panel-output_guardrail", "panel-final"]) {
    $(`#${id} .body`).innerHTML = waiting;
    document.getElementById(id).classList.remove("hidden");
  }
  $(".final").className = "card body final";
  $("#candidates").innerHTML = waiting;
  $("#reranked").innerHTML = waiting;
  $("#candToolbar").classList.add("hidden");
  $("#panel-retrieve").classList.remove("hidden");
  $("#panel-rerank").classList.remove("hidden");
  $("#panel-human_review").classList.add("hidden");
  $$(".timing").forEach((t) => (t.textContent = ""));
  syncToolbar();
  $("#panel-query").scrollIntoView();
}

function handleEvent(event, payload) {
  if (event === "stage") return onStage(payload.node, payload.data, payload.ms);
  if (event === "paused") return onPaused(payload);
  if (event === "done") return onDone(payload);
  if (event === "error") return showError(payload.message);
}

function timing(node, ms) {
  const el = $(`.timing[data-for="${node}"]`);
  if (el) el.textContent = fmtMs(ms);
  return fmtMs(ms);
}
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`);

function onStage(node, data, ms) {
  const t = timing(node, ms);
  const scrollTo = (id) => { const p = document.getElementById(id); p.scrollIntoView(); flash(p); };

  switch (node) {
    case "input_guardrail":
      renderInputGuardrail(data);
      if (data.allowed) {
        setStep(node, "done", `allowed · ${t}`);
        setStep("retrieve", "running", "searching…");
      } else {
        setStep(node, "blocked", `blocked · ${t}`);
        for (const s of STEPS.slice(1)) setStep(s.id, "skipped", "skipped");
        for (const id of ["panel-retrieve", "panel-rerank", "panel-generate", "panel-output_guardrail"])
          document.getElementById(id).classList.add("hidden");
        renderFinal(data.final_answer, "bad", "Refused by the input guardrail — no search or LLM call was made.");
        scrollTo("panel-input_guardrail");
      }
      break;

    case "retrieve":
      state.run.candidates = data.candidates;
      renderCandidates();
      $("#candToolbar").classList.remove("hidden");
      setStep(node, "done", `${countAll(data.candidates)} chunks · ${t}`);
      setStep("rerank", "running", "cross-encoding…");
      scrollTo("panel-retrieve");
      break;

    case "rerank":
      state.run.reranked = data.reranked;
      renderReranked();
      syncToolbar();
      renderCandidates();
      setStep(node, "done", `${countKept(data.reranked)} kept · ${t}`);
      setStep("generate", "running", "writing…");
      scrollTo("panel-rerank");
      break;

    case "generate":
      state.run.draft = data.draft_answer;
      $("#panel-generate .body").innerHTML =
        `<div class="answer-meta">Draft from <b>${esc(data.model)}</b> · temperature 0</div>
         <div class="answer">${md(data.draft_answer)}</div>`;
      setStep(node, "done", t);
      setStep("output_guardrail", "running", "verifying…");
      scrollTo("panel-generate");
      break;

    case "output_guardrail":
      state.run.grounding = data;
      renderGrounding(data);
      setStep(node, "done", `${data.verdict.replace("_", " ")} · ${t}`);
      setStep(data.review_required ? "human_review" : "finalise", "running", "…");
      scrollTo("panel-output_guardrail");
      break;

    case "finalise":
      setStep("output_guardrail", "done");
      setStep("finalise", "done", t);
      setStep("human_review", "skipped", "not needed");
      $("#panel-human_review").classList.add("hidden");
      renderFinal(data.final_answer, "ok", "Grounded — sent straight to the user.");
      scrollTo("panel-final");
      break;

    case "human_review":
      setStep("human_review", "done", `${state.run.reviewDecision || "reviewed"} · ${t}`);
      renderFinal(data.final_answer,
                  state.run.reviewDecision === "rejected" ? "bad" : "ok",
                  state.run.reviewDecision === "rejected"
                    ? "Rejected by the reviewer — a safe fallback message is shown."
                    : state.run.reviewEdited
                      ? "Approved with edits by the reviewer."
                      : "Approved as written by the reviewer.");
      $("#panel-human_review .body").innerHTML =
        `<div class="verdict"><span class="badge ${state.run.reviewDecision === "rejected" ? "bad" : "ok"}">
           ${esc(state.run.reviewDecision)}</span><span>Graph resumed from its checkpoint and ran <code>human_review</code>.</span></div>`;
      scrollTo("panel-final");
      break;
  }
}

function onPaused(p) {
  state.run.threadId = p.thread_id;
  setStep("output_guardrail", "done");
  setStep("finalise", "skipped", "not taken");
  setStep("human_review", "paused", "waiting for you");
  renderReview();
  renderFinal(null, "warn", "Awaiting human review — nothing has been shown to the user yet.");
  const panel = $("#panel-human_review");
  panel.classList.remove("hidden");
  panel.scrollIntoView();
  flash(panel);
}

function onDone() { /* final answer already rendered by its stage */ }

/* ------------------------------------------------------------------ */
/* Stage renderers                                                     */
/* ------------------------------------------------------------------ */
function renderInputGuardrail(d) {
  $("#panel-input_guardrail .body").innerHTML = d.allowed
    ? `<div class="verdict"><span class="badge ok">Allowed</span>
         <span>Passed by the <b>${esc(d.check)}</b> check</span></div>
       <dl class="kv"><dt>Reason</dt><dd>${esc(d.reason)}</dd></dl>`
    : `<div class="verdict"><span class="badge bad">Blocked</span>
         <span>Stopped before any retrieval or generation cost was spent.</span></div>
       <dl class="kv"><dt>Reason</dt><dd>${esc(d.reason)}</dd></dl>`;
}

function fyHead(fy, extra) {
  const doc = state.cfg.documents.find((d) => d.fiscal_year === fy && d.company === state.run.company);
  return `<div class="fy-head"><div><span class="fy-name">${esc(fy)}</span>
            <span class="fy-label">${esc(doc ? doc.year_label : "")}</span></div>
            <span class="count">${extra}</span></div>`;
}

function renderCandidates() {
  const cands = state.run.candidates;
  if (!cands) return;
  const rr = state.run.reranked;
  const html = Object.keys(cands).sort().map((fy) => {
    const rrFy = rr && rr[fy];
    const keptRank = {};
    if (rrFy) rrFy.kept.forEach((k) => (keptRank[k.id] = k.rank));
    const maxSim = Math.max(...cands[fy].map((c) => c.score), 0.0001);

    let items = cands[fy].map((c) => ({ ...c, rr: rrFy ? rrFy.all_scores[c.id] : null, kept: keptRank[c.id] }));
    if (state.sort === "rerank" && rrFy) items.sort((a, b) => b.rr - a.rr);
    if (state.filter === "kept") items = items.filter((c) => c.kept);
    if (state.filter === "dropped") items = items.filter((c) => !c.kept);

    const list = items.map((c) => `
      <li class="chunk ${rrFy ? (c.kept ? "kept" : "dropped") : ""}" data-id="${c.id}" data-fy="${fy}">
        <div class="chunk-top">
          <span class="rank">#${c.rank}</span>
          <span class="bar" title="similarity"><span style="width:${(c.score / maxSim) * 100}%"></span></span>
          <span class="score">sim ${c.score.toFixed(3)}</span>
          ${c.rr != null ? `<span class="score">rr ${c.rr.toFixed(2)}</span>` : ""}
          ${rrFy ? (c.kept ? `<span class="tag keep">KEPT → #${c.kept}</span>` : `<span class="tag drop">dropped</span>`) : ""}
        </div>
        <div class="chunk-text">${hl(c.text)}</div>
        <div class="chunk-foot"><span>chunk position ${c.position}</span><span>${c.text.length} chars</span>
          <a href="#" data-open="${c.id}">open ⤢</a></div>
      </li>`).join("");

    const countText = rrFy
      ? `${rrFy.kept.length} kept / ${cands[fy].length - rrFy.kept.length} dropped`
      : `${cands[fy].length} candidates`;
    return `<div class="fy-col" data-fy="${fy}">${fyHead(fy, countText)}
              <ul class="chunk-list">${list || `<li class="placeholder" style="padding:8px">None</li>`}</ul></div>`;
  }).join("");
  $("#candidates").innerHTML = html;
  bindChunks($("#candidates"));
}

function renderReranked() {
  const rr = state.run.reranked;
  const html = Object.keys(rr).sort().map((fy) => {
    const kept = rr[fy].kept;
    const scores = kept.map((k) => k.rerank_score);
    const min = Math.min(...scores, 0), max = Math.max(...scores, 0.0001);
    const span = max - min || 1;
    const promoted = kept.filter((k) => k.was && k.was > state.cfg.rerank_top_k).length;

    const list = kept.map((k) => {
      const delta = k.was ? k.was - k.rank : 0;
      const move = !k.was ? `<span class="move same">new</span>`
        : delta > 0 ? `<span class="move up" title="moved up ${delta}">▲ ${delta}</span>`
        : delta < 0 ? `<span class="move down" title="moved down ${-delta}">▼ ${-delta}</span>`
        : `<span class="move same">＝</span>`;
      return `
      <li class="chunk kept" data-id="${k.id}" data-fy="${fy}" data-link="${k.id}">
        <div class="chunk-top">
          <span class="rank">#${k.rank}</span>
          <span class="score">was #${k.was ?? "?"}</span>${move}
          <span class="bar" title="rerank score"><span style="width:${((k.rerank_score - min) / span) * 100}%"></span></span>
          <span class="score">rr ${k.rerank_score.toFixed(3)}</span>
        </div>
        <div class="chunk-text">${hl(k.text)}</div>
        <div class="chunk-foot"><span>sim ${k.score.toFixed(3)}</span><span>chunk position ${k.position}</span>
          <a href="#" data-open="${k.id}">open ⤢</a></div>
      </li>`;
    }).join("");
    return `<div class="fy-col rr" data-fy="${fy}">${fyHead(fy, `top ${kept.length}`)}
              <ul class="chunk-list">${list || `<li class="placeholder" style="padding:8px">No chunks</li>`}</ul>
              <div class="rr-summary">${promoted} of ${kept.length} came from outside the original top ${state.cfg.rerank_top_k}
                · ${countAllFy(fy) - kept.length} discarded · model <code>${esc(state.cfg.models.rerank)}</code></div></div>`;
  }).join("");
  $("#reranked").innerHTML = html;
  bindChunks($("#reranked"));

  // Hover a reranked chunk -> spotlight its original slot in the 20.
  $$("#reranked [data-link]").forEach((el) => {
    el.addEventListener("mouseenter", () => linkCandidate(el.dataset.link, true));
    el.addEventListener("mouseleave", () => linkCandidate(el.dataset.link, false));
  });
}

function linkCandidate(id, on) {
  const target = $(`#candidates .chunk[data-id="${id}"]`);
  if (!target) return;
  target.classList.toggle("linked", on);
  if (on) {
    const list = target.closest(".chunk-list");
    list.scrollTo({ top: target.offsetTop - list.offsetTop - 40, behavior: "smooth" });
  }
}

function renderGrounding(g) {
  const cls = g.verdict === "grounded" ? "ok" : g.verdict === "ungrounded" ? "bad" : "warn";
  const pct = g.score != null ? Math.round(g.score * 100) : 0;
  const color = cls === "ok" ? "var(--ok)" : cls === "bad" ? "var(--bad)" : "var(--warn)";
  $("#panel-output_guardrail .body").innerHTML = `
    <div class="verdict"><span class="badge ${cls}">${esc(g.verdict.replace("_", " "))}</span>
      <span>${g.review_required ? "Routed to <b>human review</b> — the graph will pause." : "Routed to <b>finalise</b> — no review needed."}</span></div>
    <div class="meter" title="grounding score ${pct}%">
      <div class="fill" style="width:${pct}%;background:${color}"></div>
      <div class="thresh" style="left:${g.threshold * 100}%" data-label="threshold ${Math.round(g.threshold * 100)}%"></div>
    </div>
    <dl class="kv">
      <dt>Grounding score</dt><dd>${g.score != null ? g.score.toFixed(2) : "—"}</dd>
      <dt>Unsupported</dt><dd>${g.unsupported_items.length ? `${g.unsupported_items.length} item(s)` : "none"}</dd>
    </dl>
    ${g.unsupported_items.length ? `<ul class="unsupported">${g.unsupported_items.map((u) => `<li>${esc(u)}</li>`).join("")}</ul>` : ""}`;

  // Mark unsupported figures inside the draft so they are easy to find.
  if (g.unsupported_items.length && state.run.draft) {
    $("#panel-generate .answer").innerHTML = md(state.run.draft, g.unsupported_items);
  }
}

function renderReview() {
  const g = state.run.grounding || {};
  const body = $("#panel-human_review .body");
  body.innerHTML = `
    <div class="paused-banner">⏸ Graph paused before <code>human_review</code> · thread ${esc(state.run.threadId.slice(0, 8))}…</div>
    ${g.unsupported_items && g.unsupported_items.length
      ? `<p>The checker could not verify: ${g.unsupported_items.map((u) => `<code>${esc(u)}</code>`).join(", ")}.
         Compare against the reranked extracts above, then decide.</p>` : `<p>Check the draft against the reranked extracts above, then decide.</p>`}
    <textarea id="reviewText">${esc(state.run.draft || "")}</textarea>
    <div class="actions">
      <button class="btn-ok" id="approveBtn">Approve &amp; resume</button>
      <button class="btn-bad" id="rejectBtn">Reject</button>
      <button class="ghost" id="resetDraft">Reset to draft</button>
    </div>`;
  $("#resetDraft").addEventListener("click", () => ($("#reviewText").value = state.run.draft));
  $("#approveBtn").addEventListener("click", () => submitReview("approved"));
  $("#rejectBtn").addEventListener("click", () => submitReview("rejected"));
}

async function submitReview(decision) {
  const text = $("#reviewText").value.trim();
  const edited = decision === "approved" && text && text !== state.run.draft.trim();
  state.run.reviewDecision = decision;
  state.run.reviewEdited = edited;
  $$("#panel-human_review button").forEach((b) => (b.disabled = true));
  setStep("human_review", "running", "resuming…");
  try {
    const resp = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: state.run.threadId, decision, note: edited ? text : null }),
    });
    if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
    const { events } = await resp.json();
    events.forEach((e) => handleEvent(e.event, e.data));
  } catch (e) {
    showError(e.message);
    $$("#panel-human_review button").forEach((b) => (b.disabled = false));
    setStep("human_review", "paused", "waiting for you");
  }
}

function renderFinal(text, cls, note) {
  const el = $(".final");
  el.className = `card body final ${cls}`;
  el.innerHTML = `<div class="answer-meta">${esc(note)}</div>
                  ${text ? `<div class="answer">${md(text)}</div>` : ""}`;
}

/* ------------------------------------------------------------------ */
/* Interactions                                                        */
/* ------------------------------------------------------------------ */
function bindToolbar() {
  $$(".seg").forEach((seg) => seg.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn || btn.disabled) return;
    state[seg.dataset.control] = btn.dataset.value;
    syncToolbar();
    renderCandidates();
  }));
  $("#highlightToggle").addEventListener("change", (e) => {
    state.highlight = e.target.checked;
    renderCandidates();
    if (state.run && state.run.reranked) renderReranked();
  });
  $("#expandAll").addEventListener("click", (e) => {
    const chunks = $$("#candidates .chunk");
    const open = !chunks.every((c) => c.classList.contains("open"));
    chunks.forEach((c) => c.classList.toggle("open", open));
    e.target.textContent = open ? "Collapse all" : "Expand all";
  });
}

function syncToolbar() {
  const hasRR = !!(state.run && state.run.reranked);
  $$(".seg button").forEach((b) => {
    const ctl = b.parentElement.dataset.control;
    if (b.dataset.value !== "rank" && b.dataset.value !== "all") b.disabled = !hasRR;
    b.classList.toggle("on", state[ctl] === b.dataset.value);
  });
}

function bindChunks(root) {
  $$(".chunk", root).forEach((el) => el.addEventListener("click", (e) => {
    const open = e.target.closest("[data-open]");
    if (open) { e.preventDefault(); openDrawer(el.dataset.id, el.dataset.fy); return; }
    el.classList.toggle("open");
  }));
}

function openDrawer(id, fy) {
  const c = state.run.candidates[fy].find((x) => x.id === id);
  const rrFy = state.run.reranked && state.run.reranked[fy];
  const kept = rrFy && rrFy.kept.find((k) => k.id === id);
  const rrScore = rrFy ? rrFy.all_scores[id] : null;
  $("#drawerTitle").innerHTML = `
    <span class="pill" style="background:var(--${fy === "FY2025" ? "fy25" : "fy26"}-soft);color:var(--${fy === "FY2025" ? "fy25" : "fy26"})">${esc(fy)}</span>
    <b>&nbsp;Search #${c.rank}</b>${kept ? ` → <b>Rerank #${kept.rank}</b>` : rrFy ? " · dropped" : ""}
    <span class="score">&nbsp;sim ${c.score.toFixed(3)}${rrScore != null ? ` · rr ${rrScore.toFixed(3)}` : ""} · position ${c.position}</span>`;
  $("#drawerBody").innerHTML = hl(c.text);
  $("#drawer").classList.remove("hidden");
}
function closeDrawer() { $("#drawer").classList.add("hidden"); }

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
function queryTerms(q) {
  return [...new Set(q.toLowerCase().match(/[a-z][a-z0-9-]{2,}/g) || [])]
    .filter((w) => !STOPWORDS.has(w));
}

function hl(text) {
  let out = esc(text);
  if (!state.highlight || !state.run || !state.run.terms.length) return out;
  const stems = state.run.terms.map((t) => t.replace(/(ies|es|s|ing|ed|ity|ability)$/, "")).filter((t) => t.length >= 3);
  if (!stems.length) return out;
  const re = new RegExp(`\\b(${stems.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})[a-z]*`, "gi");
  return out.replace(re, "<mark>$&</mark>");
}

/* Minimal, safe markdown: headings, bullets, numbered lists, bold, italics. */
function md(src, unsupported = []) {
  const lines = esc(src).split("\n");
  let html = "", list = null;
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  const inline = (s) => {
    s = s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|\W)\*(?!\s)(.+?)\*(?=\W|$)/g, "$1<i>$2</i>");
    for (const u of unsupported) {
      const pat = String(u).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      if (/^\d/.test(u)) s = s.replace(new RegExp(`(?<![\\d.])(${pat.split("").join(",?")})(?![\\d])`, "g"), `<span class="unsup" title="not found in sources">$1</span>`);
    }
    return s;
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^#{1,6}\s+(.*)/))) { close(); html += `<h3>${inline(m[1])}</h3>`; }
    else if ((m = line.match(/^\s*[-*•]\s+(.*)/))) { if (list !== "ul") { close(); html += "<ul>"; list = "ul"; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { if (list !== "ol") { close(); html += "<ol>"; list = "ol"; } html += `<li>${inline(m[1])}</li>`; }
    else if (!line.trim()) { /* blank lines keep an open list open */ }
    else if (list && /^\s+/.test(raw)) { html = html.replace(/<\/li>$/, `<br>${inline(line.trim())}</li>`); }
    else { close(); html += `<p>${inline(line)}</p>`; }
  }
  close();
  return html;
}

const countAll = (c) => Object.values(c).reduce((n, a) => n + a.length, 0);
const countKept = (r) => Object.values(r).reduce((n, a) => n + a.kept.length, 0);
const countAllFy = (fy) => (state.run.candidates[fy] || []).length;

function showError(msg) { const b = $("#errorBox"); b.textContent = msg; b.classList.remove("hidden"); }
function hideError() { $("#errorBox").classList.add("hidden"); }

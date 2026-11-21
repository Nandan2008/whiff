/* whiff — UI behaviour. No framework: one list, one detail view, one form. */

const form = document.getElementById("config");
const listEl = document.getElementById("query-list");
const tallyEl = document.getElementById("tally");
const detailEl = document.getElementById("detail");
const resetBtn = document.getElementById("reset");
const dropEl = document.getElementById("drop");
const filesEl = document.getElementById("files");
const pickEl = document.getElementById("pick");
const srcEl = document.getElementById("src");
const askForm = document.getElementById("ask");
const qEl = document.getElementById("q");

const DEFAULTS = { chunker: "fixed", chunk_size: "150", overlap: "0", k: "5" };

let selected = null;
let timer = null;

const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const hue = (verdict) => `var(--v-${verdict})`;

const params = () => new URLSearchParams(new FormData(form)).toString();

async function get(path, extra = "") {
  const res = await fetch(`${path}?${params()}${extra}`);
  return res.json();
}

/* ------------------------------------------------------------- overview */

async function loadOverview() {
  let data;
  try {
    data = await get("/api/overview");
  } catch {
    tallyEl.textContent = "Could not reach the server.";
    return;
  }
  if (data.error) {
    tallyEl.textContent = "Invalid configuration.";
    detailEl.innerHTML = `<div class="error">${esc(data.error)}</div>`;
    return;
  }

  const s = data.summary;
  const c = data.corpus;

  srcEl.innerHTML = c.source === "uploaded"
    ? `Your documents: <b>${c.docs}</b> loaded.`
    : "Using the sample warehouse documents.";

  tallyEl.innerHTML = data.has_queries
    ? `<b>${s.n_retrieved} of ${s.n_queries}</b> found &middot; ${c.chunks} chunks`
    : `<b>${c.docs}</b> documents &middot; ${c.chunks} chunks`;

  if (!data.has_queries) {
    listEl.innerHTML = `<li class="empty">No test questions for these
      documents, so there is nothing to mark right or wrong. Ask a question
      above and see what comes back.</li>`;
    return;
  }

  listEl.innerHTML = data.queries.map((q) => `
    <li>
      <button class="qbtn" data-id="${esc(q.id)}"
              aria-current="${q.id === selected}">
        <span class="qhead">
          <span class="dot" style="background:${hue(q.verdict)}"></span>
          <span class="qid">${esc(q.id)}</span>
          <span class="verdict-name">${q.ok ? "found" : esc(q.verdict)}</span>
        </span>
        <span class="qtext">${esc(q.text)}</span>
      </button>
    </li>`).join("");

  listEl.querySelectorAll(".qbtn").forEach((b) =>
    b.addEventListener("click", () => select(b.dataset.id)));

  if (selected) loadDetail(selected);
}

/* --------------------------------------------------------------- detail */

function select(id) {
  selected = id;
  listEl.querySelectorAll(".qbtn").forEach((b) =>
    b.setAttribute("aria-current", String(b.dataset.id === id)));
  loadDetail(id);
}

async function loadDetail(id) {
  const data = await get("/api/explain", `&id=${encodeURIComponent(id)}`);
  detailEl.innerHTML = data.error
    ? `<div class="error">${esc(data.error)}</div>`
    : render(data);
}

/* Positions as cells, the cutoff as a hard line, and the expected
   document's marker wherever it actually landed. */
function rankStrip(t) {
  const k = t.cutoff_k, pool = t.candidate_k, rank = t.best_rank;
  const insideMax = Math.min(k, 10);
  const beyondMax = Math.min(pool, k + 8);
  const out = [];

  const cell = (n, cls, isGold) =>
    `<div class="cell ${cls}${isGold ? " gold" : ""}"${
      isGold ? ` style="--hue:${hue(t.verdict)}"` : ""}>${n}</div>`;

  for (let i = 1; i <= insideMax; i++) out.push(cell(i, "inside", i === rank));
  if (insideMax < k) out.push(`<div class="cell ellipsis">…</div>`);

  out.push(`<div class="cutoff" data-label="cutoff k=${k}"></div>`);

  for (let i = k + 1; i <= beyondMax; i++) out.push(cell(i, "", i === rank));

  if (rank && rank > beyondMax) {
    out.push(`<div class="cell ellipsis">…</div>`, cell(rank, "", true));
  } else if (beyondMax < pool) {
    out.push(`<div class="cell ellipsis">…</div>`);
  }

  let note;
  if (rank === null) {
    note = `Nothing from the expected document reached the top ${pool}.`;
  } else if (rank <= k) {
    note = `The expected document ranked ${rank} — inside the cutoff.`;
  } else {
    note = `The expected document ranked ${rank}. The cutoff is ${k}, so it was
            found and then discarded.`;
  }

  return `<div class="strip">${out.join("")}</div>
          <p class="strip-note">${note}</p>`;
}

function render(d) {
  const t = d.triage, l = d.localize, out = [];

  const facts = [
    l.gold_chunk_position
      ? `answer in chunk <b>${l.gold_chunk_position}</b> of ${l.doc_chunk_count}` : "",
    l.doc_rank ? `whole document ranks <b>${l.doc_rank}</b> of ${l.n_docs}` : "",
  ].filter(Boolean);

  out.push(`<div class="panel">
    <p class="question">${esc(d.query.text)}</p>
    <p class="expected">Expected <code>${esc(d.query.gold_doc)}</code></p>
    ${rankStrip(t)}
    ${facts.length ? `<div class="facts">${
      facts.map((f) => `<span>${f}</span>`).join("")}</div>` : ""}
  </div>`);

  const notes = t.notes
    .concat(l.notes)
    .concat(l.fragmentation ? [l.fragmentation] : []);

  out.push(`<div class="panel">
    <h2>What broke</h2>
    <div class="verdict-line">
      <span class="dot" style="background:${hue(t.verdict)}"></span>
      <span class="verdict-code">${esc(t.verdict)}</span>
    </div>
    <p class="blame">${esc(t.blame)}</p>
    ${notes.length ? `<ul class="notes">${
      notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
  </div>`);

  if (d.fix && d.fix.found) {
    const changes = d.fix.changes.map((c) =>
      `${esc(c.field)}: <span class="from">${esc(c.from)}</span> → ` +
      `<span class="to">${esc(c.to)}</span>`).join("<br>");
    out.push(`<div class="panel">
      <h2>Smallest fix</h2>
      <div class="fix-change">${changes}</div>
      <div class="facts">
        <span>${d.fix.requires_reembedding
          ? "needs re-embedding" : "no re-embedding"}</span>
        <span class="${d.fix.safe ? "good" : "warn"}">${d.fix.safe
          ? "breaks nothing else"
          : `breaks ${d.fix.regressions.length} other ${
              d.fix.regressions.length === 1 ? "query" : "queries"}`}</span>
        ${d.fix.improvements.length
          ? `<span>also fixes ${d.fix.improvements.length}</span>` : ""}
        <span>${d.fix.configs_evaluated} tried</span>
      </div>
    </div>`);
  } else if (d.fix) {
    out.push(`<div class="panel">
      <h2>Smallest fix</h2>
      <p class="blame">No configuration within ${d.fix.max_changed} changed
      ${d.fix.max_changed === 1 ? "field" : "fields"} retrieves this. It needs
      a better retriever, not tuning.</p>
      <div class="facts"><span>${d.fix.configs_evaluated} tried</span>${
        d.fix.skipped_degenerate
          ? `<span>${d.fix.skipped_degenerate} refused for returning most of
             the corpus</span>` : ""}</div>
    </div>`);
  }

  out.push(`<div class="panel">
    <h2>What came back instead</h2>
    <table>
      <thead><tr>
        <th class="num">#</th><th class="num">score</th>
        <th>chunk</th><th>text</th>
      </tr></thead>
      <tbody>${d.results.map((r) => `
        <tr class="${r.is_gold ? "is-gold" : ""}">
          <td class="num">${r.rank}</td>
          <td class="num">${r.score.toFixed(3)}</td>
          <td class="chunk">${esc(r.chunk_id)}${
            r.is_gold ? ' <span class="tag">expected</span>' : ""}</td>
          <td class="snippet">${esc(r.text)}</td>
        </tr>`).join("")}</tbody>
    </table>
  </div>`);

  return out.join("");
}

/* --------------------------------------------------------------- search */

/* A free-text question has no known right answer, so this deliberately
   shows the ranking and nothing else - no verdict, no green tick. */
async function doSearch(question) {
  if (!question.trim()) return;
  selected = null;
  listEl.querySelectorAll(".qbtn").forEach((b) =>
    b.setAttribute("aria-current", "false"));

  detailEl.innerHTML = `<div class="panel"><p class="blame">Searching…</p></div>`;
  const d = await get("/api/search", `&q=${encodeURIComponent(question)}`);
  if (d.error) {
    detailEl.innerHTML = `<div class="error">${esc(d.error)}</div>`;
    return;
  }

  if (!d.results.length) {
    detailEl.innerHTML = `<div class="panel">
      <p class="question">${esc(d.question)}</p>
      <p class="blame">Nothing came back. The documents may be empty.</p>
    </div>`;
    return;
  }

  detailEl.innerHTML = `<div class="panel">
    <p class="question">${esc(d.question)}</p>
    <p class="expected">No expected answer was given for this question, so
      there is no verdict — this is simply what the retriever returned.
      Everything above the line would be sent to the model.</p>
    <table>
      <thead><tr>
        <th class="num">#</th><th class="num">score</th>
        <th>chunk</th><th>text</th>
      </tr></thead>
      <tbody>${d.results.map((r, i) => `
        ${i === d.cutoff ? `<tr class="cut"><td colspan="4">cutoff k=${
          d.cutoff} — everything below is discarded</td></tr>` : ""}
        <tr class="${r.inside_cutoff ? "" : "out"}">
          <td class="num">${r.rank}</td>
          <td class="num">${r.score.toFixed(3)}</td>
          <td class="chunk">${esc(r.chunk_id)}</td>
          <td class="snippet">${esc(r.text)}</td>
        </tr>`).join("")}</tbody>
    </table>
  </div>`;
}

/* --------------------------------------------------------------- upload */

function readFile(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve({ name: file.name, text: reader.result });
    reader.onerror = () => resolve({ name: file.name, text: "" });
    reader.readAsText(file);
  });
}

async function upload(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;

  srcEl.textContent = `Reading ${files.length} file${files.length > 1 ? "s" : ""}…`;
  const payload = { files: await Promise.all(files.map(readFile)) };

  let res, data;
  try {
    res = await fetch("/api/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    data = await res.json();
  } catch {
    srcEl.textContent = "Upload failed — is the server still running?";
    return;
  }

  if (!res.ok || data.error) {
    srcEl.innerHTML = `<span class="warn">${esc(data.error || "Upload failed")}</span>`;
    return;
  }

  selected = null;
  detailEl.innerHTML = `<p class="placeholder">Ask a question above to see
    what these documents return.</p>`;
  qEl.focus();
  await loadOverview();

  if (data.problems && data.problems.length) {
    srcEl.innerHTML += ` <span class="warn">${data.problems.length} file(s)
      skipped.</span>`;
  }
}

/* ---------------------------------------------------------------- events */

pickEl.addEventListener("click", () => filesEl.click());
filesEl.addEventListener("change", () => upload(filesEl.files));

["dragenter", "dragover"].forEach((e) =>
  dropEl.addEventListener(e, (ev) => {
    ev.preventDefault();
    dropEl.classList.add("over");
  }));
["dragleave", "drop"].forEach((e) =>
  dropEl.addEventListener(e, (ev) => {
    ev.preventDefault();
    dropEl.classList.remove("over");
  }));
dropEl.addEventListener("drop", (ev) => upload(ev.dataTransfer.files));

askForm.addEventListener("submit", (ev) => {
  ev.preventDefault();
  doSearch(qEl.value);
});


form.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(loadOverview, 250);
});

resetBtn.addEventListener("click", () => {
  for (const [name, value] of Object.entries(DEFAULTS)) {
    form.elements[name].value = value;
  }
  loadOverview();
});

loadOverview();

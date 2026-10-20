"use strict";
// Viewer for agent-written programs of Qwen3-1.7B's MLP down-projection components, several layers.
// Routes: #/ overview, #/l/<L> a layer's components, #/c/<L>/<i> component pages, #/glossary[/<key>].

const SHARD = 64;
const PAGE = 100;
const MODELS = {
  sol: { name: "Sol", full: "OpenAI gpt-6.1-sol, high reasoning effort" },
  haiku: { name: "Haiku 5.5", full: "Anthropic claude-haiku-5-5, high effort" },
  luna: { name: "Luna", full: "OpenAI gpt-6-luna, high reasoning effort" },
  lookup: { name: "Token lookup" },
  lookup_plus: { name: "Token lookup+", full: "token lookup fit without documents' first tokens, which stay unchanged" },
  zero: { name: "Zeroing" },
};
// One-at-a-time scores, in the order the data stores them.
const METRICS = [
  { name: "KL", tip: "KL from the original model's next-token distribution, nats/token; lower is better", lowerBetter: true },
  { name: "ΔCE", tip: "change in next-token loss, nats/token; lower is better", lowerBetter: true },
  { name: "VE", tip: "variance of the component's activation the predictor explains; higher is better", lowerBetter: false },
];
let PROGRAM_MODELS = ["sol", "haiku", "luna"];  // the current layer's; set per page


const BLOG = "https://www.goodfire.ai/research/interpreting-lm-parameters";
const SPD_PAPER = "https://arxiv.org/abs/2506.20790";

// In reading order. `short` is the hover text; `long` is the glossary entry.
const GLOSSARY = [
  {
    key: "decomposition", term: "Parameter decomposition",
    short: "A method that splits a model's weight matrices into many simple pieces, only a few of which are needed on any one token.",
    long: `<p>Most interpretability methods decompose a model's activations; sparse autoencoders are the
      best-known example. Parameter decomposition decomposes the <em>weights</em> instead. It writes each
      weight matrix as a sum of many rank-one pieces, called components. Training pushes for as few
      components as possible to be needed on each token, while the model's output stays the same.</p>
      <p>For more, see the Goodfire post <a href="${BLOG}">Interpreting Language Model Parameters</a>
      and the paper <a href="${SPD_PAPER}">Stochastic Parameter Decomposition</a>.</p>`,
  },
  {
    key: "component", term: "Component",
    short: "One rank-one piece u·vᵀ of a weight matrix. Here, each layer's MLP down-projection is split into 6,144 of them.",
    long: `<p>A rank-one matrix <code>u·vᵀ</code>. Given an input vector <code>x</code>, it reads one direction
      <code>v</code> of the input and writes along one direction <code>u</code> of the output. The decomposition
      on this site splits the down-projection of each of Qwen3-1.7B's MLPs into 6,144 components; this site shows layers 0, 1 and 15. Together with a
      small leftover term, they add up to the original weight matrix. On any given token, most components can be
      removed without changing what the model does.</p>`,
  },
  {
    key: "activation", term: "Component activation",
    short: "How much a component contributes on a token: the number (x·v)·‖u‖, where x is the matrix's input. It can be positive or negative.",
    long: `<p>A component's output on a token is <code>u·(v·x)</code>, a vector along <code>u</code>. Its activation is
      the signed length of that vector, <code>(x·v)·‖u‖</code>. Multiplying by <code>‖u‖</code> means the number
      doesn't depend on how the decomposition happens to split scale between <code>u</code> and <code>v</code>.
      Unlike a neuron after a ReLU, it can be negative.</p>`,
  },
  {
    key: "ci", term: "Causal importance",
    short: "The decomposition's own estimate, between 0 and 1, of whether a component is needed on a token. Underlined tokens have causal importance above 0.1.",
    long: `<p>Parameter decomposition trains a small network next to the components. For every component and token,
      it outputs a value between 0 and 1. Near 0 means the component can be removed on that token without changing
      the model's output. Near 1 means it can't. Training pushes these values toward 0, which is why most components
      are inactive on most tokens. On the example text, tokens where a component's causal importance is above 0.1
      are underlined.</p>`,
  },
  {
    key: "program", term: "Program",
    short: "A short Python program, written by a language-model agent, that predicts one component's activation from the text alone. The program is the explanation.",
    long: `<p>A program reads only the text, never the model's internals, and predicts the component's activation at
      each token. It is the explanation of that component: if programs capture what the components do, the model
      should still work when the programs replace them. Language-model agents (Sol, Haiku 5.5 and Luna; which ones varies by layer) each wrote one
      program per component, after studying example text.</p>
      <p>A program defines <code>step(carry, prefix, upstream)</code>. The harness calls it once per token, left to right:</p>
      <ul>
        <li><code>prefix</code> is the text up to and including the current token: token strings and ids, decoded
          text, and which document each token belongs to (training sequences pack several documents together).</li>
        <li><code>carry</code> is whatever the program returned at the previous token (<code>None</code> at the first),
          so it can keep running state without re-reading the whole prefix.</li>
        <li><code>upstream</code> holds predictions for other components the program lists in <code>READS</code>.
          No program on this site reads any, so <code>READS = ()</code> everywhere.</li>
      </ul>
      <p><code>step</code> returns the new carry and the prediction. Programs may import the standard library, numpy
      and a few table libraries, and run in a sandbox with no file or network access.</p>`,
  },
  {
    key: "baselines", term: "Baselines",
    short: "Predictors that aren't programs. Token lookup predicts each component's average activation for the current token; token lookup+ does the same but leaves each document's first token unchanged. Zeroing removes the component.",
    long: `<p><b>Token lookup</b> predicts, for each component, its average activation on the current token, measured
      on training text. It knows nothing about context. <b>Zeroing</b> predicts 0 everywhere, which removes the
      component. Applied to every component at once, zeroing removes the whole down-projection's output.</p>
      <p><b>Token lookup+</b> is the same table fit without each document's first token, which instead keeps the
      component's true activation. The first token of a document is where early layers write large values that the rest
      of the document relies on, and a context-free average there is very costly. Its whole-layer scores were measured
      with documents' first tokens left unchanged, so compare them with programs in nats rather than against the
      zeroing line.</p>`,
  },
  {
    key: "replace", term: "Replacing a component",
    short: "Running the model with the component's output set to its own direction u, scaled to the predicted activation instead of the true one.",
    long: `<p>To test a predictor, the model runs with the component's true activation swapped for the predicted one.
      The output keeps the component's direction <code>u</code>; only its size changes. Everything else runs as
      normal. The scores then compare the next-token predictions of this modified model with the original's.</p>
      <p><b>All at once</b> replaces all 6,144 components together. This is the real test: it shows whether the
      programs, taken together, can stand in for the layer. <b>One at a time</b> replaces a single component and
      averages the scores over components. Each component is a tiny part of the model, so these scores are tiny
      too, around 10⁻⁵ nats/token. They are useful for comparing predictors on the same component, not as
      absolute numbers.</p>`,
  },
  {
    key: "kl", term: "KL",
    short: "KL divergence of the modified model's next-token distribution from the original's, in nats per token, averaged over positions. Lower is better; 0 means unchanged.",
    long: `<p>At each token position, the KL divergence from the original model's next-token distribution to the
      modified model's, averaged over positions. It is measured in nats (natural-log units) per token. 0 means the
      modified model predicts exactly as the original does.</p>`,
  },
  {
    key: "dce", term: "ΔCE",
    short: "How much the modified model's next-token loss (cross-entropy) rises above the original's, in nats per token. Lower is better.",
    long: `<p>The modified model's average next-token cross-entropy loss minus the original's, in nats per token. Unlike
      KL, it only looks at the probability given to the actual next token, so it can be slightly negative by chance.</p>`,
  },
  {
    key: "ve", term: "VE (variance explained)",
    short: "The share of the variance of a component's true activation that the prediction explains. 1 is perfect; 0 is no better than always predicting the mean.",
    long: `<p><code>1 − mean((prediction − activation)²) / variance(activation)</code>, over held-out positions. 1 is a
      perfect prediction, and 0 is no better than predicting the component's mean activation everywhere. It is
      negative for predictions worse than that. Zeroing has no VE.</p>`,
  },
  {
    key: "heldout", term: "Held-out text",
    short: "Text that wasn't used to write the programs or fit the baselines. All scores use it.",
    long: `<p>Agents studied examples from the training data, and token lookup was fit on it. Scores use separate text
      sequences that neither ever saw. Each sequence is about 500 tokens long.</p>`,
  },
  {
    key: "window", term: "Example window",
    short: "A snippet of training text around one token, coloured by the component's activation.",
    long: `<p>A short snippet of training text centred on one token, with each token coloured by the component's
      activation there: red for positive, blue for negative. <b>Strongest</b> windows are centred on the tokens
      with the largest activations. <b>Weaker</b> windows are spread over the rest of the range. Agents studied
      windows like these when writing programs.</p>`,
  },
];
const GLOSSED = new Map(GLOSSARY.map((g) => [g.key, g]));
// A glossary term in running text: hover for the short definition, click for the entry.
const term = (key, text) => {
  if (!GLOSSED.has(key)) throw new Error(`no glossary entry ${key}`);
  return `<a class="term" href="#/glossary/${key}" data-gloss="${key}">${text}</a>`;
};

const cache = new Map();
// A single-file preview embeds its data files instead of fetching them.
const EMBEDDED = window.EMBEDDED_DATA;

// A preview split into several files says which file holds a page.
const META = window.EMBEDDED_META;
function elsewhere(path) {
  // Which file holds a layer's index or a shard: files are keyed "<layer>:<first shard>:<end shard>".
  const m = path.match(/^data\/l(\d+)\/(?:s(\d+)|index)/);
  if (META && m) {
    const shard = m[2] === undefined ? 0 : Number(m[2]);
    const hit = Object.entries(META.files).find(([k]) => {
      const [L, a, b] = k.split(":").map(Number);
      return L === Number(m[1]) && shard >= a && shard < b;
    });
    if (hit) return `That page is in another file: ${hit[1]}`;
  }
  return `${path} is not in this file`;
}
// The single-file build embeds every data file gzipped and base64-encoded, decompressed on first use.
async function inflate(b64) {
  const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}
function load(path) {
  if (!cache.has(path)) {
    const loaded = EMBEDDED
      ? (path in EMBEDDED ? inflate(EMBEDDED[path]) : Promise.reject(new Error(elsewhere(path))))
      : fetch(path).then((r) => {
        if (!r.ok) throw new Error(`${path}: ${r.status}`);
        return r.json();
      });
    cache.set(path, loaded);
  }
  return cache.get(path);
}
const shardOf = (L, i) => load(`data/l${L}/s${String(Math.floor(i / SHARD)).padStart(3, "0")}.json`)
  .then((units) => units[i % SHARD]);

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const missing = (x) => x === null || x === undefined || Number.isNaN(x);
const fmt = (x, d = 2) => (missing(x) ? "—" : Number(x).toFixed(d));
// One-at-a-time KL and ΔCE are tiny per-token averages, so they read best in scientific notation.
const sci = (x) => (missing(x) ? "—" : x === 0 ? "0" : Number(x).toExponential(2).replace("e-", "e−"));
const metricText = (m, x) => (m === 2 ? fmt(x, 3) : sci(x));
const int = (x) => Number(x).toLocaleString("en-US");
const app = document.getElementById("app");

// ---------- router ----------
// A split preview puts each route's data in its own small file: the overview,
// each layer's list, and each 64-component shard's pages. A route this file doesn't hold is opened in
// the file that does, so every in-page link and hash change works unchanged.
function fileFor(parts) {
  if (parts[0] === "l" && /^\d+$/.test(parts[1] || "")) return `l${parts[1]}.html`;
  if (parts[0] === "c" && /^\d+$/.test(parts[1] || "") && /^\d+$/.test(parts[2] || ""))
    return `l${parts[1]}-s${String(Math.floor(Number(parts[2]) / SHARD)).padStart(3, "0")}.html`;
  return "index.html";  // the overview and the glossary
}
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (META && META.split && fileFor(parts) !== META.file) {
    location.href = `${fileFor(parts)}${location.hash || "#/"}`;
    return;
  }
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.remove("on"));
  const nav = (k) => document.querySelector(`[data-nav="${k}"]`)?.classList.add("on");
  window.scrollTo(0, 0);
  app.onclick = null;
  app.classList.remove("wide");
  const fail = (e) => {
    const text = EMBEDDED ? e.message : `Could not load data: ${e.message}`;
    app.innerHTML = `<p class="muted">${esc(text)}</p><p><a href="#/">Back to the overview</a></p>`;
  };
  if (parts[0] === "glossary") {
    nav("glossary");
    try { glossary(parts[1]); } catch (e) { fail(e); }
  } else if (parts[0] === "l" && /^\d+$/.test(parts[1] || "")) {
    nav(`l${parts[1]}`);
    browser(Number(parts[1])).catch(fail);
  } else if (parts[0] === "c" && /^\d+$/.test(parts[1] || "") && /^\d+$/.test(parts[2] || "")) {
    nav(`l${parts[1]}`);
    unitPage(Number(parts[1]), Number(parts[2])).catch(fail);
  } else {
    nav("overview");
    overview().catch(fail);
  }
}
window.addEventListener("hashchange", route);

// ---------- overview ----------
// A bar chart of joint KL and ΔCE, one row per arm, with a dashed line at `zero`'s values.
function chart(arms, zero) {
  const max = Math.max(...arms.flatMap((a) => [a.kl, a.ce_difference])) * 1.12;
  const cell = (a, key, unitName) => {
    const v = a[key];
    const tip = `${a.label.replace(/<[^>]+>/g, " ")}\n${unitName}: ${fmt(v, 3)} nats/token`;
    return `<div class="bar-cell" data-tip="${esc(tip)}">
      ${zero ? `<div class="zero-line" style="left:${(zero[key] / max) * 100}%"></div>` : ""}
      <div class="bar" style="width:${(v / max) * 100}%;background:${a.kind === "programs" ? "var(--components)" : "var(--baseline)"}"></div>
      <div class="val" style="left:${(v / max) * 100}%">${fmt(v, 3)}</div></div>`;
  };
  return `<div class="chart">
      <div></div><div class="head">KL to the original model (nats/token)</div><div class="head">Increase in loss, ΔCE (nats/token)</div>
      ${arms.map((a) => `<div class="label">${a.label}</div>${cell(a, "kl", "KL")}${cell(a, "ce_difference", "ΔCE")}`).join("")}
    </div>`;
}
const layerName = (L) => `Layer ${L}`;

// A layer's whole-layer bar chart: its programs, the baselines and zeroing, all components substituted at once
// (the overview's chart, and the top of the layer's page).
const LEGEND = `<div class="legend"><span><i class="swatch" style="background:var(--components)"></i>programs</span>
      <span><i class="swatch" style="background:var(--baseline)"></i>baselines</span></div>`;
function layerChart(s, l) {
  const order = [...l.models, "lookup", "zero"];
  const arms = order.map((id) => ({ ...l.joint.arms[id], label: id === "zero" ? `Zero the layer<span class="sub">baseline</span>`
    : `${MODELS[id].name}<span class="sub">${l.joint.arms[id].kind}${l.runs && l.runs[id] ? ` · <span class="run">${l.runs[id].map(esc).join(", ")}</span>` : ""}</span>` }));
  // Token lookup+, beside token lookup: scored with documents' first tokens unchanged.
  const plus = s.matched && s.matched.layers.find((x) => x.layer === l.layer);
  if (plus) arms.splice(order.indexOf("zero"), 0, { ...plus.arms.lookup_plus, label: `${MODELS.lookup_plus.name}<span class="sub">baseline · documents' first tokens unchanged</span>` });
  return chart(arms, l.joint.arms.zero);
}

async function overview() {
  const s = await load("data/summary.json");
  const layerSection = (l) => `<h3>${layerName(l.layer)} <span class="muted small">· <a href="#/l/${l.layer}">${int(s.n_units)} components</a> · ${esc(l.joint.precision)}</span></h3>
      ${layerChart(s, l)}`;
  const c = s.combined.arms;
  const comb = [
    ["sol_both", "Sol, layers 0 and 1", "programs"], ["sol_l0", "Sol, layer 0 only", "programs"], ["sol_l1", "Sol, layer 1 only", "programs"],
    ["both", "Haiku, layers 0 and 1", "programs"], ["l0", "Haiku, layer 0 only", "programs"], ["l1", "Haiku, layer 1 only", "programs"],
    ["lookup", "Token lookup, both layers", "baseline"], ["zero_l0", "Zero layer 0", "baseline"], ["zero_l1", "Zero layer 1", "baseline"],
    ["zero", "Zero both layers", "baseline"],
  ].map(([id, name, kind]) => ({ ...c[id], kind, label: `${name}<span class="sub">${kind}</span>` }));
  if (s.matched) comb.splice(comb.findIndex((a) => a.label.startsWith("Token lookup,")) + 1, 0,
    { ...s.matched.combined.arms.lookup_plus, label: `Token lookup+, both layers<span class="sub">baseline · documents' first tokens unchanged</span>` });
  const soloTable = (l) => `<h3>${layerName(l.layer)} <span class="muted small">· ${esc(l.solo_precision)}</span></h3>
    <table class="data scores">
      <tr><th class="l">predictor</th>${METRICS.map((m) => `<th title="${esc(m.tip)}">mean ${m.name}</th>`).join("")}<th title="mean lines of program source, over components with a program">mean lines</th><th title="mean size of program source, UTF-8, 1 KB = 1000 bytes">mean KB</th></tr>
      ${[...l.models, "lookup", "zero"].map((m) => {
        const code = l.code && l.code[m];
        return `<tr><td class="l">${m === "zero" ? "Zero the component" : MODELS[m].name}</td>${l.solo_means[m].map((v, k) => `<td>${metricText(k, v)}</td>`).join("")}<td>${code ? fmt(code.mean_lines, 0) : ""}</td><td>${code ? fmt(code.mean_kb, 1) : ""}</td></tr>`;
      }).join("")}
    </table>`;
  app.innerHTML = `
    <h1>Agent-written programs for Qwen3-1.7B MLP components</h1>
    <p class="lede">${term("decomposition", "Parameter decomposition")} (run p-82af68dd, step 100,000) splits each layer's MLP down-projection into
    6,144 rank-one components. Language-model agents each wrote a short Python ${term("program", "program")} that predicts one component's
    activation from the text. Layers: ${s.layers.map((l) => `<a href="#/l/${l.layer}">${layerName(l.layer)}</a>`).join(", ")}.</p>

    <h2>Replacing a whole layer at once</h2>
    <p class="muted small prose">Every component's program substituted together, on ${int(4096)} held-out rows. Lower is better.
    The dashed line is zeroing the whole layer. A component with no program is zeroed.</p>
    ${s.layers.map(layerSection).join("")}
    ${LEGEND}

    <h2>Layers 0 and 1 together</h2>
    <p class="muted small prose">Both layers' down-projections substituted at once, on the same rows (TF32). The dashed line
    is zeroing both layers. Program errors add almost linearly; zeroing compounds.</p>
    ${chart(comb, c.zero)}
    ${LEGEND}

    <h2>One component at a time</h2>
    <p class="muted small prose">Replacing a single component by its predictor, on 1,024 held-out rows, averaged over the
    6,144 components. <b>KL</b> and <b>ΔCE</b> in nats/token; <b>VE</b> is the variance of the component's activation the
    predictor explains. Layer 0 was scored in strict fp32; layers 1 and 15 in TF32 with its noise floor subtracted from KL.</p>
    ${s.layers.map(soloTable).join("")}

    <h2>Notes</h2>
    <div class="prose">
      <p><b>Programs.</b> Each program defines <code>step(carry, prefix, upstream)</code>. It sees the tokens so far and
      returns a prediction for the current position. The docstring is the agent's own one-line description.
      The token lookup baseline predicts each component's mean activation for the current token.</p>
      <p><b>Token lookup+.</b> The token lookup table fit without each document's first token, which keeps its true
      value. Its whole-layer bars were measured with documents' first tokens unchanged, no leftover weight term and full
      fp32 precision, on the same held-out text. For programs those differences are small (Sol, layer 0: KL 0.139 under
      this site's rules, 0.138 under lookup+'s), but zeroing with documents' first tokens unchanged costs far less (layer 0:
      0.69 against 2.83), so lookup+'s bars are best read in nats, not against the dashed line. There are no one-at-a-time
      scores for it.</p>
      <p><b>Examples.</b> Text windows from a sample of the training data. Windows that include any text from a few
      sources we can't share (such as chat logs, video transcripts, legal documents and library books) are left out,
      before examples are chosen.</p>
      <p><b>Layer 1.</b> Component 6136 has no Haiku 5.5 program.</p>
    </div>
    <h3>Models</h3>
    <table class="data scores">
      ${["sol", "haiku", "luna"].map((m) => `<tr><td class="l">${MODELS[m].name}</td><td class="l">${MODELS[m].full}</td></tr>`).join("")}
    </table>
    <footer>Model: ${esc(s.model)} · decomposition ${esc(s.decomposition.run)} at step ${int(s.decomposition.step)}.</footer>`;
}

function glossary(focus) {
  app.innerHTML = `
    <h1>Glossary</h1>
    <p class="muted prose">The terms this site uses, in the order the ideas build on each other.</p>
    <dl class="glossary prose">${GLOSSARY.map((g) => `
      <dt id="g-${g.key}">${g.term}</dt><dd>${g.long}</dd>`).join("")}
    </dl>`;
  if (focus) {
    const entry = document.getElementById(`g-${focus}`);
    if (!entry) throw new Error(`no glossary entry ${focus}`);
    entry.classList.add("focus");
    entry.scrollIntoView({ block: "start" });
  }
}

// ---------- browser ----------
const st = { sort: "i", asc: true, q: "", page: 0, desc: "sol", metric: 0 };

async function browser(L) {
  const [idx, s] = await Promise.all([load(`data/l${L}/index.json`), load("data/summary.json")]);
  const layer = s.layers.find((x) => x.layer === L);
  PROGRAM_MODELS = idx.models;
  if (!PROGRAM_MODELS.includes(st.desc)) st.desc = PROGRAM_MODELS[0];
  if (st.sort !== "i" && ![...PROGRAM_MODELS, "lookup"].includes(st.sort)) { st.sort = "i"; st.asc = true; }
  if (st.layer !== L) { st.page = 0; st.layer = L; }
  const scoreModels = [...PROGRAM_MODELS, "lookup"];
  // A model's column can name the runs its programs came from (idx.runs), where the data has them.
  const runsOf = (m) => (idx.runs && idx.runs[m] ? `<span class="run">${idx.runs[m].map(esc).join(", ")}</span>` : "");
  const cols = [{ key: "i", label: "#" }, ...scoreModels.map((m) => ({ key: m, label: `${MODELS[m].name}${runsOf(m) ? `<br>${runsOf(m)}` : ""}` }))];
  const value = (u, key) => {
    if (key === "i") return u.i;
    const v = u.s[key][st.metric];
    return v === null ? NaN : v;
  };

  app.innerHTML = `
    <h1>Layer-${L} MLP components</h1>
    <h2>Replacing the whole layer at once</h2>
    <p class="muted small prose">Every component's program substituted together, on ${int(layer.joint.rows)} held-out rows (${esc(layer.joint.precision)}). Lower is better.
    The dashed line is zeroing the whole layer.</p>
    ${layerChart(s, layer)}
    ${LEGEND}
    <h2>Components</h2>
    <p class="muted prose">${int(idx.units.length)} components. Scores replace one component at a time. Click a column to sort
    and a row to open the component.</p>
    <div class="controls">
      <span class="metric">${METRICS.map((m, k) => `<button data-metric="${k}" title="${esc(m.tip)}"${k === st.metric ? ' aria-pressed="true"' : ""}>${m.name}</button>`).join("")}</span>
      <input type="search" placeholder="Search program descriptions or #index" value="${esc(st.q)}">
      <label class="small muted">Description from
        <select>${PROGRAM_MODELS.map((m) => `<option value="${m}"${m === st.desc ? " selected" : ""}>${MODELS[m].name}</option>`).join("")}</select></label>
    </div>
    <div class="pager top-pager"></div>
    <table class="data"><thead><tr>${cols.map((c) => `<th class="sortable" data-key="${c.key}">${c.label}</th>`).join("")}<th class="l">description</th></tr></thead>
    <tbody></tbody></table>
    <div class="pager bottom-pager"></div>`;

  const tbody = app.querySelector("tbody");
  const search = app.querySelector("input[type=search]");
  const select = app.querySelector("select");

  function render() {
    const q = st.q.trim().toLowerCase();
    let units = idx.units;
    if (q) {
      const asIndex = q.replace(/^#/, "");
      units = units.filter((u) => String(u.i) === asIndex || Object.values(u.d).some((d) => d.toLowerCase().includes(q)));
    }
    const dir = st.asc ? 1 : -1;
    units = [...units].sort((a, b) => {
      const x = value(a, st.sort), y = value(b, st.sort);
      if (Number.isNaN(x)) return 1;
      if (Number.isNaN(y)) return -1;
      return (x - y) * dir || a.i - b.i;
    });
    const pages = Math.max(1, Math.ceil(units.length / PAGE));
    st.page = Math.min(st.page, pages - 1);
    const shown = units.slice(st.page * PAGE, (st.page + 1) * PAGE);
    const better = METRICS[st.metric].lowerBetter ? Math.min : Math.max;
    tbody.innerHTML = shown.map((u) => {
      const best = better(...scoreModels.map((m) => value(u, m)).filter((v) => !Number.isNaN(v)));
      return `<tr class="row" data-i="${u.i}">${cols.map((c) => {
        const v = value(u, c.key);
        if (c.key === "i") return `<td>${v}</td>`;
        return `<td class="${v === best ? "best" : ""}">${metricText(st.metric, v)}</td>`;
      }).join("")}<td class="l desc">${esc(u.d[st.desc] || "")}</td></tr>`;
    }).join("");
    app.querySelectorAll("th.sortable").forEach((th) => {
      th.classList.toggle("sorted", th.dataset.key === st.sort);
      th.classList.toggle("asc", th.dataset.key === st.sort && st.asc);
    });
    const pager = `<button data-p="-1"${st.page === 0 ? " disabled" : ""}>Previous</button>
      <span>${int(units.length)} shown · page ${st.page + 1} of ${pages}</span>
      <button data-p="1"${st.page >= pages - 1 ? " disabled" : ""}>Next</button>`;
    app.querySelectorAll(".pager").forEach((p) => { p.innerHTML = pager; });
  }

  search.addEventListener("input", () => { st.q = search.value; st.page = 0; render(); });
  select.addEventListener("change", () => { st.desc = select.value; render(); });
  app.querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => {
    if (st.sort === th.dataset.key) st.asc = !st.asc;
    else { st.sort = th.dataset.key; st.asc = th.dataset.key === "i" || METRICS[st.metric].lowerBetter; }
    st.page = 0;
    render();
  }));
  app.onclick = (e) => {
    const metric = e.target.closest("button[data-metric]");
    if (metric) {
      st.metric = Number(metric.dataset.metric);
      if (st.sort !== "i") st.asc = METRICS[st.metric].lowerBetter;
      app.querySelectorAll("button[data-metric]").forEach((x) => x.toggleAttribute("aria-pressed", x === metric));
      render();
      return;
    }
    const b = e.target.closest(".pager button");
    if (b) { st.page += Number(b.dataset.p); render(); return; }
    const row = e.target.closest("tr.row");
    if (row) location.hash = `#/c/${L}/${row.dataset.i}`;
  };
  render();
}

// ---------- component pages ----------
function tokenSpan(text, value, ci, centre) {
  // Diverging colour: red for positive activations, blue for negative. Alpha grows as the square
  // root of |value| (relative to the component's largest shown activation) so weak activity stays visible.
  const a = Math.sqrt(Math.min(1, Math.abs(value)));
  const rgb = value >= 0 ? "var(--pos)" : "var(--neg)";
  const shown = esc(text).replace(/\n/g, '<span class="ws">↵</span>').replace(/\t/g, '<span class="ws">⇥</span>');
  const cls = `tok${ci !== null && ci > 0.1 ? " fire" : ""}${centre ? " centre" : ""}`;
  return `<span class="${cls}" style="background:rgba(${rgb},${(a * 0.85).toFixed(3)})" data-v="${value}" data-ci="${ci ?? ""}" data-t="${esc(text)}">${shown}</span>`;
}

function windowHtml(w, vocab, scale) {
  const parts = w.t.map((id, k) => {
    const v = w.a[k] / 127;
    const ci = w.ci ? w.ci[k] / 100 : null;
    return tokenSpan(vocab[id] ?? `⟨${id}⟩`, v, ci, k === w.c);
  });
  return `<div class="ex" data-scale="${scale}">${parts.join("")}</div>`;
}

function scaleLegend(scale) {
  return `<span class="scale">−${fmt(scale)}
    <span class="ramp" style="background:linear-gradient(90deg, rgba(var(--neg),.85), rgba(var(--neg),0) 50%, rgba(var(--pos),0) 50%, rgba(var(--pos),.85))"></span>
    +${fmt(scale)}</span>`;
}

function highlightPython(src) {
  const KW = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "));
  const re = /("""[\s\S]*?"""|'''[\s\S]*?'''|[rbfu]{0,2}"(?:\\.|[^"\\\n])*"|[rbfu]{0,2}'(?:\\.|[^'\\\n])*'|#[^\n]*|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b|\b[A-Za-z_]\w*\b)/g;
  let out = "", last = 0, m, prevDef = false;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    const t = m[0];
    let cls = null;
    if (/^[rbfu]{0,2}["']/.test(t)) cls = "st";
    else if (t[0] === "#") cls = "co";
    else if (/^\d/.test(t)) cls = "nu";
    else if (KW.has(t)) cls = "kw";
    else if (prevDef) cls = "fn";
    prevDef = t === "def" || t === "class";
    out += cls ? `<span class="${cls}">${esc(t)}</span>` : esc(t);
    last = m.index + t.length;
  }
  return out + esc(src.slice(last));
}

function scoresTable(unit, models) {
  const name = (m) => (m === "zero" ? "Zero the component" : MODELS[m].name) + (m === "lookup" || m === "zero" ? ' <span class="muted">(baseline)</span>' : "");
  return `<table class="data scores">
    <tr><th class="l">predictor</th>${METRICS.map((m) => `<th title="${esc(m.tip)}">${m.name}</th>`).join("")}</tr>
    ${models.map((m) => `<tr><td class="l">${name(m)}</td>${unit.scores[m].map((v, k) => `<td>${metricText(k, v)}</td>`).join("")}</tr>`).join("")}
  </table>
  <p class="muted small">KL and ΔCE in nats/token; VE is the share of the component's activation variance the predictor explains.</p>`;
}

// Storage is unavailable in sandboxed frames; the layout choice then lasts only for the visit.
const remembered = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* not remembered */ } },
};
let stacked = remembered.get("programs-stacked") === "1";

function programsHtml(unit, models) {
  const toggle = models.length > 1
    ? `<div class="controls"><button class="layout" data-stacked="0"${stacked ? "" : ' aria-pressed="true"'}>Side by side</button>
       <button class="layout" data-stacked="1"${stacked ? ' aria-pressed="true"' : ""}>One per row</button></div>`
    : "";
  return `${toggle}<div class="programs${stacked ? " stacked" : ""}">${models.map((m) => {
    const src = unit.programs[m];
    const sc = unit.scores[m];
    const note = unit.notes && unit.notes[m] ? ` <span class="muted">(${esc(unit.notes[m])})</span>` : "";
    const run = unit.runs && unit.runs[m] ? ` <span class="run" title="the explain run that wrote this program">${esc(unit.runs[m])}</span>` : "";
    return `<div class="program"><div class="ph"><span><b>${MODELS[m].name}</b>${run}${note}</span>
      <span class="muted">KL ${sci(sc[0])} · ΔCE ${sci(sc[1])} · VE ${fmt(sc[2])}${src ? ` · ${src.split("\n").length} lines` : ""}</span></div>
      <pre>${src ? highlightPython(src) : '<span class="muted">No program.</span>'}</pre></div>`;
  }).join("")}</div>`;
}

async function unitPage(L, i) {
  if (i < 0 || i >= 6144) throw new Error(`no component ${i}`);
  const [idx, unit, vocab] = await Promise.all([load(`data/l${L}/index.json`), shardOf(L, i), load("data/vocab.json")]);
  PROGRAM_MODELS = idx.models;
  const nav = `<div class="unit-nav">
    ${i > 0 ? `<a href="#/c/${L}/${i - 1}">← ${i - 1}</a>` : ""}
    <a href="#/l/${L}">all layer-${L} components</a>
    ${i < 6143 ? `<a href="#/c/${L}/${i + 1}">${i + 1} →</a>` : ""}</div>`;
  const groups = [["strongest", "Strongest windows, by activation at the marked token"], ["spread", "Weaker windows, spread over the rest of the sample"]];
  const examples = groups.map(([g, title]) => {
    const ws = unit.windows.filter((w) => w.g === g);
    return ws.length ? `<div class="ex-group">${title}</div><div class="examples">${ws.map((w) => windowHtml(w, vocab, unit.scale)).join("")}</div>` : "";
  }).join("");
  const body = `
    <h2>Programs</h2>${programsHtml(unit, PROGRAM_MODELS)}
    <h2>Scores, one component at a time</h2>${scoresTable(unit, [...PROGRAM_MODELS, "lookup", "zero"])}
    <h2>Examples</h2>
    <p class="muted small">Text windows sampled from the training data where the component is active, from sources we can share. Colour is the component's activation, red positive and blue negative${unit.scale ? `; ${scaleLegend(unit.scale)}` : ""}.
    An underline marks tokens where the decomposition counts the component as active (causal importance above 0.1). The outlined token is the window's centre. Hover a token for values.</p>
    ${examples || `<p class="muted">${unit.removed_all ? "Every example window for this component included text from a source we can't share, so none are shown." : "This component wasn't active anywhere in the sampled text."}</p>`}`;
  app.onclick = (e) => {
    const b = e.target.closest("button.layout");
    if (!b) return;
    stacked = b.dataset.stacked === "1";
    remembered.set("programs-stacked", stacked ? "1" : "0");
    app.querySelector(".programs").classList.toggle("stacked", stacked);
    app.querySelectorAll("button.layout").forEach((x) => x.toggleAttribute("aria-pressed", x === b));
  };
  app.classList.add("wide");
  app.innerHTML = `<div class="unit-head"><h1>Layer ${L}, component ${i}</h1>${nav}</div>${body}`;
}

// ---------- tooltips ----------
const tip = document.getElementById("tip");
document.addEventListener("mouseover", (e) => {
  const tok = e.target.closest(".tok");
  const bar = e.target.closest("[data-tip]");
  const gloss = e.target.closest("[data-gloss]");
  if (gloss) {
    tip.textContent = GLOSSED.get(gloss.dataset.gloss).short;
  } else if (tok) {
    const scale = Number(tok.closest(".ex").dataset.scale);
    const ci = tok.dataset.ci;
    tip.textContent = `${JSON.stringify(tok.dataset.t)}\nactivation ${fmt(Number(tok.dataset.v) * scale, 3)}${ci !== "" ? `\ncausal importance ${fmt(Number(ci), 2)}` : ""}`;
  } else if (bar) {
    tip.textContent = bar.dataset.tip;
  } else {
    tip.hidden = true;
    return;
  }
  tip.hidden = false;
});
document.addEventListener("mousemove", (e) => {
  if (tip.hidden) return;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = e.clientY + 18 + tip.offsetHeight > window.innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 18;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
});

route();

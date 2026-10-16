"use strict";
// Static viewer for agent-written programs of Qwen3-1.7B's layer-0 MLP components. Routes:
// #/ overview, #/components browser, #/c/<i> component pages, #/glossary[/<key>].

const SHARD = 64;
const PAGE = 100;
const MODELS = {
  sol: { name: "Sol", full: "OpenAI gpt-6.1-sol, high reasoning effort" },
  haiku: { name: "Haiku 5.5", full: "Anthropic claude-haiku-5-5, high effort" },
  luna: { name: "Luna", full: "OpenAI gpt-6-luna, high reasoning effort" },
  lookup: { name: "Token lookup" },
  zero: { name: "Zeroing" },
};
// One-at-a-time scores, in the order the data stores them.
const METRICS = [
  { name: "KL", gloss: "kl", lowerBetter: true },
  { name: "ΔCE", gloss: "dce", lowerBetter: true },
  { name: "VE", gloss: "ve", lowerBetter: false },
];
const PROGRAM_MODELS = ["sol", "haiku", "luna"];

// Training-data sources whose text the site doesn't show.
const EXCLUDED = "text from chat logs, video transcripts, legal documents and library books";

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
    short: "One rank-one piece u·vᵀ of a weight matrix. Here, the layer-0 MLP down-projection is split into 6,144 of them.",
    long: `<p>A rank-one matrix <code>u·vᵀ</code>. Given an input vector <code>x</code>, it reads one direction
      <code>v</code> of the input and writes along one direction <code>u</code> of the output. The decomposition
      on this site splits the down-projection of Qwen3-1.7B's first MLP into 6,144 components. Together with a
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
      should still work when the programs replace them. Three agents (Sol, Haiku 5.5 and Luna) each wrote one
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
    short: "Predictors that aren't programs. Token lookup predicts each component's average activation for the current token. Zeroing removes the component.",
    long: `<p><b>Token lookup</b> predicts, for each component, its average activation on the current token, measured
      on training text. It knows nothing about context. <b>Zeroing</b> predicts 0 everywhere, which removes the
      component. Applied to every component at once, zeroing removes the whole down-projection's output.</p>`,
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
// A single-file preview (preview_artifact.py) embeds some data files and can't fetch the rest.
const EMBEDDED = window.EMBEDDED_DATA;

function load(path) {
  if (!cache.has(path)) {
    const loaded = EMBEDDED
      ? (path in EMBEDDED
        ? Promise.resolve(EMBEDDED[path])
        : Promise.reject(new Error(`This page isn't in the single-file preview. ${EMBEDDED.note}`)))
      : fetch(path).then((r) => {
        if (!r.ok) throw new Error(`${path}: ${r.status}`);
        return r.json();
      });
    cache.set(path, loaded);
  }
  return cache.get(path);
}
const shardOf = (i) => load(`data/components/s${String(Math.floor(i / SHARD)).padStart(3, "0")}.json`)
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
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.remove("on"));
  const nav = (k) => document.querySelector(`[data-nav="${k}"]`)?.classList.add("on");
  window.scrollTo(0, 0);
  app.onclick = null;
  app.classList.remove("wide");
  const fail = (e) => {
    const text = EMBEDDED ? e.message : `Could not load data: ${e.message}`;
    app.innerHTML = `<p class="muted">${esc(text)}</p><p><a href="#/">Back to the overview</a></p>`;
  };
  if (parts[0] === "components") {
    nav("components");
    browser().catch(fail);
  } else if (parts[0] === "glossary") {
    nav("glossary");
    try { glossary(parts[1]); } catch (e) { fail(e); }
  } else if (parts[0] === "c" && /^\d+$/.test(parts[1] || "")) {
    nav("components");
    unitPage(Number(parts[1])).catch(fail);
  } else {
    nav("overview");
    overview().catch(fail);
  }
}
window.addEventListener("hashchange", route);

// ---------- overview ----------
async function overview() {
  const s = await load("data/summary.json");
  const arms = s.joint.arms;
  const label = (a) => {
    if (a.unit === "components") return `${MODELS[a.id].name}<span class="sub">programs</span>`;
    return `${a.id === "zero" ? "Zero the layer" : MODELS[a.id].name}<span class="sub">baseline</span>`;
  };
  const colour = (a) => (a.unit === "components" ? "var(--components)" : "var(--baseline)");
  const max = Math.max(...arms.flatMap((a) => [a.kl, a.ce_difference])) * 1.12;
  const zero = arms.find((a) => a.id === "zero");
  const cell = (a, key, unitName) => {
    const v = a[key];
    const tip = `${a.id === "zero" ? "Zero the layer" : MODELS[a.id].name}, ${a.unit === "components" ? "programs" : "baseline"}\n${unitName}: ${fmt(v, 3)} nats/token`;
    return `<div class="bar-cell" data-tip="${esc(tip)}">
      <div class="zero-line" style="left:${(zero[key] / max) * 100}%"></div>
      <div class="bar" style="width:${(v / max) * 100}%;background:${colour(a)}"></div>
      <div class="val" style="left:${(v / max) * 100}%">${fmt(v, 2)}</div></div>`;
  };
  const rows = arms.map((a) => `<div class="label">${label(a)}</div>${cell(a, "kl", "KL")}${cell(a, "ce_difference", "ΔCE")}`).join("");
  const soloRow = (m) => {
    const x = s.solo.means.components[m];
    const name = m === "zero" ? "Zero the component" : MODELS[m].name;
    return `<tr><td class="l">${name}</td>${x.map((v, k) => `<td>${metricText(k, v)}</td>`).join("")}</tr>`;
  };
  const best = arms.filter((a) => a.unit === "components").reduce((x, y) => (x.kl < y.kl ? x : y));
  app.innerHTML = `
    <h1>Agent-written programs for every layer-0 MLP component of Qwen3-1.7B</h1>
    <p class="lede">We split one weight matrix of Qwen3-1.7B into 6,144 simple pieces, then asked language-model
    agents to write a short Python program for each piece that predicts what it does from the text alone.
    With every piece replaced by its program, the model still mostly works.</p>

    <h2>How this works</h2>
    <ol class="prose steps">
      <li><b>Decompose the weights.</b> ${term("decomposition", "Parameter decomposition")} splits the down-projection
      of Qwen3-1.7B's first MLP layer into 6,144 rank-one ${term("component", "components")}. On any given token,
      only a few of them matter.</li>
      <li><b>Measure each component.</b> On every token, each component has an ${term("activation", "activation")}:
      a single number saying how strongly it writes its output direction.</li>
      <li><b>Write a program.</b> An agent studies ${term("window", "example text")} where a component is active, then
      writes a ${term("program", "program")} that predicts its activation from the text so far. The program never
      sees the model's internals, so it is a readable explanation of what the component responds to.</li>
      <li><b>Test it inside the model.</b> We ${term("replace", "replace")} each component's true activation with the
      program's prediction and check how much the model's next-token predictions change, on
      ${term("heldout", "held-out text")}. If the explanation is good, they barely change.</li>
    </ol>

    <h2>Replacing all 6,144 components at once</h2>
    <p class="muted small prose">Each bar is the model with every component replaced by one predictor, over
    ${int(s.joint.positions)} token positions. ${term("kl", "KL")} measures how far the model's next-token
    distribution moves from the original's; ${term("dce", "ΔCE")} measures how much its loss rises. Both are in nats
    per token, and lower is better. The dashed line is zeroing the whole layer's output, for scale.</p>
    <div class="chart">
      <div></div><div class="head">KL to the original model (nats/token)</div><div class="head">Increase in loss, ΔCE (nats/token)</div>
      ${rows}
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--components)"></i>programs</span>
      <span><i class="swatch" style="background:var(--baseline)"></i>${term("baselines", "baselines")}</span>
    </div>
    <p class="prose">The best programs (${MODELS[best.id].name}'s) raise the loss by ${fmt(best.ce_difference, 2)}
    nats/token, against ${fmt(zero.ce_difference, 2)} for removing the layer's output entirely.</p>

    <h2>One component at a time</h2>
    <p class="muted small prose">Here only one component is replaced at a time, on ${int(s.solo.rows)} held-out text
    sequences, and the scores are averaged over the 6,144 components. One component is a tiny part of the model,
    so KL and ΔCE are tiny too: compare them between rows, not against the chart above.
    ${term("ve", "VE")} is how much of the component's activation the prediction explains, where 1 is perfect.</p>
    <table class="data scores">
      <tr><th class="l">predictor</th>${METRICS.map((m) => `<th>mean ${term(m.gloss, m.name)}</th>`).join("")}</tr>
      ${[...PROGRAM_MODELS, "lookup", "zero"].map(soloRow).join("")}
    </table>

    <h2>What's here</h2>
    <div class="prose">
      <p><b><a href="#/components">Components</a>.</b> A sortable table of all 6,144 components, each with a
      one-line description from its program. Each component's page shows its programs, its scores, and example text
      coloured by its activation.</p>
      <p><b><a href="#/glossary">Glossary</a>.</b> Every term on the site, explained. Dotted-underlined words link to
      it; hover them for a short definition.</p>
      <p><b>Data.</b> Examples are drawn from the model's training mix of public datasets: web text, code, math,
      papers, Wikipedia, patents and agent coding traces. The site leaves out ${EXCLUDED}.</p>
    </div>
    <h3>Agents</h3>
    <table class="data scores">
      ${PROGRAM_MODELS.map((m) => `<tr><td class="l">${MODELS[m].name}</td><td class="l">${MODELS[m].full}</td></tr>`).join("")}
    </table>
    <footer>Model: ${esc(s.model)} · weight matrix: ${esc(s.site)} · decomposition: Goodfire run ${esc(s.decomposition.run)}, training step ${int(s.decomposition.step)}.</footer>`;
}

// ---------- glossary ----------
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

async function browser() {
  const idx = await load("data/components/index.json");
  const scoreModels = [...PROGRAM_MODELS, "lookup"];
  const cols = [{ key: "i", label: "#" }, ...scoreModels.map((m) => ({ key: m, label: MODELS[m].name }))];
  const value = (u, key) => {
    if (key === "i") return u.i;
    const v = u.s[key][st.metric];
    return v === null ? NaN : v;
  };

  app.innerHTML = `
    <h1>Layer-0 MLP components</h1>
    <p class="muted prose">All ${int(idx.units.length)} ${term("component", "components")}, with each agent's score for
    its program and the ${term("baselines", "token lookup baseline")}. Scores ${term("replace", "replace one component at a time")},
    so they are tiny; the best predictor in each row is bold. The description is the first line of a program's
    docstring. Click a column to sort and a row to open the component.</p>
    <div class="controls">
      <span class="metric">${METRICS.map((m, k) => `<button data-metric="${k}" title="${esc(GLOSSED.get(m.gloss).short)}"${k === st.metric ? ' aria-pressed="true"' : ""}>${m.name}</button>`).join("")}</span>
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
    if (row) location.hash = `#/c/${row.dataset.i}`;
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
    <tr><th class="l">predictor</th>${METRICS.map((m) => `<th>${term(m.gloss, m.name)}</th>`).join("")}</tr>
    ${models.map((m) => `<tr><td class="l">${name(m)}</td>${unit.scores[m].map((v, k) => `<td>${metricText(k, v)}</td>`).join("")}</tr>`).join("")}
  </table>
  <p class="muted small">Each row ${term("replace", "replaces only this component")}. KL and ΔCE are in nats/token, lower is
  better; VE is the share of the component's activation the prediction explains, higher is better.</p>`;
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
    return `<div class="program"><div class="ph"><b>${MODELS[m].name}</b>
      <span class="muted">KL ${sci(sc[0])} · ΔCE ${sci(sc[1])} · VE ${fmt(sc[2])}${src ? ` · ${src.split("\n").length} lines` : ""}</span></div>
      <pre>${src ? highlightPython(src) : '<span class="muted">No program.</span>'}</pre></div>`;
  }).join("")}</div>`;
}

async function unitPage(i) {
  if (i < 0 || i >= 6144) throw new Error(`no component ${i}`);
  const [unit, vocab] = await Promise.all([shardOf(i), load("data/vocab.json")]);
  const nav = `<div class="unit-nav">
    ${i > 0 ? `<a href="#/c/${i - 1}">← ${i - 1}</a>` : ""}
    <a href="#/components">all components</a>
    ${i < 6143 ? `<a href="#/c/${i + 1}">${i + 1} →</a>` : ""}</div>`;
  const groups = [["strongest", "Strongest windows, by activation at the marked token"], ["spread", "Weaker windows, spread over the rest of the sample"]];
  const examples = groups.map(([g, title]) => {
    const ws = unit.windows.filter((w) => w.g === g);
    return ws.length ? `<div class="ex-group">${title}</div><div class="examples">${ws.map((w) => windowHtml(w, vocab, unit.scale)).join("")}</div>` : "";
  }).join("");
  const body = `
    <h2>Programs</h2>
    <p class="muted small prose">Three agents each wrote a ${term("program", "program")} predicting this component's
    ${term("activation", "activation")} from the text. The harness calls <code>step</code> once per token; it returns
    a running state for the next token (<code>carry</code>) and its prediction. <code>prefix</code> is the text so far.</p>
    ${programsHtml(unit, PROGRAM_MODELS)}
    <h2>Scores, one component at a time</h2>${scoresTable(unit, [...PROGRAM_MODELS, "lookup", "zero"])}
    <h2>Examples</h2>
    <p class="muted small prose">${term("window", "Text windows")} sampled from the training data where the component is
    active, leaving out ${EXCLUDED}. Colour is the component's ${term("activation", "activation")}, red positive and blue
    negative${unit.scale ? `; ${scaleLegend(unit.scale)}` : ""}. An underline marks tokens where the decomposition
    counts the component as needed (${term("ci", "causal importance")} above 0.1). The outlined token is the window's
    centre. Hover a token for its values.</p>
    ${examples || `<p class="muted">${unit.harvested_windows
      ? `All ${unit.harvested_windows} sampled windows come from datasets whose text isn't shown here.`
      : "This component wasn't active anywhere in the sampled text."}</p>`}`;
  app.onclick = (e) => {
    const b = e.target.closest("button.layout");
    if (!b) return;
    stacked = b.dataset.stacked === "1";
    remembered.set("programs-stacked", stacked ? "1" : "0");
    app.querySelector(".programs").classList.toggle("stacked", stacked);
    app.querySelectorAll("button.layout").forEach((x) => x.toggleAttribute("aria-pressed", x === b));
  };
  app.classList.add("wide");
  app.innerHTML = `<div class="unit-head"><h1>Component ${i}</h1>${nav}</div>${body}`;
}

// ---------- tooltips ----------
const tip = document.getElementById("tip");
document.addEventListener("mouseover", (e) => {
  const tok = e.target.closest(".tok");
  const bar = e.target.closest("[data-tip]");
  const gloss = e.target.closest("[data-gloss]");
  tip.classList.toggle("prose-tip", Boolean(gloss));
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

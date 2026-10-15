"use strict";
// Static viewer for agent-written programs of Qwen3-1.7B's layer-0 MLP components. Routes:
// #/ overview, #/components browser, #/c/<i> component pages.

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
  { name: "KL", tip: "KL from the original model's next-token distribution, nats/token; lower is better", lowerBetter: true },
  { name: "ΔCE", tip: "change in next-token loss, nats/token; lower is better", lowerBetter: true },
  { name: "VE", tip: "variance of the component's activation the predictor explains; higher is better", lowerBetter: false },
];
const PROGRAM_MODELS = ["sol", "haiku", "luna"];

// Training-data sources whose text the site doesn't show.
const EXCLUDED = "text from chat logs, video transcripts, legal documents and library books";

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
  app.innerHTML = `
    <h1>Agent-written programs for every layer-0 MLP component of Qwen3-1.7B</h1>
    <p class="lede">Parameter decomposition splits the layer-0 MLP down-projection of Qwen3-1.7B into
    <a href="#/components">6,144 components</a>. Language-model agents each wrote a short Python program that
    predicts one component's activation from the text. Swapped into the model all at once, the programs keep it
    close to the original.</p>

    <h2>Replacing all 6,144 components at once</h2>
    <p class="muted small">Every component's program substituted together in layer 0, on ${int(s.joint.rows)} held-out
    rows (${int(s.joint.positions)} next-token positions). Lower is better. The dashed line is zeroing the whole layer.</p>
    <div class="chart">
      <div></div><div class="head">KL to the original model (nats/token)</div><div class="head">Increase in loss, ΔCE (nats/token)</div>
      ${rows}
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--components)"></i>programs</span>
      <span><i class="swatch" style="background:var(--baseline)"></i>baselines</span>
    </div>

    <h2>One component at a time</h2>
    <p class="muted small prose">Replacing a single component by its predictor, on ${int(s.solo.rows)} held-out rows, averaged over
    the 6,144 components. <b>KL</b> and <b>ΔCE</b> are in nats/token. <b>VE</b> is the variance of the component's activation the predictor explains.</p>
    <table class="data scores">
      <tr><th class="l">predictor</th>${METRICS.map((m) => `<th title="${esc(m.tip)}">mean ${m.name}</th>`).join("")}</tr>
      ${[...PROGRAM_MODELS, "lookup", "zero"].map(soloRow).join("")}
    </table>

    <h2>What's here</h2>
    <div class="prose">
      <p><b>Components.</b> The decomposition (run p-82af68dd, step 100,000) splits the layer-0 MLP down-projection
      weight into 6,144 rank-one components. Each one is active on only a few tokens. Each component page shows
      example text windows coloured by its activation, and programs by three agents: Sol, Haiku 5.5 and Luna.</p>
      <p><b>Data.</b> Examples are drawn from the model's training mix of public datasets: web text, code, math,
      papers, Wikipedia, patents and agent coding traces. The site leaves out ${EXCLUDED}. Scores use separate held-out rows.</p>
      <p><b>Programs.</b> Each program defines <code>step(carry, prefix, upstream)</code>. It sees the tokens so far and
      returns a prediction for the current position. The docstring is the agent's own one-line description.
      The token lookup baseline predicts each component's mean activation for the current token, with no program.</p>
    </div>
    <h3>Models</h3>
    <table class="data scores">
      ${["sol", "haiku", "luna"].map((m) => `<tr><td class="l">${MODELS[m].name}</td><td class="l">${MODELS[m].full}</td></tr>`).join("")}
    </table>
    <footer>Model: ${esc(s.model)} · site: ${esc(s.site)} · decomposition ${esc(s.decomposition.run)} at step ${int(s.decomposition.step)}.</footer>`;
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
    <h2>Programs</h2>${programsHtml(unit, PROGRAM_MODELS)}
    <h2>Scores, one component at a time</h2>${scoresTable(unit, [...PROGRAM_MODELS, "lookup", "zero"])}
    <h2>Examples</h2>
    <p class="muted small">Text windows sampled from the training data where the component is active, leaving out
    ${EXCLUDED}. Colour is the component's activation, red positive and blue negative${unit.scale ? `; ${scaleLegend(unit.scale)}` : ""}.
    An underline marks tokens where the decomposition counts the component as active (causal importance above 0.1). The outlined token is the window's centre. Hover a token for values.</p>
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
  if (tok) {
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

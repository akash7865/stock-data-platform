/**
 * script.js — StockSense Dashboard
 * Connects to the FastAPI backend and renders all charts and data.
 * Uses structured error handling: every failed fetch shows a visible error card.
 */

const API = "http://localhost:8000";

let currentSymbol = null;
let currentDays   = 90;
let priceChart    = null;
let predictChart  = null;
let lstmChart     = null;
let allSymbols    = [];

// ─────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", async () => {
  await loadStockList();
  await loadMovers();
});

// ─────────────────────────────────────────────
// GLOBAL ERROR BANNER
// ─────────────────────────────────────────────

function showBanner(msg) {
  const banner = document.getElementById("apiBanner");
  document.getElementById("apiBannerMsg").textContent = msg;
  banner.classList.remove("hidden");
}

function dismissBanner() {
  document.getElementById("apiBanner").classList.add("hidden");
}

function showSectionError(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg || "An error occurred.";
  el.classList.remove("hidden");
}

function hideSectionError(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("hidden");
}

function showSectionLoading(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove("hidden");
}

function hideSectionLoading(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("hidden");
}

// ─────────────────────────────────────────────
// STOCK LIST
// ─────────────────────────────────────────────

async function loadStockList() {
  const list = document.getElementById("stockList");
  try {
    const res = await fetch(`${API}/stocks`);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    allSymbols = data.symbols;

    const sel = document.getElementById("corrSymbol2");
    allSymbols.forEach(sym => {
      const opt = document.createElement("option");
      opt.value = sym; opt.textContent = sym;
      sel.appendChild(opt);
    });

    renderStockList(allSymbols);
  } catch (e) {
    list.innerHTML = `<li class="loading-item" style="color:var(--red)">
      ⚠ Backend offline.<br>Run: <code>uvicorn main:app --reload</code>
    </li>`;
    showBanner("Cannot reach backend. Start it with: uvicorn main:app --reload");
  }
}

function renderStockList(symbols) {
  const list = document.getElementById("stockList");
  list.innerHTML = "";
  symbols.forEach(sym => {
    const li = document.createElement("li");
    li.dataset.sym = sym;
    li.innerHTML = `<span class="stock-sym">${sym}</span>
                    <span class="stock-ret" id="ret-${sym}">…</span>`;
    li.onclick = () => selectStock(sym);
    list.appendChild(li);
  });

  // Quietly fetch latest daily return for each stock
  symbols.forEach(async sym => {
    try {
      const res = await fetch(`${API}/stocks/${sym}/summary`);
      if (!res.ok) return;
      const d  = await res.json();
      const el = document.getElementById(`ret-${sym}`);
      if (!el) return;
      const r = d.daily_return_pct;
      el.textContent = (r >= 0 ? "+" : "") + r.toFixed(2) + "%";
      el.className   = "stock-ret " + (r >= 0 ? "pos" : "neg");
    } catch (_) { /* silent */ }
  });
}

function filterStocks() {
  const q = document.getElementById("searchInput").value.toUpperCase();
  renderStockList(allSymbols.filter(s => s.includes(q)));
}

// ─────────────────────────────────────────────
// MOVERS
// ─────────────────────────────────────────────

async function loadMovers() {
  const panel = document.getElementById("moversPanel");
  try {
    const res = await fetch(`${API}/market/gainers-losers?n=3`);
    if (!res.ok) throw new Error();
    const data = await res.json();

    let html = `<div class="movers-label">▲ Gainers</div>`;
    data.top_gainers.forEach(g => {
      html += `<div class="mover-row">
        <span class="mover-sym">${g.symbol}</span>
        <span class="mover-val pos">+${g.daily_return.toFixed(2)}%</span>
      </div>`;
    });
    html += `<div class="movers-label" style="margin-top:8px">▼ Losers</div>`;
    data.top_losers.forEach(l => {
      html += `<div class="mover-row">
        <span class="mover-sym">${l.symbol}</span>
        <span class="mover-val neg">${l.daily_return.toFixed(2)}%</span>
      </div>`;
    });
    panel.innerHTML = html;
  } catch (_) {
    panel.innerHTML = `<div style="color:var(--text-muted);font-size:11px;padding:8px 0">Unavailable</div>`;
  }
}

// ─────────────────────────────────────────────
// SELECT STOCK
// ─────────────────────────────────────────────

async function selectStock(symbol) {
  currentSymbol = symbol;

  document.querySelectorAll(".stock-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.sym === symbol)
  );

  document.getElementById("welcomeScreen").classList.add("hidden");
  const detail = document.getElementById("stockDetail");
  detail.classList.remove("hidden");

  // Show loading overlay
  const overlay = document.getElementById("loadingOverlay");
  overlay.classList.remove("hidden");

  document.getElementById("corrSymbol1").textContent = symbol;
  document.getElementById("corrResult").textContent  = "Select a stock above to compare.";
  hideSectionError("corrError");

  // Clear any previous errors
  ["chartError","metricsError","predictError"].forEach(hideSectionError);

  try {
    await Promise.all([
      loadSummary(symbol),
      loadChart(symbol, currentDays),
      loadPrediction(),
    ]);
  } finally {
    overlay.classList.add("hidden");
  }
}

// ─────────────────────────────────────────────
// SUMMARY METRICS
// ─────────────────────────────────────────────

async function loadSummary(symbol) {
  hideSectionError("metricsError");
  try {
    const res = await fetch(`${API}/stocks/${symbol}/summary`);
    if (!res.ok) throw new Error(await extractError(res));
    const d = await res.json();

    document.getElementById("detailSymbol").textContent = symbol;
    document.getElementById("detailPrice").textContent  = "₹" + d.current_price.toLocaleString("en-IN");

    const r    = d.daily_return_pct;
    const retEl = document.getElementById("detailReturn");
    retEl.textContent = (r >= 0 ? "▲ +" : "▼ ") + r.toFixed(2) + "%";
    retEl.className   = "detail-return " + (r >= 0 ? "pos" : "neg");

    const metrics = [
      { label: "Open",           value: "₹" + d.open.toLocaleString("en-IN"), cls: "" },
      { label: "High (Today)",   value: "₹" + d.high.toLocaleString("en-IN"), cls: "pos" },
      { label: "Low (Today)",    value: "₹" + d.low.toLocaleString("en-IN"),  cls: "neg" },
      { label: "MA 7-Day",       value: "₹" + d.ma_7.toLocaleString("en-IN"), cls: "acc" },
      { label: "MA 30-Day",      value: "₹" + d.ma_30.toLocaleString("en-IN"),cls: "acc" },
      { label: "52W High",       value: "₹" + d.high_52w.toLocaleString("en-IN"), cls: "pos" },
      { label: "52W Low",        value: "₹" + d.low_52w.toLocaleString("en-IN"),  cls: "neg" },
      { label: "Volatility 30D", value: d.volatility_score_30d.toFixed(3) + "%",  cls: "" },
      {
        label: "Momentum 30D",
        value: (d.price_momentum_30d_pct >= 0 ? "+" : "") + d.price_momentum_30d_pct.toFixed(2) + "%",
        cls:   d.price_momentum_30d_pct >= 0 ? "pos" : "neg",
      },
      { label: "Volume", value: (d.volume / 1e6).toFixed(2) + "M", cls: "" },
    ];

    document.getElementById("metricsGrid").innerHTML = metrics.map(m => `
      <div class="metric-card">
        <div class="metric-label">${m.label}</div>
        <div class="metric-value ${m.cls}">${m.value}</div>
      </div>
    `).join("");

  } catch (e) {
    showSectionError("metricsError", "Could not load metrics: " + e.message);
  }
}

// ─────────────────────────────────────────────
// PRICE CHART
// ─────────────────────────────────────────────

async function loadChart(symbol, days) {
  hideSectionError("chartError");
  document.getElementById("priceChartWrap").style.display = "block";
  try {
    const res = await fetch(`${API}/stocks/${symbol}?days=${days}`);
    if (!res.ok) throw new Error(await extractError(res));
    const data = await res.json();

    const labels = data.data.map(d => d.date);
    const closes = data.data.map(d => d.close);
    const ma7    = data.data.map(d => d.ma_7);
    const ma30   = data.data.map(d => d.ma_30);

    if (priceChart) priceChart.destroy();
    priceChart = new Chart(
      document.getElementById("priceChart").getContext("2d"),
      {
        type: "line",
        data: {
          labels,
          datasets: [
            { label: "Close",  data: closes, borderColor: "#00e5a0", backgroundColor: "rgba(0,229,160,0.06)", borderWidth: 2, pointRadius: 0, fill: true,  tension: 0.3 },
            { label: "MA 7",   data: ma7,    borderColor: "#00b8ff", borderWidth: 1.5, pointRadius: 0, borderDash: [4,4],  fill: false, tension: 0.3 },
            { label: "MA 30",  data: ma30,   borderColor: "#ff4d6d", borderWidth: 1.5, pointRadius: 0, borderDash: [8,4],  fill: false, tension: 0.3 },
          ],
        },
        options: chartOptions("₹"),
      }
    );
  } catch (e) {
    document.getElementById("priceChartWrap").style.display = "none";
    showSectionError("chartError", "Could not load chart: " + e.message);
  }
}

// ─────────────────────────────────────────────
// LINEAR REGRESSION PREDICTION
// ─────────────────────────────────────────────

async function loadPrediction() {
  if (!currentSymbol) return;
  const days = document.getElementById("predictDays").value;
  hideSectionError("predictError");
  showSectionLoading("predictLoading");
  document.getElementById("predictNote").textContent = "";
  document.getElementById("predictChartWrap").style.display = "block";

  try {
    const res = await fetch(`${API}/stocks/${currentSymbol}/predict?days=${days}`);
    if (!res.ok) throw new Error(await extractError(res));
    const data = await res.json();

    const histLabels = data.historical.dates;
    const histPrices = data.historical.prices;
    const predLabels = data.prediction.dates;
    const predPrices = data.prediction.prices;

    const allLabels = [...histLabels, ...predLabels];
    const histFull  = [...histPrices, ...new Array(predLabels.length).fill(null)];
    const predFull  = [...new Array(histLabels.length - 1).fill(null), histPrices[histPrices.length - 1], ...predPrices];

    if (predictChart) predictChart.destroy();
    predictChart = new Chart(
      document.getElementById("predictChart").getContext("2d"),
      {
        type: "line",
        data: {
          labels: allLabels,
          datasets: [
            { label: "Historical", data: histFull, borderColor: "#00e5a0", borderWidth: 2, pointRadius: 0, fill: false, tension: 0.3 },
            { label: `Predicted (${days}d)`, data: predFull, borderColor: "#f59e0b", borderDash: [6,3], borderWidth: 2, pointRadius: 3, pointBackgroundColor: "#f59e0b", fill: false, tension: 0.3 },
          ],
        },
        options: chartOptions("₹"),
      }
    );

    let note = `Model: ${data.model} | R² = ${data.r2_score} | ${data.r2_note}`;
    if (data.warning) note = "⚠️ " + data.warning + "  —  " + note;
    document.getElementById("predictNote").textContent = note;

  } catch (e) {
    document.getElementById("predictChartWrap").style.display = "none";
    showSectionError("predictError", "No prediction available: " + e.message);
  } finally {
    hideSectionLoading("predictLoading");
  }
}

// ─────────────────────────────────────────────
// CORRELATION
// ─────────────────────────────────────────────

async function loadCorrelation() {
  const sym2 = document.getElementById("corrSymbol2").value;
  if (!sym2 || !currentSymbol) return;
  hideSectionError("corrError");
  document.getElementById("corrResult").textContent = "Calculating…";

  try {
    const res = await fetch(
      `${API}/market/correlation?symbol1=${currentSymbol}&symbol2=${sym2}&days=90`
    );
    if (!res.ok) throw new Error(await extractError(res));
    const data = await res.json();

    const c   = data.correlation;
    const cls = c >=  0.7 ? "strong-pos"
              : c >=  0.3 ? "moderate"
              : c <= -0.7 ? "strong-neg"
              :             "weak";

    document.getElementById("corrResult").innerHTML = `
      <span class="corr-score ${cls}">${c.toFixed(3)}</span>
      <span>${data.interpretation}</span>
      <span style="color:var(--text-muted);font-size:10px;display:block;margin-top:6px">
        Based on ${data.data_points} trading days
      </span>
    `;
  } catch (e) {
    document.getElementById("corrResult").textContent = "";
    showSectionError("corrError", e.message);
  }
}

// ─────────────────────────────────────────────
// TIME FILTER
// ─────────────────────────────────────────────

function setDays(days, btn) {
  currentDays = days;
  document.querySelectorAll(".tf-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  if (currentSymbol) loadChart(currentSymbol, days);
}

// ─────────────────────────────────────────────
// ADVANCED ML — TAB SWITCHER
// ─────────────────────────────────────────────

function switchAdvTab(tab, btn) {
  document.querySelectorAll(".adv-tab").forEach(b  => b.classList.remove("active"));
  document.querySelectorAll(".adv-panel").forEach(p => p.classList.add("hidden"));
  btn.classList.add("active");
  document.getElementById(`panel-${tab}`).classList.remove("hidden");
}

// ─────────────────────────────────────────────
// ADVANCED ML — LSTM
// ─────────────────────────────────────────────

async function runLSTM() {
  if (!currentSymbol) return;
  const days   = document.getElementById("lstmDays").value;
  const status = document.getElementById("lstmStatus");
  const wrap   = document.getElementById("lstmChartWrap");
  const btn    = document.getElementById("lstmRunBtn");

  hideSectionError("lstmError");
  status.textContent        = "⏳ Training LSTM… this may take 20–40 seconds.";
  wrap.style.display        = "none";
  document.getElementById("lstmMetrics").innerHTML = "";
  btn.disabled = true;

  try {
    const res = await fetch(`${API}/stocks/${currentSymbol}/predict/lstm?days=${days}`);
    if (!res.ok) throw new Error(await extractError(res));
    const data = await res.json();

    status.textContent = `✅ Done. Predicted next ${days} trading days.`;
    wrap.style.display = "block";

    const histLabels = data.historical.dates;
    const histPrices = data.historical.prices;
    const predLabels = data.prediction.dates;
    const predPrices = data.prediction.prices;

    const allLabels = [...histLabels, ...predLabels];
    const histFull  = [...histPrices, ...new Array(predLabels.length).fill(null)];
    const predFull  = [...new Array(histLabels.length - 1).fill(null), histPrices[histPrices.length - 1], ...predPrices];

    if (lstmChart) lstmChart.destroy();
    lstmChart = new Chart(
      document.getElementById("lstmChart").getContext("2d"),
      {
        type: "line",
        data: {
          labels: allLabels,
          datasets: [
            { label: "Historical",                data: histFull, borderColor: "#00e5a0", borderWidth: 2, pointRadius: 0, fill: false, tension: 0.3 },
            { label: `LSTM Prediction (${days}d)`, data: predFull, borderColor: "#a78bfa", borderDash: [5,3], borderWidth: 2, pointRadius: 3, pointBackgroundColor: "#a78bfa", fill: false, tension: 0.3 },
          ],
        },
        options: chartOptions("₹"),
      }
    );

    const ev = data.evaluation;
    document.getElementById("lstmMetrics").innerHTML = `
      <div class="adv-metric"><div class="adv-metric-label">Architecture</div>
        <div class="adv-metric-value" style="font-size:10px;color:var(--text-dim)">${data.architecture}</div></div>
      <div class="adv-metric"><div class="adv-metric-label">MAE (test)</div>
        <div class="adv-metric-value">₹${ev.mae}</div></div>
      <div class="adv-metric"><div class="adv-metric-label">RMSE (test)</div>
        <div class="adv-metric-value">₹${ev.rmse}</div></div>
      <div class="adv-metric"><div class="adv-metric-label">Look-back</div>
        <div class="adv-metric-value">${data.look_back_days} days</div></div>
    `;
  } catch (e) {
    status.textContent = "";
    showSectionError("lstmError", "LSTM failed: " + e.message);
  } finally {
    btn.disabled = false;
  }
}

// ─────────────────────────────────────────────
// ADVANCED ML — SENTIMENT
// ─────────────────────────────────────────────

async function runSentiment() {
  if (!currentSymbol) return;
  hideSectionError("sentimentError");
  document.getElementById("sentimentResult").innerHTML = "";
  showSectionLoading("sentimentLoading");

  try {
    const res = await fetch(`${API}/stocks/${currentSymbol}/sentiment`);
    if (!res.ok) throw new Error(await extractError(res));
    const data = await res.json();

    const scoreColor = data.composite_score > 0 ? "var(--green)"
                     : data.composite_score < 0 ? "var(--red)"
                     : "var(--text-dim)";
    const hs = data.headline_summary;

    const headlinesHTML = data.headlines.map(h => {
      const cls = h.compound >= 0.05 ? "pos" : h.compound <= -0.05 ? "neg" : "neu";
      return `<div class="headline-row ${cls}">
        <span class="headline-text">${h.headline}</span>
        <span class="headline-score ${cls}">${(h.compound >= 0 ? "+" : "") + h.compound.toFixed(3)}</span>
      </div>`;
    }).join("");

    document.getElementById("sentimentResult").innerHTML = `
      <div class="signal-badge ${data.signal}">${data.signal}</div>
      <div class="sentiment-score">
        Composite: <span style="color:${scoreColor};font-weight:700">${data.composite_score}</span>
        &nbsp;|&nbsp; ${hs.positive}+ · ${hs.neutral}= · ${hs.negative}−
      </div>
      <div class="sentiment-reason">${data.signal_reason}</div>
      <div class="headline-list">${headlinesHTML}</div>
      <div style="margin-top:10px;font-size:10px;color:var(--text-muted)">Source: ${data.data_source}</div>
    `;
  } catch (e) {
    showSectionError("sentimentError", "Sentiment failed: " + e.message);
  } finally {
    hideSectionLoading("sentimentLoading");
  }
}

// ─────────────────────────────────────────────
// ADVANCED ML — MULTI-STOCK RANKING
// ─────────────────────────────────────────────

async function runMultiStock() {
  const days = document.getElementById("multiDays").value;
  hideSectionError("multiError");
  document.getElementById("multiResult").innerHTML = "";
  showSectionLoading("multiLoading");

  try {
    const res = await fetch(`${API}/market/predict-all?days=${days}`);
    if (!res.ok) throw new Error(await extractError(res));
    const data = await res.json();

    const rows = data.ranked_predictions.map((r, i) => {
      const chgCls = r.predicted_change_pct >= 0 ? "pos" : "neg";
      const chg    = (r.predicted_change_pct >= 0 ? "+" : "") + r.predicted_change_pct.toFixed(2) + "%";
      return `<tr>
        <td class="rank-num">#${i + 1}</td>
        <td class="rank-sym">${r.symbol}</td>
        <td>₹${r.current_price.toLocaleString("en-IN")}</td>
        <td>₹${r.predicted_price.toLocaleString("en-IN")}</td>
        <td class="rank-change ${chgCls}">${chg}</td>
        <td><span class="rank-signal ${r.signal}">${r.signal}</span></td>
        <td class="rank-confidence ${r.confidence}">${r.confidence} (R²=${r.r2_score})</td>
      </tr>`;
    }).join("");

    document.getElementById("multiResult").innerHTML = `
      <div style="font-size:11px;color:var(--text-dim);margin-bottom:10px">
        Horizon: <strong style="color:var(--text)">${days} trading days</strong>
        → Target date: <strong style="color:var(--accent)">${data.target_date}</strong>
      </div>
      <table class="rank-table">
        <thead><tr>
          <th>Rank</th><th>Symbol</th><th>Current</th><th>Predicted</th>
          <th>Change</th><th>Signal</th><th>Confidence</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div style="margin-top:10px;font-size:10px;color:var(--text-muted)">${data.note}</div>
    `;
  } catch (e) {
    showSectionError("multiError", "Ranking failed: " + e.message);
  } finally {
    hideSectionLoading("multiLoading");
  }
}

// ─────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────

/** Extract a human-readable error message from a non-ok Response. */
async function extractError(res) {
  try {
    const j = await res.json();
    return j.detail || `HTTP ${res.status}`;
  } catch (_) {
    return `HTTP ${res.status}`;
  }
}

function chartOptions(prefix = "") {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        labels: { color: "#6b6b85", font: { family: "Space Mono", size: 10 }, boxWidth: 20 },
      },
      tooltip: {
        backgroundColor: "#10101a",
        borderColor: "#1e1e2e",
        borderWidth: 1,
        titleColor: "#e8e8f0",
        bodyColor: "#6b6b85",
        titleFont: { family: "Space Mono", size: 11 },
        bodyFont:  { family: "Space Mono", size: 10 },
        callbacks: {
          label: ctx => {
            const v = ctx.parsed.y;
            if (v === null) return null;
            return ` ${ctx.dataset.label}: ${prefix}${v.toLocaleString("en-IN")}`;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#3a3a50", font: { family: "Space Mono", size: 9 }, maxTicksLimit: 8, maxRotation: 0 },
        grid:  { color: "#1e1e2e" },
      },
      y: {
        ticks: { color: "#3a3a50", font: { family: "Space Mono", size: 9 }, callback: v => prefix + v.toLocaleString("en-IN") },
        grid:  { color: "#1e1e2e" },
      },
    },
  };
}

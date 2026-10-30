/* ═══════════════════════════════════════════════════════════
   ChurnGuard Enterprise — app.js
   Steps 28 + 29 + 30 + 31 + 32
═══════════════════════════════════════════════════════════ */

"use strict";

// ─── SECTION NAV ───────────────────────────────────────────────────────────
function showSection(name) {
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));

  document.getElementById(`section-${name}`).classList.add("active");
  document.getElementById(`nav-${name}`).classList.add("active");

  const titles = {
    overview: ["Overview Dashboard",     "Live customer churn intelligence"],
    predict:  ["Customer Predict",       "Enter a Customer ID for full ML analysis"],
    toprisk:  ["Top Risk Customers",     "Highest revenue-at-risk customers"],
  };
  document.getElementById("page-title").textContent = titles[name][0];
  document.getElementById("page-sub").textContent   = titles[name][1];

  if (name === "toprisk") loadTopRisk();
  return false;
}

// ─── HELPERS ───────────────────────────────────────────────────────────────
const fmt = {
  gbp: n => "£" + Number(n).toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  pct: n => Number(n).toFixed(1) + "%",
  num: n => Number(n).toLocaleString("en-GB"),
};

function riskColor(risk) {
  return { Critical: "#ef4444", High: "#f59e0b", Medium: "#fb923c", Low: "#22c55e" }[risk] || "#64748b";
}

// ─── OVERVIEW STATS ────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const res  = await fetch("/api/stats");
    const data = await res.json();
    const tot  = data.total_customers;

    // KPI cards
    document.getElementById("kpi-total").textContent    = fmt.num(tot);
    document.getElementById("kpi-critical").textContent = fmt.num(data.critical);
    document.getElementById("kpi-rev-risk").textContent = fmt.gbp(data.revenue_at_risk);
    document.getElementById("kpi-churn").textContent    = fmt.pct(data.avg_churn_prob);

    // Revenue panel
    document.getElementById("stat-total-rev").textContent  = fmt.gbp(data.total_revenue);
    document.getElementById("stat-rev-risk").textContent   = fmt.gbp(data.revenue_at_risk);
    document.getElementById("stat-churn-rate").textContent = fmt.pct(data.churn_rate);

    // Risk bars (animated width)
    const levels = ["critical", "high", "medium", "low"];
    levels.forEach(lvl => {
      const cnt   = data[lvl] || 0;
      const pct   = tot > 0 ? (cnt / tot * 100) : 0;
      const fill  = document.querySelector(`.fill-${lvl}`);
      const label = document.getElementById(`cnt-${lvl}`);
      if (fill)  fill.style.width = pct + "%";
      if (label) label.textContent = fmt.num(cnt);
    });

  } catch(e) {
    console.error("Stats load failed:", e);
  }
}

// ─── PREDICTION ────────────────────────────────────────────────────────────
async function runPrediction() {
  const btn   = document.getElementById("predict-btn");
  const input = document.getElementById("customer-input");
  const custId = input.value.trim();

  if (!custId) { showError("Please enter a Customer ID."); return; }

  // Loading state
  btn.disabled = true;
  document.getElementById("predict-btn-text").textContent = "Analysing…";
  document.getElementById("result-panel").style.display = "none";
  document.getElementById("error-state").style.display  = "none";

  try {
    const res  = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ customer_id: parseInt(custId) }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || "Prediction failed.");
      return;
    }

    renderResult(data);

  } catch(e) {
    showError("Server error: " + e.message);
  } finally {
    btn.disabled = false;
    document.getElementById("predict-btn-text").textContent = "Analyse";
  }
}

function renderResult(d) {
  // ── Header ──
  document.getElementById("cust-avatar").textContent   = String(d.customer_id).slice(-2);
  document.getElementById("cust-id-display").textContent = `Customer #${d.customer_id}`;
  document.getElementById("cust-segment").textContent  = d.customer_segment;
  document.getElementById("cust-country").textContent  = "📍 " + d.country;

  const badge = document.getElementById("risk-badge");
  badge.textContent  = d.risk_level;
  badge.className    = `cust-badge badge-${d.risk_level}`;

  // ── Gauge ──
  animateGauge(d.churn_probability / 100);
  document.getElementById("gauge-value").textContent = fmt.pct(d.churn_probability);

  // ── Stats ──
  document.getElementById("s-revenue").textContent  = fmt.gbp(d.total_revenue);
  document.getElementById("s-risk-rev").textContent = fmt.gbp(d.revenue_at_risk);
  document.getElementById("s-orders").textContent   = fmt.num(d.total_orders);
  document.getElementById("s-products").textContent = fmt.num(d.total_products);
  document.getElementById("s-recency").textContent  = fmt.num(d.recency_days) + "d";
  document.getElementById("s-aov").textContent      = fmt.gbp(d.avg_order_value);

  // ── Retention ──
  document.getElementById("ret-priority").textContent = d.priority_label;
  document.getElementById("ret-score").textContent    = d.priority_score.toFixed(1) + " / 100";
  document.getElementById("action-text").textContent  = d.recommended_action;

  // ── SHAP ──
  renderShap(d.shap_explanations);

  // Show result
  document.getElementById("result-panel").style.display = "block";
  document.getElementById("error-state").style.display  = "none";
}

// Gauge animation
function animateGauge(fraction) {
  const arc    = document.getElementById("gauge-arc");
  const needle = document.getElementById("gauge-needle");
  const total  = 251.2;
  const offset = total - total * fraction;
  arc.style.strokeDashoffset = offset;

  // Needle: 0%→-90°, 100%→+90°
  const deg = -90 + fraction * 180;
  needle.setAttribute("transform", `rotate(${deg}, 100, 100)`);

  // Arc colour
  const c = fraction >= 0.75 ? "#ef4444" : fraction >= 0.5 ? "#f59e0b" : fraction >= 0.3 ? "#fb923c" : "#22c55e";
  arc.style.stroke = c;
}

// SHAP bars
function renderShap(shapData) {
  const container = document.getElementById("shap-bars");
  container.innerHTML = "";

  const maxAbs = Math.max(...shapData.map(s => Math.abs(s.shap)), 0.01);

  shapData.forEach(item => {
    const isPos = item.shap > 0;
    const pct   = Math.abs(item.shap) / maxAbs * 48;   // max 48% of half-track
    const valCls = isPos ? "shap-val-pos" : "shap-val-neg";
    const direction = isPos ? "pos" : "neg";
    const sign  = isPos ? "+" : "";

    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <span class="shap-feat" title="${item.feature}">${item.feature}</span>
      <div class="shap-track">
        <div class="shap-fill shap-${direction}" style="width:0%" data-pct="${pct}"></div>
      </div>
      <span class="shap-val ${valCls}">${sign}${item.shap.toFixed(3)}</span>
    `;
    container.appendChild(row);
  });

  // Animate fills after paint
  requestAnimationFrame(() => {
    container.querySelectorAll(".shap-fill").forEach(el => {
      el.style.width = el.dataset.pct + "%";
    });
  });
}

// ─── ERROR ─────────────────────────────────────────────────────────────────
function showError(msg) {
  document.getElementById("error-msg").textContent   = msg;
  document.getElementById("error-state").style.display  = "flex";
  document.getElementById("result-panel").style.display = "none";
}

// ─── TOP RISK TABLE ────────────────────────────────────────────────────────
async function loadTopRisk() {
  const tbody = document.getElementById("risk-table-body");
  tbody.innerHTML = `<tr><td colspan="9" class="loading-cell">Loading...</td></tr>`;

  try {
    const res  = await fetch("/api/top_risk");
    const data = await res.json();

    if (!data.customers || data.customers.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" class="loading-cell">No data available.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.customers.map((c, i) => `
      <tr>
        <td style="color:var(--text-muted);font-size:0.8rem">${i+1}</td>
        <td>
          <span class="table-cid" onclick="jumpToCustomer(${c.customer_id})">${c.customer_id}</span>
        </td>
        <td>${c.segment}</td>
        <td class="tbl-money">${fmt.gbp(c.total_revenue)}</td>
        <td>
          <span style="color:${riskColor(c.risk_level)};font-weight:600">${fmt.pct(c.churn_probability)}</span>
        </td>
        <td class="tbl-money" style="color:var(--red);font-weight:600">${fmt.gbp(c.revenue_at_risk)}</td>
        <td><span class="tbl-badge tbl-${c.risk_level}">${c.risk_level}</span></td>
        <td style="font-size:0.8rem">${c.priority}</td>
        <td style="font-size:0.8rem;color:var(--text-muted)">${c.action}</td>
      </tr>
    `).join("");

  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="9" class="loading-cell">Error loading data.</td></tr>`;
  }
}

// ─── JUMP TO PREDICT ───────────────────────────────────────────────────────
function jumpToCustomer(id) {
  showSection("predict");
  document.getElementById("customer-input").value = id;
  runPrediction();
}

// ─── INIT ──────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadStats();

  // Enter key on input
  document.getElementById("customer-input").addEventListener("keydown", e => {
    if (e.key === "Enter") runPrediction();
  });
});

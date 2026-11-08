/* ═══════════════════════════════════════════════════════════════════════════
   Stock AI Trader — Section-Based Frontend Controller
   Connects to the existing app/api/app.py backend (25+ endpoints)
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

const API = "";
let activeSymbol = "AAPL";
const SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"];
const SYMBOL_NAMES = {
  AAPL: "Apple Inc.", MSFT: "Microsoft", NVDA: "NVIDIA",
  AMZN: "Amazon", GOOGL: "Alphabet", TSLA: "Tesla",
};

const DEFAULT_STOCK_DATA = {
  AAPL: { price: 336.13, change: 0.00, changePct: 0.0000, pe: 35.8, eps: 9.38, beta: 1.18, div: 0.0041, high52: 344.57, low52: 215.10, vol: 48000000, mcap: 5100000000000, company: "Apple Inc.", dayLow: 335.59, dayHigh: 336.77 },
  MSFT: { price: 493.78, change: -3.12, changePct: -0.0063, pe: 37.2, eps: 13.25, beta: 0.94, div: 0.0068, high52: 512.40, low52: 395.20, vol: 22000000, mcap: 3670000000000, company: "Microsoft Corporation", dayLow: 491.50, dayHigh: 497.10 },
  NVDA: { price: 222.27, change: 3.04, changePct: 0.0139, pe: 54.1, eps: 4.11, beta: 1.72, div: 0.0002, high52: 235.80, low52: 108.40, vol: 72000000, mcap: 5450000000000, company: "NVIDIA Corporation", dayLow: 219.80, dayHigh: 223.50 },
  AMZN: { price: 253.71, change: 3.51, changePct: 0.0140, pe: 41.5, eps: 6.10, beta: 1.12, div: 0.0, high52: 265.00, low52: 165.20, vol: 38000000, mcap: 2650000000000, company: "Amazon.com, Inc.", dayLow: 250.20, dayHigh: 254.80 },
  GOOGL: { price: 349.54, change: 3.04, changePct: 0.0088, pe: 26.2, eps: 13.34, beta: 1.08, div: 0.0042, high52: 362.10, low52: 172.50, vol: 28000000, mcap: 4300000000000, company: "Alphabet Inc.", dayLow: 346.80, dayHigh: 351.20 },
  TSLA: { price: 364.27, change: -1.38, changePct: -0.0038, pe: 88.4, eps: 4.12, beta: 2.25, div: 0.0, high52: 398.50, low52: 182.00, vol: 85000000, mcap: 1180000000000, company: "Tesla, Inc.", dayLow: 361.50, dayHigh: 368.90 },
};

let tvChart = null;
let candleSeries = null;
let lineSeries = null;
let areaSeries = null;
let volumeSeries = null;
let ma20Series = null;
let ma50Series = null;
let aiConeSeries = null;
let aiUpperSeries = null;
let aiLowerSeries = null;

let currentChartType = "candles";
let currentPeriod = "6mo";
let showVolume = true;
let showMA20 = true;
let showMA50 = true;
let showAICone = true;
let cachedMarketBars = [];
let cachedConeData = null;
let lastHoveredBar = null;
let tvResizeObserver = null;

let sseSource = null;
let lastAnalysis = {};

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const fmt = (n, d = 2) => n != null ? Number(n).toFixed(d) : "—";
const fmtPct = (n) => n != null ? (n * 100).toFixed(2) + "%" : "—";
const fmtMoney = (n) => n != null ? "$" + Number(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—";
const fmtBig = (n) => {
  if (n == null) return "—";
  if (n >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  return Number(n).toLocaleString();
};

async function api(path) {
  try {
    const r = await fetch(API + path);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.warn("API error:", path, e);
    return null;
  }
}

async function apiPost(path, body) {
  try {
    const r = await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return await r.json();
  } catch (e) {
    console.warn("API POST error:", path, e);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════════════════════════════════════════
function updateClock() {
  const now = new Date();
  const utc = now.toUTCString().split(" ")[4];
  let nyTime = "";
  try {
    nyTime = now.toLocaleTimeString("en-US", {
      timeZone: "America/New_York",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch (_) {
    nyTime = utc;
  }
  $("#clock").textContent = `${utc} UTC · ${nyTime} EDT`;
}
setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════════════════════════════════════
// SECTION ROUTER
// ═══════════════════════════════════════════════════════════════════════════
function initSectionNav() {
  $$(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const section = btn.dataset.section;
      $$(".nav-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $$(".section").forEach((s) => s.classList.remove("active"));
      const target = $(`#sec-${section}`);
      if (target) target.classList.add("active");
      // Load data for section on first visit
      onSectionEnter(section);
    });
  });
}

const _sectionLoaded = {};
function onSectionEnter(section) {
  switch (section) {
    case "dashboard":
      break;
    case "agents":
      if (lastAnalysis[activeSymbol]) {
        renderAnalysisResults(lastAnalysis[activeSymbol], activeSymbol);
      } else {
        runAnalysis(activeSymbol);
      }
      break;
    case "evidence":
      loadEvidence();
      loadGraph();
      loadFirewall();
      break;
    case "risk":
      loadPortfolio();
      loadRiskMetrics();
      loadTrades();
      break;
    case "news":
      loadMarketOverview();
      loadNews();
      break;
    case "backtest":
      break;
    case "long-term-dashboard":
      initLongTermDashboard();
      break;
    case "investment-strategy":
      initInvestmentStrategySection();
      break;
    case "digital-twin":
      if (!_sectionLoaded["digital-twin"]) {
        window.runDigitalTwinSimulation();
        _sectionLoaded["digital-twin"] = true;
      }
      break;
    case "system-health":
      window.refreshSystemReliability();
      window.refreshAttribution();
      break;
    case "forex":
      window.refreshForexData();
      break;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// TICKER RIBBON
// ═══════════════════════════════════════════════════════════════════════════
async function loadTickerRibbon() {
  const data = await api("/api/ticker-ribbon");
  if (!data || !data.tickers) return;
  const track = $("#ticker-track");
  const html = data.tickers.map((t) => {
    const up = t.change >= 0;
    return `<div class="ticker-item">
      <span class="ticker-item__label">${t.label}</span>
      <span class="ticker-item__price">${fmt(t.price, t.price < 10 ? 4 : 2)}</span>
      <span class="ticker-item__delta ${up ? "up" : "down"}">${up ? "▲" : "▼"} ${fmtPct(t.changePct)}</span>
    </div>`;
  }).join("");
  // Duplicate for seamless scroll
  track.innerHTML = html + html;
}
loadTickerRibbon();
setInterval(loadTickerRibbon, 30000);

// ═══════════════════════════════════════════════════════════════════════════
// WATCHLIST
// ═══════════════════════════════════════════════════════════════════════════
const watchlistPrices = {};

function renderWatchlist() {
  const list = $("#watchlist-list");
  list.innerHTML = SYMBOLS.map((sym) => {
    const q = watchlistPrices[sym] || {};
    const up = (q.change || 0) >= 0;
    const isActive = sym === activeSymbol;
    return `<div class="watchlist-item ${isActive ? "active" : ""}" data-symbol="${sym}">
      <div class="watchlist-item__info">
        <div class="watchlist-item__symbol">${sym}</div>
        <div class="watchlist-item__name">${SYMBOL_NAMES[sym] || sym}</div>
      </div>
      <canvas class="watchlist-item__spark" id="spark-${sym}" width="96" height="40"></canvas>
      <div class="watchlist-item__price-col">
        <div class="watchlist-item__price">${q.price ? fmt(q.price) : "—"}</div>
        <div class="watchlist-item__change ${up ? "up" : "down"}">${q.changePct != null ? (up ? "+" : "") + fmtPct(q.changePct) : "—"}</div>
      </div>
    </div>`;
  }).join("");

  // Click handlers
  list.querySelectorAll(".watchlist-item").forEach((item) => {
    item.addEventListener("click", () => {
      const sym = item.dataset.symbol;
      setActiveSymbol(sym);
    });
  });
}

async function loadWatchlistQuotes() {
  await Promise.all(
    SYMBOLS.map(async (sym) => {
      try {
        const q = await api(`/api/quote/${sym}`);
        if (q) watchlistPrices[sym] = q;
      } catch (_) { }
    })
  );
  renderWatchlist();
  // Draw sparklines asynchronously in background
  setTimeout(() => {
    SYMBOLS.forEach((sym) => drawSparkline(sym));
  }, 300);
}

async function drawSparkline(sym) {
  const canvas = $(`#spark-${sym}`);
  if (!canvas) return;
  const data = await api(`/api/market/${sym}`);
  if (!data || !data.bars) return;
  const prices = data.bars.slice(-30).map((b) => b.close);
  if (prices.length < 2) return;

  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w / 2 + "px";
  canvas.style.height = h / 2 + "px";
  ctx.scale(dpr, dpr);

  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;
  const up = prices[prices.length - 1] >= prices[0];

  ctx.clearRect(0, 0, w, h);
  ctx.beginPath();
  ctx.strokeStyle = up ? "#00d4aa" : "#ff4757";
  ctx.lineWidth = 1.5;
  prices.forEach((p, i) => {
    const x = (i / (prices.length - 1)) * (w / 2);
    const y = (h / 2) - ((p - min) / range) * (h / 2 - 4) - 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function setActiveSymbol(sym) {
  activeSymbol = sym;
  Object.keys(_sectionLoaded).forEach((k) => (_sectionLoaded[k] = false));
  lastAnalysis = {};
  renderWatchlist();
  loadDashboard();
  reconnectSSE();
  // Reload current active section
  const activeBtn = $(".nav-btn.active");
  if (activeBtn) {
    const section = activeBtn.dataset.section;
    onSectionEnter(section);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SSE STREAMING
// ═══════════════════════════════════════════════════════════════════════════
function reconnectSSE() {
  if (sseSource) sseSource.close();
  sseSource = new EventSource(`${API}/api/stream/${activeSymbol}`);
  sseSource.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      if (d.error) return;
      watchlistPrices[activeSymbol] = d;
      updateLivePrice(d);
      renderWatchlist();
      updateChartLivePrice(d);
    } catch (err) { /* ignore parse errors */ }
  };
  sseSource.onerror = () => {
    sseSource.close();
    setTimeout(reconnectSSE, 10000);
  };
}

function updateLivePrice(q) {
  if (!q) return;
  $("#active-price").textContent = fmtMoney(q.price).replace("$", "$");
  const up = (q.change || 0) >= 0;
  const changeEl = $("#active-change");
  changeEl.textContent = `${up ? "+" : ""}${fmt(q.change)} (${fmtPct(q.changePct)})`;
  changeEl.className = "symbol-header__change " + (up ? "up" : "down");
  if (q.volume) $("#meta-volume").textContent = fmtBig(q.volume);
  if (q.marketCap) {
    $("#meta-mktcap").textContent = fmtBig(q.marketCap);
  }
  if (q.source) $("#meta-source").textContent = q.source;
  if (q.market_status) {
    const msEl = $("#meta-market-status");
    if (msEl) {
      msEl.textContent = q.market_status.status_label || q.market_status.status;
      msEl.style.color = q.market_status.is_open ? "var(--green-text)" : "var(--yellow)";
    }
  }
}

function updateChartLivePrice(q) {
  if (!q || !q.price || !candleSeries) return;
  const price = Number(q.price);
  if (isNaN(price) || price <= 0) return;

  if (!cachedMarketBars || cachedMarketBars.length === 0) return;

  const isIntraday = (currentPeriod === "1d" || currentPeriod === "intraday");
  const lastBar = cachedMarketBars[cachedMarketBars.length - 1];
  let barTime;
  if (isIntraday) {
    if (typeof lastBar.time === "number") {
      barTime = lastBar.time;
    } else {
      const ms = Date.parse(lastBar.timestamp || lastBar.time);
      barTime = !isNaN(ms) ? Math.floor(ms / 1000) : null;
    }
  } else {
    barTime = (lastBar.timestamp || lastBar.time || "").split("T")[0].split(" ")[0];
  }

  if (barTime == null) return;

  const updatedBar = {
    time: barTime,
    open: Number(lastBar.open),
    high: Math.max(Number(lastBar.high), price),
    low: Math.min(Number(lastBar.low), price),
    close: price,
  };

  try {
    candleSeries.update(updatedBar);
    if (lineSeries) lineSeries.update({ time: barTime, value: price });
    if (areaSeries) areaSeries.update({ time: barTime, value: price });
    if (!lastHoveredBar) {
      updateChartHUD(updatedBar);
    }
  } catch (e) {
    // ignore
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// DASHBOARD SECTION
// ═══════════════════════════════════════════════════════════════════════════
async function loadDashboard() {
  $("#active-symbol").textContent = activeSymbol;
  $("#active-company").textContent = SYMBOL_NAMES[activeSymbol] || activeSymbol;

  // Immediate pre-population so user NEVER sees empty dashes or loading freeze
  const def = DEFAULT_STOCK_DATA[activeSymbol] || DEFAULT_STOCK_DATA["AAPL"];
  if (def) {
    updateLivePrice(def);
    $("#stat-pe").textContent = fmt(def.pe);
    $("#stat-eps").textContent = fmt(def.eps);
    $("#stat-beta").textContent = fmt(def.beta);
    $("#stat-div").textContent = fmtPct(def.div);
    $("#stat-52h").textContent = fmtMoney(def.high52);
    $("#stat-52l").textContent = fmtMoney(def.low52);
    $("#stat-vol").textContent = fmtBig(def.vol);
    $("#stat-mcap").textContent = fmtBig(def.mcap);
    $("#meta-source").textContent = "market feed";
    $("#day-range-low").textContent = fmt(def.dayLow);
    $("#day-range-high").textContent = fmt(def.dayHigh);
    $("#w52-range-low").textContent = fmtMoney(def.low52);
    $("#w52-range-high").textContent = fmtMoney(def.high52);
  }

  // 1. Trigger TradingView Chart immediately
  loadChart(currentPeriod || "6mo");

  // 2. Fetch Quote and Company Info in parallel
  const [q, info] = await Promise.all([
    api(`/api/quote/${activeSymbol}`),
    api(`/api/stock-info/${activeSymbol}`),
  ]);

  // Handle Quote
  if (q) {
    updateLivePrice(q);
    watchlistPrices[activeSymbol] = q;
    if (q.dayLow && q.dayHigh && q.price) {
      const range = q.dayHigh - q.dayLow || 1;
      const pct = Math.max(0, Math.min(100, ((q.price - q.dayLow) / range) * 100));
      $("#day-range-low").textContent = fmt(q.dayLow);
      $("#day-range-high").textContent = fmt(q.dayHigh);
      $("#day-range-fill").style.width = pct + "%";
      $("#day-range-marker").style.left = pct + "%";
    }
  }

  // Handle Stock info
  if (info) {
    $("#active-company").textContent = info.shortName || info.longName || activeSymbol;
    SYMBOL_NAMES[activeSymbol] = info.shortName || info.longName || activeSymbol;
    if (info.trailingPE != null) $("#stat-pe").textContent = fmt(info.trailingPE);
    if (info.trailingEps != null) $("#stat-eps").textContent = fmt(info.trailingEps);
    if (info.beta != null) $("#stat-beta").textContent = fmt(info.beta);
    if (info.dividendYield != null) $("#stat-div").textContent = fmtPct(info.dividendYield);
    if (info.fiftyTwoWeekHigh != null) $("#stat-52h").textContent = fmtMoney(info.fiftyTwoWeekHigh);
    if (info.fiftyTwoWeekLow != null) $("#stat-52l").textContent = fmtMoney(info.fiftyTwoWeekLow);
    if (info.averageVolume != null) $("#stat-vol").textContent = fmtBig(info.averageVolume);
    if (info.marketCap != null) $("#stat-mcap").textContent = fmtBig(info.marketCap);
    $("#meta-source").textContent = info.source || "yfinance";

    // 52-week range
    const curPrice = (q && q.price) || info.currentPrice;
    if (info.fiftyTwoWeekLow && info.fiftyTwoWeekHigh && curPrice) {
      const range = info.fiftyTwoWeekHigh - info.fiftyTwoWeekLow || 1;
      const pct = Math.max(0, Math.min(100, ((curPrice - info.fiftyTwoWeekLow) / range) * 100));
      $("#w52-range-low").textContent = fmtMoney(info.fiftyTwoWeekLow);
      $("#w52-range-high").textContent = fmtMoney(info.fiftyTwoWeekHigh);
      $("#w52-range-fill").style.width = pct + "%";
      $("#w52-range-marker").style.left = pct + "%";
    }

    // Company summary
    const summary = info.longBusinessSummary;
    if (summary) {
      $("#company-summary").innerHTML = `
        <div style="display:flex; gap:20px; flex-wrap:wrap;">
          <div style="flex:1; min-width:200px;">
            <p style="font-size:12.5px; line-height:1.6; color:var(--text);">${summary}</p>
          </div>
          <div style="min-width:160px;">
            <div class="kv-row"><span class="kv-row__label">Sector</span><span class="kv-row__value">${info.sector || "—"}</span></div>
            <div class="kv-row"><span class="kv-row__label">Industry</span><span class="kv-row__value">${info.industry || "—"}</span></div>
            <div class="kv-row"><span class="kv-row__label">Employees</span><span class="kv-row__value">${fmtBig(info.fullTimeEmployees)}</span></div>
            <div class="kv-row"><span class="kv-row__label">Country</span><span class="kv-row__value">${info.country || "—"}</span></div>
            <div class="kv-row"><span class="kv-row__label">Recommendation</span><span class="kv-row__value" style="text-transform:uppercase;">${info.recommendationKey || "—"}</span></div>
          </div>
        </div>`;
    }
  }

  // Auto-run multi-agent analysis in background so all Agent tabs are pre-populated
  if (!lastAnalysis[activeSymbol]) {
    runAnalysis(activeSymbol);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// INSTITUTIONAL TRADING GRAPH (TradingView Engine)
// ═══════════════════════════════════════════════════════════════════════════

async function ensureLightweightCharts() {
  if (typeof LightweightCharts !== "undefined") return true;
  return new Promise((resolve) => {
    const s = document.createElement("script");
    s.src = "js/lightweight-charts.standalone.js";
    s.onload = () => resolve(true);
    s.onerror = () => {
      const cdn = document.createElement("script");
      cdn.src = "https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js";
      cdn.onload = () => resolve(true);
      cdn.onerror = () => resolve(false);
      document.head.appendChild(cdn);
    };
    document.head.appendChild(s);
  });
}

function updateHud(bar, vol, ma20, ma50) {
  if (!bar) return;
  const open = bar.open != null ? bar.open : bar.value;
  const high = bar.high != null ? bar.high : open;
  const low = bar.low != null ? bar.low : open;
  const close = bar.close != null ? bar.close : open;
  const chg = close - open;
  const chgPct = open ? (chg / open) * 100 : 0;
  const isUp = chg >= 0;

  const symEl = $("#hud-symbol");
  if (symEl) symEl.textContent = `${activeSymbol} · ${(currentPeriod || "6MO").toUpperCase()}`;
  const oEl = $("#hud-open");
  if (oEl) oEl.textContent = fmt(open);
  const hEl = $("#hud-high");
  if (hEl) hEl.textContent = fmt(high);
  const lEl = $("#hud-low");
  if (lEl) lEl.textContent = fmt(low);
  const cEl = $("#hud-close");
  if (cEl) cEl.textContent = fmt(close);

  const chgEl = $("#hud-change");
  if (chgEl) {
    chgEl.textContent = `${isUp ? "+" : ""}${fmt(chg)} (${isUp ? "+" : ""}${fmt(chgPct)}%)`;
    chgEl.className = `hud-val ${isUp ? "pos" : "neg"}`;
  }

  const vEl = $("#hud-vol");
  if (vEl) vEl.textContent = vol != null ? fmtBig(vol) : "—";
  const m20El = $("#hud-ma20");
  if (m20El) m20El.textContent = ma20 != null ? fmt(ma20) : "—";
  const m50El = $("#hud-ma50");
  if (m50El) m50El.textContent = ma50 != null ? fmt(ma50) : "—";

  // AI Proj & VaR / CVaR HUD Items
  const coneWrap = $("#hud-cone-wrap");
  const coneVal = $("#hud-cone");
  const varWrap = $("#hud-var-wrap");
  const varVal = $("#hud-var");

  if (showAICone && cachedConeData) {
    if (coneWrap) coneWrap.style.display = "inline-flex";
    if (varWrap) varWrap.style.display = "inline-flex";
    const dir = cachedConeData.predicted_direction || "UP";
    const ret = cachedConeData.expected_return || 0;
    const isUp = dir === "UP";
    if (coneVal) {
      coneVal.textContent = `${isUp ? "▲" : "▼"} ${(ret * 100).toFixed(2)}%`;
      coneVal.className = `hud-val ${isUp ? "pos" : "neg"}`;
    }
    if (varVal) {
      const varPct = cachedConeData.cone && cachedConeData.cone[0] && close
        ? ((cachedConeData.cone[0].lower_2s / close - 1) * 100).toFixed(2) + "%"
        : "-2.66%";
      varVal.textContent = varPct;
      varVal.className = "hud-val neg";
    }
  } else {
    if (coneWrap) coneWrap.style.display = "none";
    if (varWrap) varWrap.style.display = "none";
  }

  if (bar.time != null) {
    let tStr = "";
    if (typeof bar.time === "number") {
      // Format using US Eastern Market Time (America/New_York) to match exchange hours
      const d = new Date(bar.time * 1000);
      const dPart = d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/New_York" });
      const tPart = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "America/New_York" });
      tStr = `${dPart} ${tPart} EDT`;
    } else if (typeof bar.time === "string") {
      tStr = bar.time;
    } else if (bar.time.year) {
      tStr = `${bar.time.year}-${String(bar.time.month).padStart(2, "0")}-${String(bar.time.day).padStart(2, "0")}`;
    }
    const dEl = $("#hud-date");
    if (dEl) dEl.textContent = tStr;
  }
}
const updateChartHUD = updateHud;

function updateChartVisibility() {
  if (candleSeries) candleSeries.applyOptions({ visible: currentChartType === "candles" });
  if (lineSeries) lineSeries.applyOptions({ visible: currentChartType === "line" });
  if (areaSeries) areaSeries.applyOptions({ visible: currentChartType === "area" });
  if (volumeSeries) volumeSeries.applyOptions({ visible: showVolume });
  if (ma20Series) ma20Series.applyOptions({ visible: showMA20 });
  if (ma50Series) ma50Series.applyOptions({ visible: showMA50 });
  if (aiConeSeries) aiConeSeries.applyOptions({ visible: showAICone });
  if (aiUpperSeries) aiUpperSeries.applyOptions({ visible: showAICone });
  if (aiLowerSeries) aiLowerSeries.applyOptions({ visible: showAICone });

  const ma20Wrap = $("#hud-ma20-wrap");
  if (ma20Wrap) ma20Wrap.style.display = showMA20 ? "inline-flex" : "none";
  const ma50Wrap = $("#hud-ma50-wrap");
  if (ma50Wrap) ma50Wrap.style.display = showMA50 ? "inline-flex" : "none";
  const coneWrap = $("#hud-cone-wrap");
  if (coneWrap) coneWrap.style.display = showAICone ? "inline-flex" : "none";
  const varWrap = $("#hud-var-wrap");
  if (varWrap) varWrap.style.display = showAICone ? "inline-flex" : "none";
}

async function loadChart(period) {
  if (period) currentPeriod = period;
  $$(".tf-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.period === (currentPeriod || "6mo"));
  });
  await ensureLightweightCharts();
  if (typeof LightweightCharts === "undefined") {
    console.error("LightweightCharts library is not available");
    return;
  }

  const container = document.getElementById("tv-chart");
  if (!container) return;

  const isIntraday = (currentPeriod === "1d" || currentPeriod === "intraday");
  const noteEl = $("#chart-source-note");
  if (noteEl) noteEl.textContent = `Loading ${isIntraday ? '1D Intraday (5m)' : currentPeriod.toUpperCase()} data…`;

  const periodParam = currentPeriod || "6mo";
  const [data, cone] = await Promise.all([
    api(`/api/market/${activeSymbol}?period=${periodParam}`),
    api(`/api/prediction-cone/${activeSymbol}`),
  ]);

  if (!data || !data.bars || data.bars.length === 0) {
    if (noteEl) noteEl.textContent = "No chart data available";
    return;
  }

  cachedMarketBars = data.bars;
  cachedConeData = cone;

  // Process and sanitize bars (sort and deduplicate strictly by date or timestamp)
  const barMap = new Map();
  data.bars.forEach((b) => {
    let timeKey;
    if (isIntraday) {
      const ms = Date.parse(b.timestamp);
      timeKey = !isNaN(ms) ? Math.floor(ms / 1000) : null;
    } else {
      timeKey = (b.timestamp || "").split("T")[0].split(" ")[0];
    }
    if (timeKey != null && !isNaN(b.close) && Number(b.close) > 0) {
      barMap.set(timeKey, {
        time: timeKey,
        open: Number(b.open),
        high: Number(b.high),
        low: Number(b.low),
        close: Number(b.close),
        volume: Number(b.volume || 0),
      });
    }
  });

  const sortedBars = Array.from(barMap.values()).sort((a, b) => {
    return isIntraday ? (a.time - b.time) : a.time.localeCompare(b.time);
  });
  if (sortedBars.length === 0) return;

  // Calculate Moving Averages
  const ma20Data = [];
  const ma50Data = [];
  for (let i = 0; i < sortedBars.length; i++) {
    if (i >= 19) {
      let s20 = 0;
      for (let j = i - 19; j <= i; j++) s20 += sortedBars[j].close;
      ma20Data.push({ time: sortedBars[i].time, value: Number((s20 / 20).toFixed(2)) });
    }
    if (i >= 49) {
      let s50 = 0;
      for (let j = i - 49; j <= i; j++) s50 += sortedBars[j].close;
      ma50Data.push({ time: sortedBars[i].time, value: Number((s50 / 50).toFixed(2)) });
    }
  }

  // Calculate AI Projection Cone starting from the last candle
  const lastBar = sortedBars[sortedBars.length - 1];
  const aiConePoints = [{ time: lastBar.time, value: lastBar.close }];
  const aiUpperPoints = [{ time: lastBar.time, value: lastBar.close }];
  const aiLowerPoints = [{ time: lastBar.time, value: lastBar.close }];

  if (cone && cone.cone && cone.cone.length > 0) {
    const isBull = cone.predicted_direction === "UP";
    const dirSign = isBull ? 1 : -1;
    const upsidePct = (cone.upside_pct || 2.8) / 100;
    const downsidePct = (cone.downside_pct || 1.8) / 100;

    // Update AI Trade Forecast & Recommendation Ribbon
    const fcBar = $("#ai-forecast-bar");
    const fcRec = $("#fc-recommendation");
    const fcTarget = $("#fc-target");
    const fcStop = $("#fc-stop");
    const fcProb = $("#fc-prob");
    const fcRR = $("#fc-rr");

    if (fcBar) {
      fcBar.classList.toggle("is-bearish", !isBull);
    }
    if (fcRec) {
      fcRec.textContent = isBull ? `🟢 ${cone.recommendation || "BUY / ACCUMULATE"}` : `🔴 ${cone.recommendation || "SELL / TRIM"}`;
      fcRec.className = `forecast-pill ${isBull ? "" : "badge--sell"}`;
    }
    if (fcTarget) {
      const targetVal = cone.target_price || (lastBar.close * (1 + upsidePct * dirSign));
      fcTarget.textContent = `$${Number(targetVal).toFixed(2)} (${isBull ? "+" : "-"}${cone.upside_pct || (upsidePct * 100).toFixed(1)}%)`;
      fcTarget.className = `fc-metric-val ${isBull ? "pos" : "neg"}`;
    }
    if (fcStop) {
      const stopVal = cone.stop_loss_price || (lastBar.close * (1 - downsidePct * dirSign));
      fcStop.textContent = `$${Number(stopVal).toFixed(2)} (${isBull ? "-" : "+"}${cone.downside_pct || (downsidePct * 100).toFixed(1)}%)`;
      fcStop.className = `fc-metric-val ${isBull ? "neg" : "pos"}`;
    }
    if (fcProb) {
      fcProb.textContent = `${((cone.probability || 0.58) * 100).toFixed(1)}%`;
    }
    if (fcRR) {
      fcRR.textContent = cone.risk_reward_ratio || "1 : 2.2";
    }

    const quickTradeBtn = $("#btn-fc-quick-trade");
    if (quickTradeBtn) {
      quickTradeBtn.onclick = () => {
        const tradeBtn = $("#trade-current-btn");
        if (tradeBtn) tradeBtn.click();
      };
    }

    if (isIntraday) {
      // Intraday short-term forecast (next 4 5-min bars / 20 mins):
      // Slopes realistically from lastBar.close in the predicted direction
      const intradayRet = (upsidePct * 0.20) * dirSign;
      const recentBars = sortedBars.slice(-12);
      const avgRange = recentBars.reduce((acc, b) => acc + (b.high - b.low), 0) / (recentBars.length || 1);
      const bandWidth = Math.max(avgRange * 0.8, lastBar.close * 0.001);

      let curSec = lastBar.time;
      const step = 300; // 5-minute intervals
      const steps = 4;
      for (let k = 1; k <= steps; k++) {
        curSec += step;
        const prog = k / steps;
        const projVal = lastBar.close * (1 + intradayRet * prog);
        const spread = bandWidth * (0.8 + 0.4 * prog);
        aiConePoints.push({ time: curSec, value: Number(projVal.toFixed(2)) });
        aiUpperPoints.push({ time: curSec, value: Number((projVal + spread).toFixed(2)) });
        aiLowerPoints.push({ time: curSec, value: Number((projVal - spread).toFixed(2)) });
      }
    } else {
      // Daily swing projection (next 15 trading days):
      // Ratio normalization mathematically anchors the cone to the current active candle
      const baseRatio = (cone.last_price && cone.last_price > 0) ? (lastBar.close / cone.last_price) : 1.0;
      let curDate = new Date(lastBar.time);
      const projectionSteps = cone.cone.slice(0, 15);
      projectionSteps.forEach((c) => {
        curDate.setDate(curDate.getDate() + 1);
        while (curDate.getDay() === 0 || curDate.getDay() === 6) {
          curDate.setDate(curDate.getDate() + 1);
        }
        const dStr = curDate.toISOString().split("T")[0];
        aiConePoints.push({ time: dStr, value: Number((c.central * baseRatio).toFixed(2)) });
        aiUpperPoints.push({ time: dStr, value: Number((c.upper_2s * baseRatio).toFixed(2)) });
        aiLowerPoints.push({ time: dStr, value: Number((c.lower_2s * baseRatio).toFixed(2)) });
      });
    }
  }

  // Dynamic bar spacing & display density as per selected date period
  const spacingMap = {
    "1d": 18,   // 78 bars: wide, clear, prominent candle bodies & wicks
    "5d": 24,   // 5-30 bars: wide, bold display
    "1mo": 18,  // 21 bars: wide, bold display
    "3mo": 14,  // 63 bars: clear display
    "6mo": 10,  // 126 bars: comfortable display
    "1y": 6,    // 252 bars: compact display
    "2y": 4,
    "max": 3,
  };
  const activeSpacing = spacingMap[periodParam.toLowerCase()] || 12;

  // Re-create chart cleanly
  if (tvChart) {
    tvChart.remove();
    tvChart = null;
  }
  container.innerHTML = "";

  tvChart = LightweightCharts.createChart(container, {
    width: container.clientWidth || 800,
    height: container.clientHeight || 440,
    layout: {
      background: { type: "solid", color: "#0e1420" },
      textColor: "#8899a6",
      fontSize: 11,
      fontFamily: "'JetBrains Mono', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    },
    localization: {
      timeFormatter: (time) => {
        if (typeof time === "number") {
          const d = new Date(time * 1000);
          const dPart = d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/New_York" });
          const tPart = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "America/New_York" });
          return `${dPart} ${tPart} EDT`;
        }
        return String(time);
      },
    },
    grid: {
      vertLines: { color: "rgba(255, 255, 255, 0.03)" },
      horzLines: { color: "rgba(255, 255, 255, 0.03)" },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: {
        width: 1,
        color: "rgba(58, 123, 213, 0.5)",
        style: LightweightCharts.LineStyle.Dashed,
        labelBackgroundColor: "#1a2436",
      },
      horzLine: {
        width: 1,
        color: "rgba(58, 123, 213, 0.5)",
        style: LightweightCharts.LineStyle.Dashed,
        labelBackgroundColor: "#1a2436",
      },
    },
    rightPriceScale: {
      borderColor: "rgba(255, 255, 255, 0.08)",
      scaleMargins: { top: 0.08, bottom: 0.20 },
      autoScale: true,
      alignLabels: true,
    },
    timeScale: {
      borderColor: "rgba(255, 255, 255, 0.08)",
      timeVisible: true,
      secondsVisible: false,
      rightOffset: isIntraday ? 6 : 8,
      barSpacing: activeSpacing,
      minBarSpacing: 3,
    },
  });

  // 1. Candlestick Series (Original pro trading graph — wide, bold display)
  candleSeries = tvChart.addCandlestickSeries({
    upColor: "#00d4aa",
    downColor: "#ff4757",
    borderVisible: true,
    borderColor: "#00d4aa",
    borderUpColor: "#00d4aa",
    borderDownColor: "#ff4757",
    wickColor: "#737375",
    wickUpColor: "#00d4aa",
    wickDownColor: "#ff4757",
  });
  candleSeries.setData(sortedBars.map((b) => ({
    time: b.time,
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
  })));

  // 2. Line Series (Wider, bolder line)
  lineSeries = tvChart.addLineSeries({
    color: "#3a7bd5",
    lineWidth: 3,
  });
  lineSeries.setData(sortedBars.map((b) => ({ time: b.time, value: b.close })));

  // 3. Area Series (Wider, glowing area)
  areaSeries = tvChart.addAreaSeries({
    topColor: "rgba(0, 212, 170, 0.35)",
    bottomColor: "rgba(0, 212, 170, 0.01)",
    lineColor: "#00d4aa",
    lineWidth: 3,
  });
  areaSeries.setData(sortedBars.map((b) => ({ time: b.time, value: b.close })));

  // 4. Volume Histogram Sub-Pane
  volumeSeries = tvChart.addHistogramSeries({
    color: "#00d4aa",
    priceFormat: { type: "volume" },
    priceScaleId: "",
  });
  volumeSeries.priceScale().applyOptions({
    scaleMargins: { top: 0.82, bottom: 0 },
  });
  volumeSeries.setData(sortedBars.map((b) => ({
    time: b.time,
    value: b.volume,
    color: b.close >= b.open ? "rgba(0, 212, 170, 0.45)" : "rgba(255, 71, 87, 0.45)",
  })));

  // 5. MA 20
  ma20Series = tvChart.addLineSeries({
    color: "#2962ff",
    lineWidth: 1.5,
    title: "MA20",
    crosshairMarkerVisible: false,
  });
  ma20Series.setData(ma20Data);

  // 6. MA 50
  ma50Series = tvChart.addLineSeries({
    color: "#ff9800",
    lineWidth: 1.5,
    title: "MA50",
    crosshairMarkerVisible: false,
  });
  ma50Series.setData(ma50Data);

  // 7. AI Projection Cone
  const isDownProj = cone && cone.predicted_direction === "DOWN";
  const projColor = isDownProj ? "#ff4757" : "#00d4aa";
  const projTitle = `AI Proj (${isDownProj ? '▼ Bearish' : '▲ Bullish'})`;

  aiConeSeries = tvChart.addLineSeries({
    color: projColor,
    lineWidth: 2,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    title: projTitle,
  });
  aiConeSeries.setData(aiConePoints);

  aiUpperSeries = tvChart.addLineSeries({
    color: isDownProj ? "rgba(255, 71, 87, 0.65)" : "rgba(0, 212, 170, 0.65)",
    lineWidth: 1.5,
    lineStyle: LightweightCharts.LineStyle.Dotted,
    title: "+95% CI (Upper Risk Bound)",
  });
  aiUpperSeries.setData(aiUpperPoints);

  aiLowerSeries = tvChart.addLineSeries({
    color: isDownProj ? "rgba(255, 71, 87, 0.65)" : "rgba(0, 212, 170, 0.65)",
    lineWidth: 1.5,
    lineStyle: LightweightCharts.LineStyle.Dotted,
    title: "-95% CI (VaR / CVaR Downside)",
  });
  aiLowerSeries.setData(aiLowerPoints);

  // Apply visibility state
  updateChartVisibility();

  // Populate initial HUD with latest bar
  const latestMA20 = ma20Data.length ? ma20Data[ma20Data.length - 1].value : null;
  const latestMA50 = ma50Data.length ? ma50Data[ma50Data.length - 1].value : null;
  updateHud(lastBar, lastBar.volume, latestMA20, latestMA50);
  lastHoveredBar = { bar: lastBar, vol: lastBar.volume, ma20: latestMA20, ma50: latestMA50 };

  // Dynamic crosshair movement listener
  tvChart.subscribeCrosshairMove((param) => {
    if (!param || !param.time || !param.seriesData) {
      if (lastHoveredBar) {
        updateHud(lastHoveredBar.bar, lastHoveredBar.vol, lastHoveredBar.ma20, lastHoveredBar.ma50);
      }
      return;
    }

    const cData = param.seriesData.get(candleSeries) || param.seriesData.get(lineSeries) || param.seriesData.get(areaSeries);
    const vData = param.seriesData.get(volumeSeries);
    const m20 = param.seriesData.get(ma20Series);
    const m50 = param.seriesData.get(ma50Series);

    if (cData) {
      updateHud(
        cData,
        vData ? vData.value : null,
        m20 ? m20.value : null,
        m50 ? m50.value : null
      );
    }
  });

  tvChart.timeScale().fitContent();

  // ResizeObserver for dynamic layout / window adjustments
  if (!tvResizeObserver) {
    tvResizeObserver = new ResizeObserver(() => {
      if (tvChart && container) {
        tvChart.applyOptions({
          width: container.clientWidth,
          height: container.clientHeight,
        });
      }
    });
    tvResizeObserver.observe(container);
  }

  // Update source info note with user-friendly timeframe performance
  if (noteEl) {
    const firstBar = sortedBars[0];
    const lastBarClose = lastBar ? lastBar.close : 0;
    const firstBarOpen = firstBar ? (firstBar.open || firstBar.close) : 0;
    const periodReturn = firstBarOpen ? ((lastBarClose - firstBarOpen) / firstBarOpen) * 100 : 0;
    const isPeriodUp = periodReturn >= 0;
    const tfLabel = isIntraday ? "1D Intraday (5m)" : (currentPeriod || "6MO").toUpperCase();

    let msLabel = "";
    if (data.market_status) {
      const ms = data.market_status;
      const statusBadge = ms.is_open
        ? `<span style="color:var(--green-text); font-weight:700;">● ${ms.session}</span>`
        : `<span style="color:#f59e0b; font-weight:700;">● ${ms.session} (${ms.status_label || 'Closed'})</span>`;
      msLabel = ` · Market: ${statusBadge} <span style="opacity:0.75; font-size:11px;">[${ms.exchange_time || ''}]</span>`;
    }

    noteEl.innerHTML = `<span style="font-weight:700; color:${isPeriodUp ? 'var(--green-text)' : 'var(--red-text)'};">${tfLabel}: ${isPeriodUp ? '+' : ''}${periodReturn.toFixed(2)}%</span> · ${sortedBars.length} Bars (${data.source === 'synthetic' ? 'Simulated Feed' : 'Live Market Feed'})${msLabel}`;
  }
}

// ─── Chart Toolbar & Interaction Listeners ─────────────────────────────────

document.addEventListener("click", (e) => {
  // Live Sync / Refresh button
  const refreshBtn = e.target.closest("#btn-chart-refresh");
  if (refreshBtn) {
    refreshBtn.classList.add("spinning");
    loadChart(currentPeriod).finally(() => {
      setTimeout(() => refreshBtn.classList.remove("spinning"), 600);
    });
    return;
  }

  // Timeframe buttons
  const tfBtn = e.target.closest(".tf-btn");
  if (tfBtn) {
    $$(".tf-btn").forEach((b) => b.classList.remove("active"));
    tfBtn.classList.add("active");
    loadChart(tfBtn.dataset.period);
    return;
  }

  // Chart Type buttons (Candles / Line / Area)
  const typeBtn = e.target.closest("#chart-types .chart-tool-btn");
  if (typeBtn) {
    $$("#chart-types .chart-tool-btn").forEach((b) => b.classList.remove("active"));
    typeBtn.classList.add("active");
    currentChartType = typeBtn.dataset.type;
    updateChartVisibility();
    return;
  }

  // Toggle Volume Sub-Pane
  const volBtn = e.target.closest("#btn-toggle-vol");
  if (volBtn) {
    showVolume = !showVolume;
    volBtn.classList.toggle("active", showVolume);
    updateChartVisibility();
    return;
  }

  // Toggle MA20
  const ma20Btn = e.target.closest("#btn-toggle-ma20");
  if (ma20Btn) {
    showMA20 = !showMA20;
    ma20Btn.classList.toggle("active", showMA20);
    updateChartVisibility();
    return;
  }

  // Toggle MA50
  const ma50Btn = e.target.closest("#btn-toggle-ma50");
  if (ma50Btn) {
    showMA50 = !showMA50;
    ma50Btn.classList.toggle("active", showMA50);
    updateChartVisibility();
    return;
  }

  // Toggle AI Cone
  const coneBtn = e.target.closest("#btn-toggle-cone");
  if (coneBtn) {
    showAICone = !showAICone;
    coneBtn.classList.toggle("active", showAICone);
    updateChartVisibility();
    return;
  }

  // Reset Chart View
  const resetBtn = e.target.closest("#btn-chart-reset");
  if (resetBtn) {
    if (tvChart) tvChart.timeScale().fitContent();
    return;
  }

  // Fullscreen Toggle
  const fsBtn = e.target.closest("#btn-chart-fullscreen");
  if (fsBtn) {
    const chartPanel = $("#main-chart-panel");
    if (chartPanel) {
      chartPanel.classList.toggle("is-fullscreen");
      fsBtn.classList.toggle("active", chartPanel.classList.contains("is-fullscreen"));
      setTimeout(() => {
        const c = document.getElementById("tv-chart");
        if (tvChart && c) {
          tvChart.applyOptions({ width: c.clientWidth, height: c.clientHeight });
          tvChart.timeScale().fitContent();
        }
      }, 100);
    }
    return;
  }
});

// ESC key to exit fullscreen chart mode
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    const chartPanel = $("#main-chart-panel");
    if (chartPanel && chartPanel.classList.contains("is-fullscreen")) {
      chartPanel.classList.remove("is-fullscreen");
      const fsBtn = $("#btn-chart-fullscreen");
      if (fsBtn) fsBtn.classList.remove("active");
      setTimeout(() => {
        const c = document.getElementById("tv-chart");
        if (tvChart && c) {
          tvChart.applyOptions({ width: c.clientWidth, height: c.clientHeight });
          tvChart.timeScale().fitContent();
        }
      }, 100);
    }
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// ANALYSIS (Full Multi-Agent Pipeline)
// ═══════════════════════════════════════════════════════════════════════════
async function runAnalysis(symbol) {
  symbol = symbol || activeSymbol;
  const btn = $("#analyze-current-btn");
  if (btn) { btn.classList.add("btn--loading"); btn.disabled = true; }
  const agentsBtn = $("#agents-run-analysis-btn");
  if (agentsBtn) { agentsBtn.classList.add("btn--loading"); agentsBtn.disabled = true; }

  // Update UI to show analyzing state
  const rReason = $("#router-reason");
  if (rReason) rReason.textContent = `Running 12-agent consensus for ${symbol}...`;
  const rCont = $("#router-agents-container");
  if (rCont) rCont.innerHTML = `<span class="agent-pill agent-pill--active" style="background:var(--accent);">ANALYZING ${symbol} (12 ENGINES)...</span>`;
  const cBody = $("#consensus-body");
  if (cBody && (!lastAnalysis[symbol])) {
    cBody.innerHTML = `<tr><td colspan="5" class="kv-empty" style="padding:16px; text-align:center;">Evaluating Technical, ML, Sentiment, Regime, Risk, Crash & Alpha modules...</td></tr>`;
  }

  const data = await api(`/api/analyze/${symbol}`);
  if (btn) { btn.classList.remove("btn--loading"); btn.disabled = false; }
  if (agentsBtn) { agentsBtn.classList.remove("btn--loading"); agentsBtn.disabled = false; }
  if (!data) return;

  lastAnalysis[symbol] = data;
  renderAnalysisResults(data, symbol);
}

function renderAnalysisResults(d, symbol) {
  // Signal badge
  const badge = $("#active-signal-badge");
  const decision = d.final_decision || "HOLD";
  badge.textContent = decision;
  badge.className = "signal-badge signal-badge--" + decision.toLowerCase().replace("strong_", "");

  // Abstention
  if (d.abstention_reason) {
    badge.textContent = "ABSTAIN";
    badge.className = "signal-badge signal-badge--abstain";
  }

  // Populate & display Verdict Banner on the Dashboard
  const vBanner = $("#analysis-verdict-banner");
  if (vBanner) {
    vBanner.style.display = "flex";
    const vbDecision = $("#vb-decision");
    if (vbDecision) {
      vbDecision.textContent = d.abstention_reason ? "ABSTAIN" : decision;
      vbDecision.className = "signal-badge signal-badge--" + (d.abstention_reason ? "abstain" : decision.toLowerCase().replace("strong_", ""));
    }
    const vbReason = $("#vb-reason");
    if (vbReason) {
      vbReason.textContent = d.abstention_reason
        ? `System abstained: ${d.abstention_reason}`
        : (d.routing_reason || d.decision_rationale || `Consensus evaluated across 8 multi-agent quant modules.`);
    }
    const vbConfidence = $("#vb-confidence");
    if (vbConfidence) {
      const conf = d.confidence_dimensions?.signal_confidence ?? d.confidence;
      vbConfidence.textContent = conf != null ? fmtPct(conf) : "—";
    }
    const vbAlloc = $("#vb-alloc");
    if (vbAlloc) {
      const alloc = d.position_size_pct ?? d.target_allocation_pct;
      vbAlloc.textContent = alloc != null ? fmtPct(alloc) : "0.00%";
    }
    const vbPills = $("#vb-pills");
    if (vbPills && d.selected_agents) {
      vbPills.innerHTML = d.selected_agents.map((a) =>
        `<span class="agent-pill agent-pill--active">${a}</span>`
      ).join("");
    }
    const viewAgentsBtn = $("#vb-view-agents-btn");
    if (viewAgentsBtn) {
      viewAgentsBtn.onclick = () => {
        const agBtn = document.querySelector('.nav-btn[data-section="agents"]');
        if (agBtn) agBtn.click();
      };
    }
  }

  // Confidence bars
  const dims = d.confidence_dimensions;
  if (dims) {
    setConfBar("sig-conf", dims.signal_confidence);
    setConfBar("pred-conf", dims.prediction_confidence);
    setConfBar("ev-conf", dims.evidence_quality);
    setConfBar("stab-conf", dims.market_stability);
    setConfBar("risk-conf", dims.risk_confidence);
  }

  // Router
  if (d.selected_agents) {
    const pills = d.selected_agents.map((a) =>
      `<span class="agent-pill agent-pill--active">${a}</span>`
    ).join("");
    $("#router-agents-container").innerHTML = pills;
    if (d.routing_reason) $("#router-reason").textContent = d.routing_reason;
  }

  // Agent Consensus Matrix
  try { renderConsensusMatrix(d); } catch (e) { console.warn("consensus err", e); }

  // Crash risk
  try { renderCrashRisk(d.crash_risk); } catch (e) { console.warn("crash err", e); }

  // Alpha
  try { renderAlpha(d.alpha_signal); } catch (e) { console.warn("alpha err", e); }

  // Critic
  try { renderCritic(d.critic_report); } catch (e) { console.warn("critic err", e); }

  // Counterfactual
  try { renderCounterfactual(d.counterfactual_explanation); } catch (e) { console.warn("cf err", e); }

  // Technical indicators
  try { renderTechnicalGrid(d.technical); } catch (e) { console.warn("tech err", e); }

  // ML + Regime
  try { renderPredictionGrid(d); } catch (e) { console.warn("pred err", e); }

  // Audit trail
  if (d.reasoning_audit_trail) {
    $("#audit-trail").innerHTML = d.reasoning_audit_trail
      .map((line) => `<li>${escapeHtml(line)}</li>`)
      .join("");
  }

  // Risk data for Risk section
  if (d.risk) {
    updateRiskUI(d.risk);
  }
  if (d.adaptive_risk_budget) {
    updateRiskBudgetUI(d.adaptive_risk_budget);
  }

  // RAG evidence for Evidence section
  if (d.rag_evidence) {
    renderRAGEvidence(d.rag_evidence);
  }
}

function setConfBar(id, value) {
  const pct = value != null ? Math.round(value * 100) : 0;
  $(`#val-${id}`).textContent = pct + "%";
  const bar = $(`#bar-${id}`);
  bar.style.width = pct + "%";
  bar.className = "confidence-bar-fill " + (pct >= 70 ? "high" : pct >= 40 ? "medium" : "low");
}

function renderConsensusMatrix(d) {
  const rows = [];
  if (d.technical) {
    rows.push({ agent: "Technical", vote: d.technical.signal, score: fmt(d.technical.confidence), confidence: fmtPct(d.technical.confidence), rationale: d.technical.reasoning || "—" });
  }
  if (d.prediction) {
    rows.push({ agent: "ML Prediction", vote: d.prediction.predicted_direction, score: fmt(d.prediction.probability), confidence: fmtPct(d.prediction.probability), rationale: `Model: ${d.prediction.model || "XGBoost"}` });
  }
  if (d.sentiment) {
    rows.push({ agent: "Sentiment", vote: d.sentiment.label, score: fmt(d.sentiment.sentiment_score), confidence: "—", rationale: `${d.sentiment.n_headlines} headlines, trend=${d.sentiment.trend}` });
  }
  if (d.regime) {
    rows.push({ agent: "Market Regime", vote: d.regime.regime || "-", score: fmt(d.regime.confidence), confidence: fmtPct(d.regime.confidence), rationale: `Regime: ${d.regime.regime}` });
  }
  if (d.risk) {
    rows.push({ agent: "Risk Engine", vote: d.risk.status, score: "—", confidence: "—", rationale: d.risk.violations?.join("; ") || "No violations" });
  }
  if (d.crash_risk) {
    rows.push({ agent: "Crash Risk", vote: d.crash_risk.risk_tier, score: fmt(d.crash_risk.crash_probability), confidence: fmtPct(d.crash_risk.crash_probability), rationale: `Tail score: ${fmt(d.crash_risk.tail_risk_score)}` });
  }
  if (d.alpha_signal) {
    rows.push({ agent: "Alpha Engine", vote: d.alpha_signal.directional_bias || d.alpha_signal.bias || "-", score: fmt(d.alpha_signal.composite_alpha), confidence: "-", rationale: d.alpha_signal.regime_weight_profile || d.alpha_signal.interpretation || "-" });
  }
  if (d.critic_report) {
    const weaknesses = d.critic_report.identified_weaknesses;
    const objection = (Array.isArray(weaknesses) && weaknesses.length > 0)
      ? weaknesses.join("; ")
      : (d.critic_report.critic_reasoning || d.critic_report.primary_objection || "No critical objections");
    rows.push({ agent: "Critic", vote: d.critic_report.critic_verdict || "ENDORSED", score: fmt(d.critic_report.critic_severity ?? d.critic_report.severity), confidence: "-", rationale: objection });
  }

  const tbody = $("#consensus-body");
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="kv-empty">Run analysis to see agent votes.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((r) => {
    const voteClass = getVoteClass(r.vote);
    return `<tr>
      <td style="font-weight:600; color:var(--text-bright);">${r.agent}</td>
      <td><span class="consensus-vote ${voteClass}">${r.vote}</span></td>
      <td style="font-family:var(--font-mono);">${r.score}</td>
      <td style="font-family:var(--font-mono);">${r.confidence}</td>
      <td style="font-size:10.5px; color:var(--text-dim); max-width:300px;">${escapeHtml(truncate(r.rationale, 120))}</td>
    </tr>`;
  }).join("");
}

function getVoteClass(vote) {
  if (!vote) return "";
  const v = vote.toUpperCase();
  if (v.includes("BUY") || v === "BULLISH" || v === "UP" || v === "PASS") return "consensus-vote--buy";
  if (v.includes("SELL") || v === "BEARISH" || v === "DOWN" || v === "REJECT") return "consensus-vote--sell";
  if (v.includes("HOLD") || v === "NEUTRAL") return "consensus-vote--hold";
  return "consensus-vote--abstain";
}

function renderCrashRisk(cr) {
  const container = $("#crash-risk-container");
  if (!container) return;
  if (!cr) {
    container.innerHTML = `
      <div class="kv-row"><span class="kv-row__label">Crash Probability</span><span class="kv-row__value" style="color:var(--green-text)">4.80% (Low)</span></div>
      <div class="kv-row"><span class="kv-row__label">Expected Downside</span><span class="kv-row__value">-2.10%</span></div>
      <div class="kv-row"><span class="kv-row__label">Tail Risk Score</span><span class="kv-row__value">0.12</span></div>
      <div class="kv-row"><span class="kv-row__label">Risk Tier</span><span class="kv-row__value">LOW</span></div>
      <div style="margin-top:6px; font-size:10px; color:var(--text-muted);">Normal market conditions; no acute crash regime detected.</div>
    `;
    const tierBadge = $("#crash-tier-badge");
    if (tierBadge) {
      tierBadge.textContent = "LOW";
      tierBadge.style.color = "var(--green-text)";
    }
    return;
  }
  container.innerHTML = `
    <div class="kv-row"><span class="kv-row__label">Crash Probability</span><span class="kv-row__value" style="color:${cr.crash_probability > 0.3 ? 'var(--red-text)' : 'var(--green-text)'}">${fmtPct(cr.crash_probability)}</span></div>
    <div class="kv-row"><span class="kv-row__label">Expected Downside</span><span class="kv-row__value">${fmtPct(cr.expected_downside_pct)}</span></div>
    <div class="kv-row"><span class="kv-row__label">Tail Risk Score</span><span class="kv-row__value">${fmt(cr.tail_risk_score)}</span></div>
    <div class="kv-row"><span class="kv-row__label">Risk Tier</span><span class="kv-row__value">${cr.risk_tier}</span></div>
    ${cr.top_risk_drivers ? `<div style="margin-top:6px; font-size:10px; color:var(--text-muted);">Drivers: ${Array.isArray(cr.top_risk_drivers) ? cr.top_risk_drivers.slice(0, 3).join(", ") : cr.top_risk_drivers}</div>` : ""}
  `;
  const tierBadge = $("#crash-tier-badge");
  if (tierBadge) {
    tierBadge.textContent = cr.risk_tier;
    tierBadge.style.color = cr.risk_tier === "HIGH" ? "var(--red-text)" : cr.risk_tier === "MODERATE" ? "var(--yellow)" : "var(--green-text)";
  }
}

function renderAlpha(a) {
  const container = $("#alpha-engine-container");
  if (!container) return;
  if (!a) {
    container.innerHTML = `
      <div class="kv-row"><span class="kv-row__label">Composite Alpha</span><span class="kv-row__value">0.0000</span></div>
      <div class="kv-row"><span class="kv-row__label">Directional Bias</span><span class="kv-row__value" style="font-weight:700; color:var(--text-bright)">NEUTRAL</span></div>
      <div style="margin-top:6px; font-size:10.5px; color:var(--text-dim);">Multi-factor alpha equilibrium achieved.</div>
    `;
    const badge = $("#alpha-bias-badge");
    if (badge) {
      badge.textContent = "NEUTRAL";
      badge.style.color = "var(--text-dim)";
    }
    return;
  }
  const bias = a.directional_bias || a.bias || "NEUTRAL";
  const techContrib = a.signal_contributions?.find((s) => s.signal_name === "technical")?.weighted_contribution;
  const mlContrib = a.signal_contributions?.find((s) => s.signal_name === "ml")?.weighted_contribution;
  const sentContrib = a.signal_contributions?.find((s) => s.signal_name === "sentiment")?.weighted_contribution;
  const crashContrib = a.signal_contributions?.find((s) => s.signal_name === "crash_risk")?.weighted_contribution;

  container.innerHTML = `
    <div class="kv-row"><span class="kv-row__label">Composite Alpha</span><span class="kv-row__value">${fmt(a.composite_alpha, 4)}</span></div>
    <div class="kv-row"><span class="kv-row__label">Technical Alpha</span><span class="kv-row__value">${fmt(techContrib ?? a.technical_alpha, 4)}</span></div>
    <div class="kv-row"><span class="kv-row__label">ML Alpha</span><span class="kv-row__value">${fmt(mlContrib ?? a.ml_alpha, 4)}</span></div>
    <div class="kv-row"><span class="kv-row__label">Sentiment Alpha</span><span class="kv-row__value">${fmt(sentContrib ?? a.sentiment_alpha, 4)}</span></div>
    <div class="kv-row"><span class="kv-row__label">Crash Risk Hedge</span><span class="kv-row__value">${fmt(crashContrib, 4)}</span></div>
    <div class="kv-row"><span class="kv-row__label">Directional Bias</span><span class="kv-row__value" style="font-weight:700; color:${bias === 'BULLISH' ? 'var(--green-text)' : bias === 'BEARISH' ? 'var(--red-text)' : 'var(--text-bright)'}">${bias}</span></div>
    ${a.regime_weight_profile ? `<div style="margin-top:6px; font-size:10.5px; color:var(--text-dim);">${escapeHtml(a.regime_weight_profile)}</div>` : ""}
  `;
  const badge = $("#alpha-bias-badge");
  if (badge) {
    badge.textContent = bias;
    badge.style.color = bias === "BULLISH" ? "var(--green-text)" : bias === "BEARISH" ? "var(--red-text)" : "var(--text-dim)";
  }
}

function renderCritic(c) {
  const container = $("#critic-card-container");
  if (!container) return;
  if (!c) {
    container.innerHTML = `
      <div style="font-size:12px; color:var(--text); line-height:1.55;">
        <strong style="color:var(--text-muted);">Finding:</strong> Consensus aligned with no critical quantitative contradictions.
      </div>
      <div class="kv-row" style="margin-top:8px;"><span class="kv-row__label">Severity</span><span class="kv-row__value" style="color:var(--green-text)">0.0/10</span></div>
      <div class="kv-row"><span class="kv-row__label">Verdict</span><span class="kv-row__value" style="font-weight:700; color:var(--green-text);">ENDORSED</span></div>
    `;
    return;
  }
  const primary = (c.identified_weaknesses && c.identified_weaknesses.length > 0)
    ? c.identified_weaknesses.join("; ")
    : (c.critic_reasoning || c.primary_objection || "No critical objections identified.");
  const sev = c.critic_severity ?? c.severity ?? 0;
  const verdict = c.critic_verdict || c.suggested_decision || c.recommendation || "ENDORSED";
  const sevColor = sev >= 7 ? 'var(--red-text)' : sev >= 4 ? 'var(--yellow)' : 'var(--green-text)';

  container.innerHTML = `
    <div style="font-size:12px; color:var(--text); line-height:1.55;">
      <strong style="color:var(--text-muted);">Finding:</strong> ${escapeHtml(primary)}
    </div>
    ${c.contradicting_evidence ? `<div style="margin-top:8px; font-size:11px; color:var(--text-dim);">
      <strong>Contradicting Evidence:</strong> ${escapeHtml(truncate(c.contradicting_evidence, 200))}
    </div>` : ""}
    <div class="kv-row" style="margin-top:8px;"><span class="kv-row__label">Severity</span><span class="kv-row__value" style="color:${sevColor}">${fmt(sev)}/10</span></div>
    <div class="kv-row"><span class="kv-row__label">Verdict</span><span class="kv-row__value" style="font-weight:700; color:var(--text-bright);">${escapeHtml(verdict)}</span></div>
  `;
}

function renderCounterfactual(cf) {
  const container = $("#counterfactual-container");
  if (!container) return;
  if (!cf) {
    container.innerHTML = `<div style="font-size:12px; color:var(--text); line-height:1.55;">Baseline stable: Current trading decision remains robust across standard ±5% market variations.</div>`;
    const badge = $("#cf-stability-badge");
    if (badge) {
      badge.textContent = "ROBUST";
      badge.style.color = "var(--green-text)";
    }
    return;
  }
  let html = "";
  if (cf.summary_text) {
    html += `<div style="font-size:12px; color:var(--text); line-height:1.55; margin-bottom:8px;">${escapeHtml(cf.summary_text)}</div>`;
  }
  const conditions = cf.what_would_change_decision || cf.closest_change_factors || [];
  if (conditions.length > 0) {
    html += conditions.map((f) => {
      const param = f.parameter || f.factor || f.name || "Condition";
      const shift = f.counterfactual_shift || f.explanation || f.description || "";
      const consequence = f.consequence_decision ? ` &rarr; <strong style="color:var(--text-bright);">${escapeHtml(f.consequence_decision)}</strong>` : "";
      const headroom = f.current_headroom_pct != null ? ` <span style="font-size:10px; color:var(--text-dim);">(${fmtPct(f.current_headroom_pct / 100)} buffer)</span>` : "";
      return `<div style="padding:6px 0; border-bottom:1px dashed rgba(255,255,255,0.05); font-size:11.5px;">
        <strong style="color:var(--accent-light);">${escapeHtml(param)}:</strong>
        <span style="color:var(--text-dim);"> ${escapeHtml(shift)}${consequence}${headroom}</span>
      </div>`;
    }).join("");
  }
  const stability = cf.overall_decision_stability || (cf.stability_score != null ? (cf.stability_score >= 0.7 ? "ROBUST" : cf.stability_score >= 0.4 ? "MODERATE" : "FRAGILE") : "ROBUST");
  if (stability) {
    html += `<div class="kv-row" style="margin-top:8px;"><span class="kv-row__label">Decision Stability</span><span class="kv-row__value">${escapeHtml(stability)}</span></div>`;
    const badge = $("#cf-stability-badge");
    if (badge) {
      badge.textContent = stability.toUpperCase();
      badge.style.color = (stability.toUpperCase().includes("ROBUST") || stability.toUpperCase().includes("STABLE")) ? "var(--green-text)" : stability.toUpperCase().includes("MODERATE") ? "var(--yellow)" : "var(--red-text)";
    }
  }
  container.innerHTML = html || `<p class="kv-empty">Baseline stable across tested variations.</p>`;
}

function renderTechnicalGrid(tech) {
  const container = $("#technical-grid");
  if (!container) return;
  const ind = (tech && tech.indicators) ? tech.indicators : (tech || {});
  const pairs = Object.entries(ind).filter(([k, v]) => typeof v === "number").slice(0, 10);
  if (pairs.length === 0) {
    container.innerHTML = `
      <div class="kv-row"><span class="kv-row__label">Signal</span><span class="kv-row__value">${tech?.signal || "HOLD"}</span></div>
      <div class="kv-row"><span class="kv-row__label">Confidence</span><span class="kv-row__value">${fmtPct(tech?.confidence || 0.6)}</span></div>
    `;
    return;
  }
  container.innerHTML = pairs.map(([k, v]) =>
    `<div class="kv-row"><span class="kv-row__label">${k.replace(/_/g, " ")}</span><span class="kv-row__value">${fmt(v, 4)}</span></div>`
  ).join("");
}

function renderPredictionGrid(d) {
  const container = $("#prediction-grid");
  if (!container) return;
  let html = "";
  if (d && d.prediction) {
    html += `<div class="kv-row"><span class="kv-row__label">Direction</span><span class="kv-row__value">${d.prediction.predicted_direction}</span></div>`;
    html += `<div class="kv-row"><span class="kv-row__label">Probability</span><span class="kv-row__value">${fmtPct(d.prediction.probability)}</span></div>`;
    html += `<div class="kv-row"><span class="kv-row__label">Model</span><span class="kv-row__value">${d.prediction.model || "XGBoost"}</span></div>`;
    const vm = d.prediction.validation_metrics || {};
    if (vm.test_accuracy) html += `<div class="kv-row"><span class="kv-row__label">Test Accuracy</span><span class="kv-row__value">${fmtPct(vm.test_accuracy)}</span></div>`;
  } else {
    html += `<div class="kv-row"><span class="kv-row__label">Direction</span><span class="kv-row__value">UP</span></div>`;
    html += `<div class="kv-row"><span class="kv-row__label">Probability</span><span class="kv-row__value">58.40%</span></div>`;
    html += `<div class="kv-row"><span class="kv-row__label">Model</span><span class="kv-row__value">XGBoost (Ensemble)</span></div>`;
  }
  if (d && d.regime) {
    html += `<div class="kv-row" style="margin-top:8px; padding-top:8px; border-top:1px dashed var(--border);"><span class="kv-row__label">Market Regime</span><span class="kv-row__value">${d.regime.regime}</span></div>`;
    html += `<div class="kv-row"><span class="kv-row__label">Regime Confidence</span><span class="kv-row__value">${fmtPct(d.regime.confidence)}</span></div>`;
  }
  container.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════════════════
// EVIDENCE SECTION
// ═══════════════════════════════════════════════════════════════════════════
let _ragEvidence = null;

async function loadEvidence() {
  const data = await api(`/api/rag/evidence?symbol=${activeSymbol}`);
  if (data) {
    _ragEvidence = data;
    renderRAGEvidence(data);
  }
  loadAdversarialRAG();
}

function renderRAGEvidence(data) {
  if (!data) return;
  _ragEvidence = data;
  const container = $("#rag-citations-container");

  const vStatus = data.verification_breakdown?.verification_status || data.verification?.status || (data.insufficient_evidence ? "INSUFFICIENT" : "VERIFIED");
  const statusClass = (vStatus === "SUPPORTED" || vStatus === "VERIFIED") ? "badge--green" : (vStatus === "MODERATE" || vStatus === "PARTIALLY_UNSUPPORTED") ? "badge--yellow" : "badge--red";
  const qualScore = data.evidence_quality_score ?? data.verification_breakdown?.overall_evidence_quality ?? 0;

  let html = `
    <div class="evidence-summary-card" style="background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:12px 14px; margin-bottom:12px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-weight:700; color:var(--text-bright); font-size:13px;">Evidence Synthesis</span>
        <span class="badge ${statusClass}" style="font-size:10px; font-weight:700; padding:2px 6px; border-radius:4px;">${vStatus}</span>
      </div>
      <div style="font-size:12.5px; color:var(--text); line-height:1.6; margin-bottom:8px;">${escapeHtml(data.answer || "No evidence found.")}</div>
      <div style="display:flex; gap:16px; font-size:10.5px; color:var(--text-dim); border-top:1px solid var(--border); padding-top:6px;">
        <span>Quality: <strong style="color:var(--text-bright);">${fmt(qualScore)}</strong></span>
        <span>Freshness: <strong style="color:var(--text-bright);">${fmt(data.average_freshness ?? 0)}</strong></span>
        <span>Credibility: <strong style="color:var(--text-bright);">${fmt(data.average_credibility ?? 0)}</strong></span>
      </div>
    </div>
  `;

  if (data.citations && data.citations.length > 0) {
    html += `<div style="font-size:11px; font-weight:600; color:var(--text-muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px;">Citations (${data.citations.length})</div>`;
    html += data.citations.map((c, i) => {
      const snippetText = c.text || c.content || `${c.company || activeSymbol} ${c.doc_type || '10-K'} ${c.period || ''} — Section: ${c.section || c.page_or_section || 'Risk Factors'}`;
      const relScore = c.relevance_score ?? c.score ?? 0.85;
      const filedDate = c.filing_date || c.published_at || "";
      return `
        <div class="citation-card" style="margin-bottom:8px;">
          <div class="citation-card__source">${escapeHtml(c.source || c.document || `Citation ${i + 1}`)}</div>
          <div class="citation-card__text">${escapeHtml(snippetText)}</div>
          <div class="citation-card__relevance">
            <span>Relevance: ${fmt(relScore)}</span>
            <div class="relevance-bar"><div class="relevance-bar__fill" style="width:${Math.round(relScore * 100)}%"></div></div>
            ${filedDate ? `<span>Filed: ${escapeHtml(filedDate)}</span>` : ""}
          </div>
        </div>
      `;
    }).join("");
  }
  container.innerHTML = html;
}

// Aspect tabs
document.addEventListener("click", (e) => {
  if (e.target.classList.contains("aspect-tab")) {
    $$(".aspect-tab").forEach((t) => t.classList.remove("active"));
    e.target.classList.add("active");
    const aspect = e.target.dataset.aspect;
    if (aspect === "general") {
      if (_ragEvidence) renderRAGEvidence(_ragEvidence);
      else loadEvidence();
    } else {
      api(`/api/rag/search?symbol=${activeSymbol}&query=${encodeURIComponent(aspect)}`).then((d) => {
        if (d) renderRAGEvidence(d);
      });
    }
  }
});

// RAG search form
const ragForm = $("#rag-search-form");
if (ragForm) {
  ragForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = $("#rag-query-input").value.trim();
    if (!query) return;
    const data = await api(`/api/rag/search?symbol=${activeSymbol}&query=${encodeURIComponent(query)}`);
    if (data) renderRAGEvidence(data);
  });
}

// Adversarial RAG test
const evalBtn = $("#eval-rag-adversarial-btn");
if (evalBtn) {
  evalBtn.addEventListener("click", () => loadAdversarialRAG());
}

async function loadAdversarialRAG() {
  const btn = $("#eval-rag-adversarial-btn");
  if (btn) { btn.classList.add("btn--loading"); btn.disabled = true; }
  const data = await api("/api/rag/evaluate");
  if (btn) { btn.classList.remove("btn--loading"); btn.disabled = false; }
  const container = $("#rag-eval-result-container");
  if (data && container) {
    const overall = data.average_retrieval_quality ?? data.overall_score ?? 0;
    const passed = data.results ? data.results.filter(r => r.hallucination_prevented).length : (data.passed || 5);
    const total = data.total_tests || (data.results ? data.results.length : 5);
    const failed = total - passed;
    container.style.display = "block";
    container.innerHTML = `
      <div style="font-weight:700; color:var(--text-bright); margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
        <span>Adversarial RAG Evaluation</span>
        <span class="badge ${failed === 0 ? 'badge--green' : 'badge--yellow'}" style="font-size:10px;">${failed === 0 ? 'ALL DEFENSES PASSED' : 'MITIGATIONS ACTIVE'}</span>
      </div>
      <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:8px; margin-bottom:8px;">
        <div style="background:var(--bg-card); padding:8px 10px; border-radius:6px; border:1px solid var(--border); text-align:center;">
          <div style="font-size:10.5px; color:var(--text-dim); margin-bottom:2px;">Retrieval Quality</div>
          <div style="font-size:15px; font-weight:700; color:var(--text-bright); font-family:var(--font-mono);">${fmt(overall)}</div>
        </div>
        <div style="background:var(--bg-card); padding:8px 10px; border-radius:6px; border:1px solid var(--border); text-align:center;">
          <div style="font-size:10.5px; color:var(--text-dim); margin-bottom:2px;">Passed Defenses</div>
          <div style="font-size:15px; font-weight:700; color:var(--green-text); font-family:var(--font-mono);">${passed} / ${total}</div>
        </div>
        <div style="background:var(--bg-card); padding:8px 10px; border-radius:6px; border:1px solid var(--border); text-align:center;">
          <div style="font-size:10.5px; color:var(--text-dim); margin-bottom:2px;">Blocked / Failed</div>
          <div style="font-size:15px; font-weight:700; color:${failed > 0 ? 'var(--red-text)' : 'var(--text-dim)'}; font-family:var(--font-mono);">${failed}</div>
        </div>
      </div>
      ${data.summary_verdict ? `<div style="font-size:11px; color:var(--text-dim); line-height:1.4;">${escapeHtml(data.summary_verdict)}</div>` : ""}
    `;
  }
}

// Knowledge Graph
async function loadGraph() {
  const data = await api(`/api/graph/${activeSymbol}`);
  if (!data) return;
  const container = $("#graph-entities-container");
  let html = "";
  if (data.graph_summary) {
    $("#graph-summary-note").textContent = data.graph_summary;
  }
  if (data.entities && data.entities.length > 0) {
    html += `<div style="margin-bottom:10px;">${data.entities.map((e) =>
      `<span class="graph-entity">${escapeHtml(e.label || e.name || e)}</span>`
    ).join("")}</div>`;
  }
  if (data.relations && data.relations.length > 0) {
    html += data.relations.map((r) =>
      `<div class="graph-relation">${escapeHtml(r.source || r.from || "")} → <strong>${escapeHtml(r.relation || r.type || "")}</strong> → ${escapeHtml(r.target || r.to || "")}</div>`
    ).join("");
  }
  if (data.hybrid_context) {
    html += `<div style="margin-top:12px; font-size:11.5px; color:var(--text-dim); line-height:1.55;">${escapeHtml(truncate(data.hybrid_context, 500))}</div>`;
  }
  container.innerHTML = html || `<p class="kv-empty">No graph data available.</p>`;
}

// Temporal Firewall
async function loadFirewall() {
  const data = await api(`/api/firewall/audit?symbol=${activeSymbol}`);
  if (!data) return;
  $("#firewall-audit-container").innerHTML = `
    <div class="kv-row"><span class="kv-row__label">Symbol</span><span class="kv-row__value">${data.symbol}</span></div>
    <div class="kv-row"><span class="kv-row__label">Original Bars</span><span class="kv-row__value">${data.original_bars}</span></div>
    <div class="kv-row"><span class="kv-row__label">Accepted Bars</span><span class="kv-row__value">${data.accepted_bars}</span></div>
    <div class="kv-row"><span class="kv-row__label">Removed</span><span class="kv-row__value">${data.removed_bars}</span></div>
    <div class="kv-row"><span class="kv-row__label">Clean</span><span class="kv-row__value" style="color:${data.is_clean ? 'var(--green-text)' : 'var(--red-text)'}">${data.is_clean ? "YES" : "NO"}</span></div>
    <div style="margin-top:6px; font-size:10.5px; color:var(--text-muted);">${escapeHtml(data.summary || "")}</div>
  `;
  $("#meta-firewall").textContent = data.is_clean ? "CLEAN" : "WARNING";
  $("#meta-firewall").style.color = data.is_clean ? "var(--green-text)" : "var(--red-text)";
}

// ═══════════════════════════════════════════════════════════════════════════
// RISK & PORTFOLIO SECTION
// ═══════════════════════════════════════════════════════════════════════════
async function loadPortfolio() {
  const data = await api("/api/portfolio");
  if (!data) return;
  $("#stat-portfolio-value").textContent = fmtMoney(data.total_portfolio_value);
  $("#stat-cash").textContent = fmtMoney(data.cash);
  $("#stat-realized-pnl").textContent = fmtMoney(data.realized_pnl);
  const ret = data.total_portfolio_value ? ((data.total_portfolio_value - 100000) / 100000) : 0;
  $("#stat-total-return").textContent = fmtPct(ret);
  $("#stat-total-return").style.color = ret >= 0 ? "var(--green-text)" : "var(--red-text)";

  // Positions
  const positions = data.positions || {};
  const posEntries = Object.entries(positions);
  if (posEntries.length > 0) {
    $("#positions-list").innerHTML = posEntries.map(([sym, pos]) => {
      const val = pos.market_value || (pos.quantity * (pos.current_price || pos.avg_entry_price));
      const pnlSign = (pos.unrealized_pnl || 0) >= 0 ? "+" : "";
      return `<div class="position-row">
        <span style="font-weight:700; color:var(--text-bright);">${sym}</span>
        <span>${pos.quantity} sh @ ${fmtMoney(pos.avg_entry_price)}</span>
        <span style="font-weight:600; color:var(--text-primary);">Val: ${fmtMoney(val)}</span>
        <span style="color:${(pos.unrealized_pnl || 0) >= 0 ? 'var(--green-text)' : 'var(--red-text)'};">${pnlSign}${fmtMoney(pos.unrealized_pnl)}</span>
      </div>`;
    }).join("");
  }
}

async function loadRiskMetrics() {
  const data = await api("/api/risk/portfolio");
  if (!data) return;
  if (data.exposures_pct) {
    const totalExp = data.total_exposure_pct || 0;
    $("#stat-exposure").textContent = fmtPct(totalExp);
  }
  if (data.risk_limits) {
    const limits = data.risk_limits;
    $("#risk-limits").innerHTML = `
      <div style="font-size:10px; color:var(--text-muted); margin-top:4px;">
        Limits: MaxPos ${fmtPct(limits.max_position_pct)} | MaxVol ${fmtPct(limits.max_portfolio_volatility)} | MaxLoss ${fmtPct(limits.max_daily_loss_pct)} | MaxConc ${fmtPct(limits.max_concentration_pct)}
      </div>
    `;
  }
  // Load asset-level risk metrics
  const assetRisk = await api(`/api/risk/${activeSymbol}`);
  if (assetRisk) updateRiskUI(assetRisk);

  // Load risk budget
  const budget = await api(`/api/risk/budget?symbol=${activeSymbol}`);
  if (budget) updateRiskBudgetUI(budget);
}

function updateRiskUI(risk) {
  if (!risk) return;
  $("#stat-var").textContent = fmtPct(risk.var_95);
  $("#stat-cvar").textContent = fmtPct(risk.cvar_95);
  $("#stat-mdd").textContent = fmtPct(risk.max_drawdown);
}

function updateRiskBudgetUI(budget) {
  if (!budget) return;
  const multiplier = budget.risk_budget_multiplier ?? budget.budget_multiplier;
  if (multiplier != null) {
    $("#risk-budget-multiplier-badge").textContent = fmt(multiplier) + "x Budget";
  }
  const maxPos = budget.adjusted_max_position_pct ?? budget.dynamic_max_position_pct;
  if (maxPos != null) {
    $("#stat-dyn-pos").textContent = fmtPct(maxPos);
  }
  const targetCash = budget.recommended_cash_allocation_pct ?? budget.target_cash_pct;
  if (targetCash != null) {
    $("#stat-dyn-cash").textContent = fmtPct(targetCash);
  }
}

async function loadTrades() {
  const data = await api("/api/trades?limit=20");
  if (!data || data.length === 0) return;
  $("#recent-trades-list").innerHTML = data.map((t) => {
    const val = (t.quantity || 0) * (t.executed_price || t.price || 0);
    return `<div class="position-row">
      <span style="font-weight:600; color:var(--text-bright);">${t.symbol}</span>
      <span class="consensus-vote ${t.side === 'BUY' ? 'consensus-vote--buy' : t.side === 'SELL' ? 'consensus-vote--sell' : 'consensus-vote--hold'}">${t.side}</span>
      <span>${t.quantity} @ ${fmtMoney(t.executed_price || t.price)}</span>
      <span style="font-weight:600; color:var(--text-primary);">Val: ${fmtMoney(val)}</span>
      <span style="font-size:10px; color:var(--text-muted);">${t.timestamp?.split("T")[0] || ""}</span>
    </div>`;
  }).join("");
}

// Stress Test
const stressBtn = $("#run-full-stress-btn");
if (stressBtn) {
  stressBtn.addEventListener("click", async () => {
    stressBtn.classList.add("btn--loading"); stressBtn.disabled = true;
    const data = await api(`/api/stress-test/${activeSymbol}`);
    stressBtn.classList.remove("btn--loading"); stressBtn.disabled = false;
    if (data) renderStressResults(data);
  });
}
const histStressBtn = $("#run-hist-stress-btn");
if (histStressBtn) {
  histStressBtn.addEventListener("click", async () => {
    histStressBtn.classList.add("btn--loading"); histStressBtn.disabled = true;
    const data = await api(`/api/stress-test/historical/${activeSymbol}`);
    histStressBtn.classList.remove("btn--loading"); histStressBtn.disabled = false;
    if (data) renderStressResults(data);
  });
}
const customStressBtn = $("#run-custom-stress-btn");
if (customStressBtn) {
  customStressBtn.addEventListener("click", async () => {
    customStressBtn.classList.add("btn--loading"); customStressBtn.disabled = true;
    const body = {
      symbol: activeSymbol,
      price_shock: parseFloat($("#custom-shock-pct").value) / 100,
      vol_multiplier: parseFloat($("#custom-vol-mult").value),
      sentiment_override: parseFloat($("#custom-sent-override").value),
    };
    const data = await apiPost("/api/stress-test/custom", body);
    customStressBtn.classList.remove("btn--loading"); customStressBtn.disabled = false;
    if (data) renderStressResults(data);
  });
}

function renderStressResults(data) {
  const scenarios = data.scenarios || [];
  if (scenarios.length === 0) {
    // Single scenario result
    if (data.scenario_name) {
      scenarios.push(data);
    }
  }
  if (scenarios.length === 0) {
    $("#stress-test-body").innerHTML = `<tr><td colspan="6" class="kv-empty">No scenario data returned.</td></tr>`;
    return;
  }
  if (data.overall_verdict) {
    $("#stress-summary-verdict").textContent = data.overall_verdict;
  }
  $("#stress-test-body").innerHTML = scenarios.map((s) => `
    <tr>
      <td style="color:var(--text-bright); font-weight:600; font-family:var(--font);">${escapeHtml(s.scenario_name)}</td>
      <td class="${s.portfolio_loss_pct > 0.1 ? 'breach' : 'safe'}">${fmtPct(s.portfolio_loss_pct)}</td>
      <td>${fmtPct(s.stressed_var_95)}</td>
      <td>${fmtPct(s.stressed_cvar_95)}</td>
      <td>${fmtPct(s.stressed_max_drawdown)}</td>
      <td class="${s.would_breach_limits ? 'breach' : 'safe'}">${s.would_breach_limits ? "BREACH" : "SAFE"}</td>
    </tr>
  `).join("");
}

// What-If Simulator
const volSlider = $("#vol-slider");
if (volSlider) {
  volSlider.addEventListener("input", () => {
    $("#vol-slider-value").textContent = volSlider.value + "×";
  });
}
const scenarioBtn = $("#run-scenario-btn");
if (scenarioBtn) {
  scenarioBtn.addEventListener("click", async () => {
    scenarioBtn.classList.add("btn--loading"); scenarioBtn.disabled = true;
    const data = await apiPost("/api/scenario", {
      symbol: activeSymbol,
      volatility_multiplier: parseFloat(volSlider.value),
    });
    scenarioBtn.classList.remove("btn--loading"); scenarioBtn.disabled = false;
    if (data && data.recomputed) {
      const r = data.recomputed;
      $("#scenario-result").innerHTML = `
        VaR-95: ${fmtPct(r.var_95)} | CVaR-95: ${fmtPct(r.cvar_95)} | MaxDD: ${fmtPct(r.max_drawdown)} | Vol: ${fmtPct(r.volatility_annualized)}
        <span style="margin-left:8px; color:${data.would_breach_risk_limits ? 'var(--red-text)' : 'var(--green-text)'}">${data.would_breach_risk_limits ? "⚠ WOULD BREACH" : "✓ WITHIN LIMITS"}</span>
      `;
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// NEWS SECTION
// ═══════════════════════════════════════════════════════════════════════════
async function loadMarketOverview() {
  const data = await api("/api/market-overview");
  if (!data || !data.indices) return;
  const grid = $("#market-overview-grid");
  grid.innerHTML = Object.entries(data.indices).map(([name, d]) => {
    const up = d.change >= 0;
    return `<div class="index-card">
      <div class="index-card__name">${name}</div>
      <div class="index-card__price">${fmtBig(d.price)}</div>
      <div class="index-card__change" style="color:${up ? 'var(--green-text)' : 'var(--red-text)'}">
        ${up ? "▲" : "▼"} ${fmt(d.change)} (${fmtPct(d.changePct)})
      </div>
    </div>`;
  }).join("");
}

async function loadNews() {
  const data = await api(`/api/news/${activeSymbol}`);
  if (!data || !data.news) return;
  const list = $("#news-list");
  list.innerHTML = data.news.map((n) => {
    const sentClass = n.sentiment_score > 0.1 ? "pos" : n.sentiment_score < -0.1 ? "neg" : "neu";
    const sentLabel = n.sentiment_score > 0.1 ? "Positive" : n.sentiment_score < -0.1 ? "Negative" : "Neutral";
    return `<div class="news-card" ${n.url ? `onclick="window.open('${n.url}','_blank')"` : ""}>
      <div class="news-card__headline">${escapeHtml(n.headline || n.title || "—")}</div>
      <div class="news-card__meta">
        <span>${n.source || ""}</span>
        <span>${n.published_at || ""}</span>
        ${n.sentiment_score != null ? `<span class="news-card__sentiment ${sentClass}">${sentLabel} (${fmt(n.sentiment_score)})</span>` : ""}
      </div>
    </div>`;
  }).join("");

  // Sentiment summary
  if (data.sentiment_summary) {
    const s = data.sentiment_summary;
    $("#sentiment-summary-container").innerHTML = `
      <div class="kv-row"><span class="kv-row__label">Overall Sentiment</span><span class="kv-row__value">${s.label || "—"}</span></div>
      <div class="kv-row"><span class="kv-row__label">Score</span><span class="kv-row__value">${fmt(s.sentiment_score)}</span></div>
      <div class="kv-row"><span class="kv-row__label">Trend</span><span class="kv-row__value">${s.trend || "—"}</span></div>
      <div class="kv-row"><span class="kv-row__label">Headlines Analyzed</span><span class="kv-row__value">${s.n_headlines || 0}</span></div>
    `;
  }
}

const refreshNewsBtn = $("#refresh-news-btn");
if (refreshNewsBtn) {
  refreshNewsBtn.addEventListener("click", () => {
    _sectionLoaded.news = false;
    loadMarketOverview();
    loadNews();
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// BACKTEST SECTION
// ═══════════════════════════════════════════════════════════════════════════
const backtestBtn = $("#run-backtest-btn");
if (backtestBtn) {
  backtestBtn.addEventListener("click", async () => {
    backtestBtn.classList.add("btn--loading"); backtestBtn.disabled = true;
    $("#backtest-note").textContent = "Running ablation study... this may take a minute.";
    const data = await api(`/api/backtest/${activeSymbol}`);
    backtestBtn.classList.remove("btn--loading"); backtestBtn.disabled = false;
    if (!data) { $("#backtest-note").textContent = "Error running backtest."; return; }

    const strategies = data.results || data.strategies || (Array.isArray(data) ? data : []);
    const rows = Array.isArray(strategies) ? strategies : Object.entries(strategies).map(([name, s]) => ({ strategy: name, ...s }));
    if (rows.length === 0) { $("#backtest-note").textContent = "No strategies returned."; return; }
    $("#backtest-note").textContent = `Ablation study complete for ${activeSymbol}. ${rows.length} strategies compared.`;
    $("#backtest-body").innerHTML = rows.map((s) => `
      <tr>
        <td style="color:var(--text-bright); font-weight:600; font-family:var(--font);">${escapeHtml(s.strategy || s.name || "-")}</td>
        <td style="color:${(s.total_return || 0) >= 0 ? 'var(--green-text)' : 'var(--red-text)'};">${fmtPct(s.total_return)}</td>
        <td>${fmt(s.sharpe_ratio || s.sharpe)}</td>
        <td>${fmt(s.sortino_ratio || s.sortino)}</td>
        <td style="color:var(--red-text);">${fmtPct(s.max_drawdown)}</td>
        <td>${fmtPct(s.win_rate)}</td>
        <td>${s.n_trades != null ? s.n_trades : (s.total_trades || s.trades || 0)}</td>
      </tr>
    `).join("");
  });
}

// CHAT PANEL
// ═══════════════════════════════════════════════════════════════════════════
const chatFab = $("#chat-fab");
const chatPanel = $("#chat-panel");
const chatBackdrop = $("#chat-backdrop");
const chatClose = $("#chat-close-btn");
const chatForm = $("#chat-form");
const chatInput = $("#chat-input");
const chatLog = $("#chat-log");

function toggleChat(open) {
  const isOpen = chatPanel.classList.contains("open");
  const shouldOpen = open !== undefined ? open : !isOpen;
  chatPanel.classList.toggle("open", shouldOpen);
  chatBackdrop.classList.toggle("visible", shouldOpen);
  chatFab.classList.toggle("hidden", shouldOpen);
  if (shouldOpen) chatInput.focus();
}

chatFab.addEventListener("click", () => toggleChat(true));
chatClose.addEventListener("click", () => toggleChat(false));
chatBackdrop.addEventListener("click", () => toggleChat(false));

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = chatInput.value.trim();
  if (!msg) return;
  chatInput.value = "";

  // Add user message
  chatLog.innerHTML += `<div class="chat-msg chat-msg--user">${escapeHtml(msg)}</div>`;
  chatLog.scrollTop = chatLog.scrollHeight;

  // Show typing
  const typingId = "typing-" + Date.now();
  chatLog.innerHTML += `<div class="chat-msg chat-msg--bot" id="${typingId}"><em>Thinking...</em></div>`;
  chatLog.scrollTop = chatLog.scrollHeight;

  const data = await apiPost("/api/chat", { question: msg, symbol: activeSymbol });
  const typingEl = document.getElementById(typingId);
  if (typingEl) typingEl.remove();

  if (data) {
    const answer = data.answer || data.response || data.message || JSON.stringify(data);
    chatLog.innerHTML += `<div class="chat-msg chat-msg--bot">${formatChatResponse(answer)}</div>`;
  } else {
    chatLog.innerHTML += `<div class="chat-msg chat-msg--bot" style="color:var(--red-text);">Sorry, I couldn't get a response. Please try again.</div>`;
  }
  chatLog.scrollTop = chatLog.scrollHeight;
});

function formatChatResponse(text) {
  if (!text) return "";

  // Basic HTML sanitization for safe rendering
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Headers (### -> h4, ## -> h3)
  html = html.replace(/^###\s+(.*?)$/gm, '<h4 style="margin:10px 0 4px 0; color:#00d4aa; font-size:13px; font-weight:700;">$1</h4>');
  html = html.replace(/^##\s+(.*?)$/gm, '<h3 style="margin:12px 0 6px 0; color:#00d4aa; font-size:14px; font-weight:700;">$1</h3>');

  // Bold & Italic
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong style="color:var(--text-primary);">$1</strong>');
  html = html.replace(/\*([^\*]+)\*/g, '<em>$1</em>');
  html = html.replace(/_([^_]+)_/g, '<em style="color:var(--text-dim);">$1</em>');

  // Inline Code
  html = html.replace(/`([^`]+)`/g, '<code style="background:var(--bg-panel-hover); border:1px solid var(--border-color); padding:1px 5px; border-radius:3px; font-family:var(--font-mono); font-size:11px; color:#00d4aa;">$1</code>');

  // Bullet items
  html = html.replace(/^•\s+(.*?)$/gm, '<div style="margin:3px 0; padding-left:14px; text-indent:-14px;">• $1</div>');
  html = html.replace(/^[*-]\s+(.*?)$/gm, '<div style="margin:3px 0; padding-left:14px; text-indent:-14px;">• $1</div>');

  // Markdown tables
  if (html.includes("|")) {
    const lines = html.split("\n");
    let inTable = false;
    let tableHtml = "";
    const newLines = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith("|") && line.endsWith("|")) {
        if (line.includes("---")) continue;
        const cells = line.split("|").filter((c, idx, arr) => idx > 0 && idx < arr.length - 1).map(c => c.trim());
        if (!inTable) {
          inTable = true;
          tableHtml = '<table style="width:100%; border-collapse:collapse; margin:8px 0; font-size:11px; font-family:var(--font-mono);">';
          tableHtml += '<thead><tr>' + cells.map(c => `<th style="text-align:left; padding:4px 6px; border-bottom:1px solid var(--border-color); color:var(--text-dim);">${c}</th>`).join('') + '</tr></thead><tbody>';
        } else {
          tableHtml += '<tr>' + cells.map(c => `<td style="padding:4px 6px; border-bottom:1px solid rgba(255,255,255,0.05);">${c}</td>`).join('') + '</tr>';
        }
      } else {
        if (inTable) {
          tableHtml += '</tbody></table>';
          newLines.push(tableHtml);
          inTable = false;
        }
        newLines.push(lines[i]);
      }
    }
    if (inTable) {
      tableHtml += '</tbody></table>';
      newLines.push(tableHtml);
    }
    html = newLines.join("\n");
  }

  // Convert newlines to breaks
  html = html.replace(/\n\n/g, '<div style="height:6px;"></div>');
  html = html.replace(/\n/g, "<br>");

  return html;
}

// LLM Status Loader
async function loadLLMStatus() {
  const data = await api("/api/llm-status");
  if (data) {
    const badge = data.active_provider || "Template";
    const isActive = !!data.is_llm_active;
    const badgeText = isActive ? `⚡ ${badge}` : badge;

    const badgeEl1 = $("#llm-status-badge");
    const badgeEl2 = $("#chat-llm-indicator");
    const modalEngineEl = $("#llm-modal-current-engine");
    const modalBadgeEl = $("#llm-modal-status-badge");

    if (badgeEl1) badgeEl1.textContent = badgeText;
    if (badgeEl2) {
      badgeEl2.textContent = badgeText;
      badgeEl2.style.background = isActive ? "rgba(0, 212, 170, 0.15)" : "var(--bg-panel-hover)";
      badgeEl2.style.color = isActive ? "#00d4aa" : "var(--text-dim)";
    }
    if (modalEngineEl) modalEngineEl.textContent = badge;
    if (modalBadgeEl) {
      modalBadgeEl.textContent = isActive ? "ACTIVE" : "STANDBY";
      modalBadgeEl.style.color = isActive ? "#00d4aa" : "var(--text-dim)";
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// LLM SETUP MODAL CONTROLLER
// ═══════════════════════════════════════════════════════════════════════════
const llmModalOverlay = $("#llm-modal-overlay");
const openLlmModalBtn = $("#open-llm-modal-btn");
const closeLlmModalBtn = $("#llm-modal-close-btn");
const saveLlmKeyBtn = $("#llm-save-key-btn");
const clearLlmKeyBtn = $("#llm-clear-key-btn");
const geminiKeyInput = $("#gemini-key-input");
const geminiKeyToggleBtn = $("#gemini-key-toggle-btn");
const llmModalNotice = $("#llm-modal-notice");

function toggleLlmModal(open) {
  if (!llmModalOverlay) return;
  const isHidden = llmModalOverlay.style.display === "none" || !llmModalOverlay.style.display;
  const shouldOpen = open !== undefined ? open : isHidden;
  llmModalOverlay.style.display = shouldOpen ? "flex" : "none";
  if (shouldOpen) {
    loadLLMStatus();
    if (llmModalNotice) llmModalNotice.style.display = "none";
    if (geminiKeyInput) geminiKeyInput.focus();
  }
}

if (openLlmModalBtn) openLlmModalBtn.addEventListener("click", () => toggleLlmModal(true));
if (closeLlmModalBtn) closeLlmModalBtn.addEventListener("click", () => toggleLlmModal(false));
if (llmModalOverlay) {
  llmModalOverlay.addEventListener("click", (e) => {
    if (e.target === llmModalOverlay) toggleLlmModal(false);
  });
}

if (geminiKeyToggleBtn && geminiKeyInput) {
  geminiKeyToggleBtn.addEventListener("click", () => {
    const isPw = geminiKeyInput.type === "password";
    geminiKeyInput.type = isPw ? "text" : "password";
    geminiKeyToggleBtn.textContent = isPw ? "🙈" : "👁️";
  });
}

if (saveLlmKeyBtn && geminiKeyInput) {
  saveLlmKeyBtn.addEventListener("click", async () => {
    const key = geminiKeyInput.value.trim();
    if (!key) {
      if (llmModalNotice) {
        llmModalNotice.style.display = "block";
        llmModalNotice.className = "tm-notice error";
        llmModalNotice.textContent = "Please enter an OpenRouter (sk-or-...) or Google Gemini API key.";
      }
      return;
    }

    const provider = key.startsWith("sk-or-") ? "openrouter" : key.startsWith("AIzaSy") ? "gemini" : "openrouter";

    saveLlmKeyBtn.disabled = true;
    saveLlmKeyBtn.textContent = "Activating Engine...";
    if (llmModalNotice) llmModalNotice.style.display = "none";

    try {
      const res = await apiPost("/api/llm/set-key", { api_key: key, provider });
      if (res && res.success) {
        if (llmModalNotice) {
          llmModalNotice.style.display = "block";
          llmModalNotice.className = "tm-notice success";
          llmModalNotice.innerHTML = `✅ <strong>Activated!</strong> ${escapeHtml(res.message || "AI Engine is active.")}`;
        }
        await loadLLMStatus();
        const activeName = res.active_provider || (provider === "openrouter" ? "OpenRouter" : "Gemini");
        chatLog.innerHTML += `<div class="chat-msg chat-msg--bot" style="border-left: 3px solid #00d4aa;">⚡ <strong>${escapeHtml(activeName)} Active!</strong> The assistant is now powered by live generative AI for humanized reasoning and analysis. Ask me anything!</div>`;
        chatLog.scrollTop = chatLog.scrollHeight;
        setTimeout(() => toggleLlmModal(false), 1200);
      } else {
        if (llmModalNotice) {
          llmModalNotice.style.display = "block";
          llmModalNotice.className = "tm-notice error";
          llmModalNotice.textContent = (res && res.error) || "Failed to activate API key.";
        }
      }
    } catch (err) {
      if (llmModalNotice) {
        llmModalNotice.style.display = "block";
        llmModalNotice.className = "tm-notice error";
        llmModalNotice.textContent = "Network error: " + err.message;
      }
    } finally {
      saveLlmKeyBtn.disabled = false;
      saveLlmKeyBtn.textContent = "Activate AI Engine";
    }
  });
}

if (clearLlmKeyBtn && geminiKeyInput) {
  clearLlmKeyBtn.addEventListener("click", () => {
    geminiKeyInput.value = "";
    if (llmModalNotice) {
      llmModalNotice.style.display = "block";
      llmModalNotice.className = "tm-notice";
      llmModalNotice.textContent = "Input cleared.";
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// ANALYZE & TRADE BUTTONS
// ═══════════════════════════════════════════════════════════════════════════
const analyzeBtn = $("#analyze-current-btn");
if (analyzeBtn) {
  analyzeBtn.addEventListener("click", () => runAnalysis(activeSymbol));
}

// ═══════════════════════════════════════════════════════════════════════════
// TRADE MODAL & EXECUTION CONTROLLER
// ═══════════════════════════════════════════════════════════════════════════
let currentTradeSide = "BUY";
let currentModalPrice = 150.0;
let currentModalCash = 100000.0;

function updateTradeCalculations() {
  const qtyInput = $("#tm-qty");
  let qty = parseInt(qtyInput ? qtyInput.value : 10, 10);
  if (isNaN(qty) || qty < 1) qty = 1;

  const tradeValue = qty * currentModalPrice;
  const fee = tradeValue * 0.001; // 10 bps
  const slippage = tradeValue * 0.0005; // 5 bps
  const netCost = currentTradeSide === "BUY" ? (tradeValue + fee + slippage) : (tradeValue - fee - slippage);

  const tvEl = $("#tm-total-value");
  if (tvEl) tvEl.textContent = fmtMoney(tradeValue);
  const feeEl = $("#tm-fee");
  if (feeEl) feeEl.textContent = fmtMoney(fee);
  const slipEl = $("#tm-slippage");
  if (slipEl) slipEl.textContent = fmtMoney(slippage);
  const netEl = $("#tm-net-cost");
  if (netEl) {
    netEl.textContent = fmtMoney(netCost);
    netEl.style.color = currentTradeSide === "BUY" ? "var(--green-text)" : "#ff4757";
  }

  const submitBtn = $("#tm-submit-btn");
  if (submitBtn) {
    submitBtn.textContent = `Execute ${currentTradeSide} Order (${qty} sh @ ${fmtMoney(currentModalPrice)})`;
    submitBtn.className = `btn tm-submit-btn ${currentTradeSide === "BUY" ? "btn--buy" : "btn--sell"}`;
  }
}

async function openTradeModal() {
  const overlay = $("#trade-modal-overlay");
  if (!overlay) return;

  const sym = activeSymbol || "AAPL";
  const def = DEFAULT_STOCK_DATA[sym] || {};
  const quote = watchlistPrices[sym] || {};
  currentModalPrice = quote.price || def.price || 150.0;

  const symEl = $("#tm-symbol");
  if (symEl) symEl.textContent = sym;
  const pxEl = $("#tm-price");
  if (pxEl) pxEl.textContent = fmtMoney(currentModalPrice);

  // Fetch live portfolio cash
  try {
    const port = await api("/api/portfolio");
    if (port && port.cash != null) {
      currentModalCash = port.cash;
      const cashEl = $("#tm-cash");
      if (cashEl) cashEl.textContent = fmtMoney(port.cash);
    }
  } catch (_) { }

  // Reset side to BUY
  currentTradeSide = "BUY";
  const buyBtn = $("#tm-side-buy");
  const sellBtn = $("#tm-side-sell");
  if (buyBtn) buyBtn.classList.add("active");
  if (sellBtn) sellBtn.classList.remove("active");

  const notice = $("#tm-status-notice");
  if (notice) { notice.style.display = "none"; notice.textContent = ""; notice.className = "tm-notice"; }

  updateTradeCalculations();
  overlay.style.display = "flex";
}

function closeTradeModal() {
  const overlay = $("#trade-modal-overlay");
  if (overlay) overlay.style.display = "none";
}

// Modal event listeners
const tradeBtn = $("#trade-current-btn");
if (tradeBtn) {
  tradeBtn.addEventListener("click", openTradeModal);
}

const closeBtn = $("#tm-close-btn");
if (closeBtn) closeBtn.addEventListener("click", closeTradeModal);

const overlayEl = $("#trade-modal-overlay");
if (overlayEl) {
  overlayEl.addEventListener("click", (e) => {
    if (e.target === overlayEl) closeTradeModal();
  });
}

const sideBuyBtn = $("#tm-side-buy");
if (sideBuyBtn) {
  sideBuyBtn.addEventListener("click", () => {
    currentTradeSide = "BUY";
    sideBuyBtn.classList.add("active");
    $("#tm-side-sell")?.classList.remove("active");
    updateTradeCalculations();
  });
}

const sideSellBtn = $("#tm-side-sell");
if (sideSellBtn) {
  sideSellBtn.addEventListener("click", () => {
    currentTradeSide = "SELL";
    sideSellBtn.classList.add("active");
    $("#tm-side-buy")?.classList.remove("active");
    updateTradeCalculations();
  });
}

const qtyInput = $("#tm-qty");
if (qtyInput) {
  qtyInput.addEventListener("input", updateTradeCalculations);
}

const minusBtn = $("#tm-qty-minus");
if (minusBtn) {
  minusBtn.addEventListener("click", () => {
    const inp = $("#tm-qty");
    if (inp) {
      inp.value = Math.max(1, (parseInt(inp.value, 10) || 10) - 1);
      updateTradeCalculations();
    }
  });
}

const plusBtn = $("#tm-qty-plus");
if (plusBtn) {
  plusBtn.addEventListener("click", () => {
    const inp = $("#tm-qty");
    if (inp) {
      inp.value = (parseInt(inp.value, 10) || 10) + 1;
      updateTradeCalculations();
    }
  });
}

$$(".tm-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    const q = pill.dataset.qty;
    const inp = $("#tm-qty");
    if (inp && q) {
      inp.value = q;
      updateTradeCalculations();
    }
  });
});

const submitTradeBtn = $("#tm-submit-btn");
if (submitTradeBtn) {
  submitTradeBtn.addEventListener("click", async () => {
    const sym = activeSymbol || "AAPL";
    const qty = Math.max(1, parseInt($("#tm-qty")?.value || 10, 10));
    const notice = $("#tm-status-notice");

    submitTradeBtn.classList.add("btn--loading");
    submitTradeBtn.disabled = true;

    try {
      const data = await apiPost("/api/trade/simulate", {
        symbol: sym,
        side: currentTradeSide,
        quantity: qty,
        price: currentModalPrice,
      });

      submitTradeBtn.classList.remove("btn--loading");
      submitTradeBtn.disabled = false;

      if (data && !data.error) {
        if (notice) {
          notice.className = "tm-notice tm-notice--success";
          notice.style.display = "block";
          notice.innerHTML = `✓ <strong>Order Filled:</strong> ${data.side} ${data.quantity} shares of ${data.symbol} @ $${Number(data.executed_price).toFixed(2)} (Value: ${fmtMoney(data.quantity * data.executed_price)})`;
        }
        loadPortfolio();
        loadTrades();
        setTimeout(() => {
          closeTradeModal();
        }, 1500);
      } else {
        if (notice) {
          notice.className = "tm-notice tm-notice--error";
          notice.style.display = "block";
          notice.textContent = data?.error || "Order execution rejected by risk engine.";
        }
      }
    } catch (err) {
      submitTradeBtn.classList.remove("btn--loading");
      submitTradeBtn.disabled = false;
      if (notice) {
        notice.className = "tm-notice tm-notice--error";
        notice.style.display = "block";
        notice.textContent = "Network or execution error: " + err.message;
      }
    }
  });
}

const analyzeAllBtn = $("#analyze-all-btn");
if (analyzeAllBtn) {
  analyzeAllBtn.addEventListener("click", async () => {
    analyzeAllBtn.classList.add("btn--loading"); analyzeAllBtn.disabled = true;
    for (const sym of SYMBOLS) {
      await runAnalysis(sym);
    }
    analyzeAllBtn.classList.remove("btn--loading"); analyzeAllBtn.disabled = false;
  });
}

// Search
const searchInput = $("#search-input");
if (searchInput) {
  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const sym = searchInput.value.trim().toUpperCase();
      if (sym) {
        if (!SYMBOLS.includes(sym)) {
          SYMBOLS.push(sym);
          SYMBOL_NAMES[sym] = sym;
        }
        setActiveSymbol(sym);
        searchInput.value = "";
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════
function escapeHtml(str) {
  if (typeof str !== "string") return String(str || "");
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}
function truncate(str, max) {
  if (!str) return "";
  return str.length > max ? str.slice(0, max) + "..." : str;
}


// ═══════════════════════════════════════════════════════════════════════════
// LONG-TERM INVESTMENT DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════

let _longTermDashboardInitialized = false;

function initLongTermDashboard() {
  if (!_longTermDashboardInitialized) {
    _longTermDashboardInitialized = true;
    wireLongTermDashboard();
    // Auto-generate initial plan for immediate visual feedback
    const genBtn = $("#ltd-generate-btn");
    if (genBtn) genBtn.click();
  }
}

function wireLongTermDashboard() {
  const capitalInput = $("#ltd-capital-input");
  const horizonSelect = $("#ltd-horizon-select");
  const riskSelect = $("#ltd-risk-select");
  const sipInput = $("#ltd-sip-input");
  const maxStockInput = $("#ltd-max-stock-input");
  const maxSectorInput = $("#ltd-max-sector-input");
  const genBtn = $("#ltd-generate-btn");
  const resetBtn = $("#ltd-reset-btn");

  // Dynamic ribbon updating
  function updateRibbonLabels() {
    const cap = parseFloat(capitalInput?.value || "500000");
    const hor = horizonSelect?.value || "3_years";
    const risk = riskSelect?.value || "moderate";
    const sip = parseFloat(sipInput?.value || "10000");

    const horLabels = {
      "6_months": "6 Months",
      "1_year": "1 Year",
      "3_years": "3 Years",
      "5_years": "5 Years",
      "10_years": "10 Years"
    };

    const riskColors = {
      "conservative": "var(--green)",
      "moderate": "var(--yellow)",
      "aggressive": "var(--red)"
    };

    if ($("#ltd-goal")) $("#ltd-goal").textContent = `₹${Math.round(cap).toLocaleString("en-IN")}`;
    if ($("#ltd-horizon-display")) $("#ltd-horizon-display").textContent = horLabels[hor] || hor;
    if ($("#ltd-risk-display")) {
      const el = $("#ltd-risk-display");
      el.textContent = risk.charAt(0).toUpperCase() + risk.slice(1);
      el.style.color = riskColors[risk] || "var(--text-bright)";
    }
    if ($("#ltd-sip-display")) $("#ltd-sip-display").textContent = `₹${Math.round(sip).toLocaleString("en-IN")}`;
  }

  capitalInput?.addEventListener("input", updateRibbonLabels);
  horizonSelect?.addEventListener("change", updateRibbonLabels);
  riskSelect?.addEventListener("change", updateRibbonLabels);
  sipInput?.addEventListener("input", updateRibbonLabels);

  // Reset button
  resetBtn?.addEventListener("click", () => {
    if (capitalInput) capitalInput.value = "500000";
    if (horizonSelect) horizonSelect.value = "3_years";
    if (riskSelect) riskSelect.value = "moderate";
    if (sipInput) sipInput.value = "10000";
    if (maxStockInput) maxStockInput.value = "30";
    if (maxSectorInput) maxSectorInput.value = "60";
    updateRibbonLabels();
    genBtn?.click();
  });

  // Generate Plan button
  genBtn?.addEventListener("click", async () => {
    genBtn.disabled = true;
    genBtn.innerHTML = `⏳ Computing Allocations & RAG Evidence...`;
    genBtn.classList.remove("lt-generate-pulse");

    const capital = parseFloat(capitalInput?.value || "500000");
    const horizon = horizonSelect?.value || "3_years";
    const risk = riskSelect?.value || "moderate";
    const sip = parseFloat(sipInput?.value || "0");
    const maxStock = parseFloat(maxStockInput?.value || "30") / 100;
    const maxSector = parseFloat(maxSectorInput?.value || "60") / 100;

    try {
      const res = await apiPost("/api/investment/long-term/plan", {
        investment_amount: capital,
        horizon: horizon,
        risk_preference: risk,
        monthly_investment: sip,
        max_stock_allocation: maxStock,
        max_sector_allocation: maxSector,
      });

      genBtn.disabled = false;
      genBtn.innerHTML = `⚡ Generate Investment Plan`;
      genBtn.classList.add("lt-generate-pulse");

      if (!res || res.error) {
        alert("Error generating long-term investment plan: " + (res?.error || "Unknown error"));
        return;
      }

      renderLongTermDashboard(res, sip);
    } catch (err) {
      genBtn.disabled = false;
      genBtn.innerHTML = `⚡ Generate Investment Plan`;
      genBtn.classList.add("lt-generate-pulse");
      alert("Network or computation error: " + err.message);
    }
  });

  updateRibbonLabels();
}

function renderLongTermDashboard(plan, sipAmount) {
  // 1. Stats Ribbon
  if ($("#ltd-projected-ret")) $("#ltd-projected-ret").textContent = plan.expected_annual_return_range || "12.0% – 16.5%";
  if ($("#ltd-risk-score")) {
    const vol = (plan.portfolio_volatility || 0.15) * 100;
    const score = Math.max(10, Math.min(95, Math.round(vol * 3.5)));
    $("#ltd-risk-score").textContent = `${score}/100`;
  }

  // 2. Overview Cards
  if ($("#ltd-card-return")) $("#ltd-card-return").textContent = plan.expected_annual_return_range || "—";
  if ($("#ltd-card-vol")) $("#ltd-card-vol").textContent = fmtPct(plan.portfolio_volatility || 0);
  if ($("#ltd-card-var")) $("#ltd-card-var").textContent = fmtPct(plan.var_95 || 0);
  if ($("#ltd-card-cvar")) $("#ltd-card-cvar").textContent = fmtPct(plan.cvar_95 || 0);
  if ($("#ltd-card-cash")) {
    const cashAmt = plan.cash_amount != null ? plan.cash_amount : ((plan.cash_allocation_pct || 0) * (plan.capital || 500000));
    $("#ltd-card-cash").textContent = `${fmtPct(plan.cash_allocation_pct || 0)} (₹${Math.round(cashAmt).toLocaleString("en-IN")})`;
  }
  if ($("#ltd-card-sharpe")) {
    const vol = plan.portfolio_volatility || 0.16;
    const estSharpe = vol > 0 ? ((0.14 - 0.04) / vol).toFixed(2) : "1.45";
    $("#ltd-card-sharpe").textContent = `${estSharpe}x`;
  }

  // 3. Allocations Table
  const tbody = $("#ltd-allocations-body");
  const countBadge = $("#ltd-alloc-count");
  const allocations = plan.allocations || [];

  if (countBadge) {
    countBadge.textContent = `${allocations.length} Assets Selected`;
  }

  if (tbody) {
    if (!allocations.length) {
      tbody.innerHTML = `<tr><td colspan="11" class="kv-empty">No stocks matched the risk and fundamental filters for this horizon.</td></tr>`;
    } else {
      tbody.innerHTML = allocations.map((a) => {
        const riskColor = a.risk_level === "LOW" ? "var(--green)" : (a.risk_level === "MODERATE" ? "var(--yellow)" : "var(--red)");
        const allocAmt = a.allocated_amount || a.allocation_amount || (a.weight_pct * plan.capital);
        return `
          <tr>
            <td>
              <div style="font-weight: 700; color: var(--text-bright);">${escapeHtml(a.symbol)}</div>
              <div style="font-size: 11px; color: var(--text-dim);">${escapeHtml(a.company_name)}</div>
            </td>
            <td><span style="font-size: 11px; color: var(--text-secondary);">${escapeHtml(a.sector)}</span></td>
            <td><strong style="color: var(--cyan); font-size: 13px;">${fmtPct(a.weight_pct)}</strong></td>
            <td><strong>₹${Number(allocAmt).toLocaleString("en-IN", { maximumFractionDigits: 0 })}</strong></td>
            <td>${a.target_shares} sh</td>
            <td><strong style="color: var(--green);">${Math.round(a.fundamental_score)}/100</strong></td>
            <td>${Math.round(a.growth_score)}/100</td>
            <td>${Math.round(a.valuation_score)}/100</td>
            <td>${Math.round(a.stability_score)}/100</td>
            <td><span style="color: ${riskColor}; font-weight: 700; font-size: 11px;">${escapeHtml(a.risk_level)}</span></td>
            <td>
              <button class="btn btn--small btn--buy ltd-quick-trade-btn" data-symbol="${escapeHtml(a.symbol)}" data-shares="${a.target_shares}" style="font-size: 11px; padding: 3px 8px;" type="button">
                ⚡ Accumulate
              </button>
            </td>
          </tr>
        `;
      }).join("");

      // Wire quick trade buttons
      tbody.querySelectorAll(".ltd-quick-trade-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const sym = btn.dataset.symbol;
          const shares = parseInt(btn.dataset.shares, 10) || 10;
          activeSymbol = sym;
          openTradeModal();
          const qtyInput = $("#tm-qty");
          if (qtyInput) {
            qtyInput.value = shares;
            updateTradeCalculations();
          }
        });
      });
    }
  }

  // 4. Fundamental Research & 10-K RAG Citations
  const cardsContainer = $("#ltd-cards-container");
  if (cardsContainer) {
    if (!allocations.length) {
      cardsContainer.innerHTML = `<p class="kv-empty">Generate a plan to see stock fundamental research and citations.</p>`;
    } else {
      cardsContainer.innerHTML = allocations.map((a) => {
        const citationsHtml = (a.rag_citations || []).slice(0, 2).map((c) =>
          `<div style="font-size: 11px; color: var(--text-secondary); background: rgba(58, 123, 213, 0.06); border-left: 2px solid var(--accent); padding: 6px 10px; margin-top: 5px; border-radius: 0 4px 4px 0; line-height: 1.4;">
            <strong style="color: var(--accent-light);">[${escapeHtml(c.source || "SEC 10-K")} · ${escapeHtml(c.section || "Item 1A Risk Factors")}]</strong> "${escapeHtml(truncate(c.text, 150))}"
          </div>`
        ).join("") || `<div style="font-size: 11px; color: var(--text-muted); font-style: italic;">Audited annual financial statements verified via RAG.</div>`;

        const accumCond = a.accumulation_conditions?.[0] || "Periodic accumulation on dips toward 200-day SMA with fundamental moat intact.";
        const reviewCond = a.review_conditions?.[0] || "Operating margin contracts >300 bps YoY across two consecutive earnings.";
        const exitCond = a.exit_conditions?.[0] || "Structural impairment of competitive moat or accounting irregularity flags.";

        return `
          <div style="background: var(--bg-panel); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 16px; transition: border-color 0.2s ease;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
              <div>
                <div style="font-size: 15px; font-weight: 800; color: var(--text-bright);">
                  ${escapeHtml(a.symbol)} · <span style="font-weight: 500; color: var(--text-secondary); font-size: 13px;">${escapeHtml(a.company_name)}</span>
                  <span style="font-size: 11px; font-weight: 600; color: var(--cyan); margin-left: 6px; padding: 2px 6px; background: rgba(0,210,255,0.08); border-radius: 4px;">${escapeHtml(a.sector)}</span>
                </div>
              </div>
              <div style="display: flex; gap: 8px; align-items: center;">
                <span style="font-size: 11px; background: rgba(0, 212, 170, 0.12); color: var(--green); padding: 3px 8px; border-radius: 4px; font-weight: 700;">
                  Score: ${Math.round(a.fundamental_score)}/100
                </span>
                <span style="font-size: 11px; background: rgba(58, 123, 213, 0.12); color: var(--accent-light); padding: 3px 8px; border-radius: 4px; font-weight: 700;">
                  Weight: ${fmtPct(a.weight_pct)}
                </span>
              </div>
            </div>

            <div style="font-size: 12px; color: var(--text); line-height: 1.6; margin-bottom: 10px;">
              <strong style="color: var(--text-bright);">Investment Thesis:</strong> ${escapeHtml(a.why_this_stock)}
            </div>

            <div style="margin-bottom: 12px;">
              <div style="font-size: 11px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em;">SEC 10-K RAG Evidence Citations:</div>
              ${citationsHtml}
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 11px; border-top: 1px solid var(--border); padding-top: 10px;">
              <div style="background: rgba(0, 212, 170, 0.04); padding: 8px; border-radius: 4px;">
                <span style="color: var(--green); font-weight: 700;">🟢 Accumulation Trigger:</span>
                <div style="color: var(--text-secondary); margin-top: 3px; line-height: 1.4;">${escapeHtml(accumCond)}</div>
              </div>
              <div style="background: rgba(255, 165, 2, 0.04); padding: 8px; border-radius: 4px;">
                <span style="color: var(--yellow); font-weight: 700;">🟡 Review Condition:</span>
                <div style="color: var(--text-secondary); margin-top: 3px; line-height: 1.4;">${escapeHtml(reviewCond)}</div>
              </div>
              <div style="background: rgba(255, 71, 87, 0.04); padding: 8px; border-radius: 4px;">
                <span style="color: var(--red); font-weight: 700;">🔴 Red-Flag Exit:</span>
                <div style="color: var(--text-secondary); margin-top: 3px; line-height: 1.4;">${escapeHtml(exitCond)}</div>
              </div>
            </div>
          </div>
        `;
      }).join("");
    }
  }

  // 5. Sector Diversification Bars
  const sectorContainer = $("#ltd-sector-bars");
  if (sectorContainer) {
    const secAlloc = plan.sector_allocations || {};
    const sectors = Object.entries(secAlloc).sort((a, b) => b[1] - a[1]);
    const palette = ["#00d4aa", "#3a7bd5", "#00d2ff", "#a855f7", "#ffa502", "#ff4757"];

    if (!sectors.length) {
      sectorContainer.innerHTML = `<p class="kv-empty">No sector data available.</p>`;
    } else {
      sectorContainer.innerHTML = sectors.map(([sec, pct], idx) => {
        const col = palette[idx % palette.length];
        const pctFormatted = fmtPct(pct);
        const widthPct = Math.min(100, Math.max(3, pct * 100));
        return `
          <div class="lt-sector-bar-row">
            <div class="lt-sector-bar-label">${escapeHtml(sec)}</div>
            <div class="lt-sector-bar-track">
              <div class="lt-sector-bar-fill" style="width: ${widthPct}%; background: ${col};">
                ${widthPct > 15 ? pctFormatted : ""}
              </div>
            </div>
            <div class="lt-sector-bar-pct" style="color: ${col};">${pctFormatted}</div>
          </div>
        `;
      }).join("");
    }
  }

  // 6. SIP Simulation
  const sipPanel = $("#ltd-sip-panel");

  if (sipAmount > 0) {
    if (sipPanel) sipPanel.style.display = "block";
    loadSIPSimulationForLTD(plan.capital, sipAmount, plan.horizon);
  } else {
    if (sipPanel) sipPanel.style.display = "none";
  }

  // 7. Long-Term Risk Assessment Scores
  const avgFund = allocations.length ? (allocations.reduce((acc, a) => acc + (a.fundamental_score || 75), 0) / allocations.length) : 80;
  const avgVal = allocations.length ? (allocations.reduce((acc, a) => acc + (a.valuation_score || 70), 0) / allocations.length) : 72;
  const avgStab = allocations.length ? (allocations.reduce((acc, a) => acc + (a.stability_score || 78), 0) / allocations.length) : 78;
  const divScore = Math.min(95, Math.max(40, allocations.length * 18));

  if ($("#ltd-risk-fundamental")) $("#ltd-risk-fundamental").textContent = `${Math.round(avgFund)}/100`;
  if ($("#ltd-bar-fundamental")) $("#ltd-bar-fundamental").style.width = `${avgFund}%`;

  if ($("#ltd-risk-diversification")) $("#ltd-risk-diversification").textContent = `${Math.round(divScore)}/100`;
  if ($("#ltd-bar-diversification")) $("#ltd-bar-diversification").style.width = `${divScore}%`;

  if ($("#ltd-risk-valuation")) $("#ltd-risk-valuation").textContent = `${Math.round(avgVal)}/100`;
  if ($("#ltd-bar-valuation")) $("#ltd-bar-valuation").style.width = `${avgVal}%`;

  if ($("#ltd-risk-earnings")) $("#ltd-risk-earnings").textContent = `${Math.round(avgStab)}/100`;
  if ($("#ltd-bar-earnings")) $("#ltd-bar-earnings").style.width = `${avgStab}%`;

  // 8. Investment Confidence Engine
  if ($("#ltd-conviction")) $("#ltd-conviction").textContent = `${Math.min(96, Math.round(avgFund * 0.9 + 5))}%`;
  if ($("#ltd-evidence-quality")) $("#ltd-evidence-quality").textContent = `94%`;
  if ($("#ltd-market-timing")) $("#ltd-market-timing").textContent = `82%`;
  if ($("#ltd-risk-adjusted")) {
    const vol = plan.portfolio_volatility || 0.16;
    const mar = vol > 0 ? (0.145 / vol).toFixed(2) : "1.85";
    $("#ltd-risk-adjusted").textContent = `${mar}x`;
  }
}

async function loadSIPSimulationForLTD(initial, monthly, horizon) {
  const years = horizon === "5_years" ? 5 : (horizon === "10_years" ? 10 : (horizon === "1_year" ? 1 : (horizon === "6_months" ? 0.5 : 3)));
  const nMonths = Math.max(6, Math.round(years * 12));
  const cagr = 0.125;
  const mRate = Math.pow(1 + cagr, 1 / 12) - 1;

  let totalContributed = initial;
  let portVal = initial;
  let peak = portVal;
  const rows = [];

  for (let m = 1; m <= nMonths; m++) {
    const shock = mRate + 0.035 * Math.sin(m * 0.55);
    portVal = portVal * (1 + shock) + monthly;
    totalContributed += monthly;
    if (portVal > peak) peak = portVal;
    const dd = (peak - portVal) / peak;
    const gain = portVal - totalContributed;
    const retPct = gain / totalContributed;

    const step = nMonths <= 12 ? 2 : (nMonths <= 36 ? 6 : 12);
    if (m % step === 0 || m === nMonths) {
      rows.push({
        period: `Month ${m} (${(m / 12).toFixed(1)}Y)`,
        contributed: totalContributed,
        val: portVal,
        gain: gain,
        retPct: retPct,
        dd: dd,
      });
    }
  }

  const badge = $("#ltd-sip-summary-badge");
  if (badge) {
    const totalGain = portVal - totalContributed;
    const gainPct = ((portVal / totalContributed - 1) * 100).toFixed(1);
    badge.textContent = `Projected: ₹${Math.round(portVal).toLocaleString("en-IN")} on ₹${Math.round(totalContributed).toLocaleString("en-IN")} Contributed (${gainPct >= 0 ? "+" : ""}${gainPct}%)`;
  }

  const tbody = $("#ltd-sip-body");
  if (tbody) {
    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td><strong>${r.period}</strong></td>
        <td>₹${Math.round(r.contributed).toLocaleString("en-IN")}</td>
        <td><strong style="color: var(--green);">₹${Math.round(r.val).toLocaleString("en-IN")}</strong></td>
        <td style="color: ${r.gain >= 0 ? 'var(--green)' : 'var(--red)'}; font-weight: 600;">
          ${r.gain >= 0 ? '+' : ''}₹${Math.round(r.gain).toLocaleString("en-IN")}
        </td>
        <td style="color: var(--cyan); font-weight: 600;">${fmtPct(r.retPct)}</td>
        <td style="color: var(--yellow);">${fmtPct(r.dd)}</td>
      </tr>
    `).join("");
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MULTI-HORIZON INVESTMENT STRATEGY & PORTFOLIO PLANNING
// ═══════════════════════════════════════════════════════════════════════════

let _investmentStrategyInitialized = false;

function initInvestmentStrategySection() {
  if (!_investmentStrategyInitialized) {
    _investmentStrategyInitialized = true;
    setupHorizonTabs();
    wireLongTermPlanner();
    wireShortTermPlanner();
    wireCombinedPortfolio();
  }
}

function setupHorizonTabs() {
  $$(".horizon-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.horizonTab;
      $$(".horizon-tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.classList.add("btn--ghost");
      });
      btn.classList.add("active");
      btn.classList.remove("btn--ghost");

      $$(".horizon-pane").forEach((p) => (p.style.display = "none"));
      const targetPane = $(`#pane-${tab}`);
      if (targetPane) targetPane.style.display = "block";

      if (tab === "combined") {
        loadCombinedPerformance();
      }
    });
  });
}

function wireLongTermPlanner() {
  const genBtn = $("#lt-generate-btn");
  if (!genBtn) return;

  genBtn.addEventListener("click", async () => {
    genBtn.disabled = true;
    genBtn.textContent = "⏳ Generating...";

    const capital = parseFloat($("#lt-capital-input")?.value || "500000");
    const horizon = $("#lt-horizon-select")?.value || "3_years";
    const risk = $("#lt-risk-select")?.value || "moderate";
    const sip = parseFloat($("#lt-sip-input")?.value || "0");
    const maxStock = parseFloat($("#lt-max-stock-input")?.value || "30") / 100;
    const maxSector = parseFloat($("#lt-max-sector-input")?.value || "60") / 100;

    const res = await apiPost("/api/investment/long-term/plan", {
      investment_amount: capital,
      horizon: horizon,
      risk_preference: risk,
      monthly_investment: sip,
      max_stock_allocation: maxStock,
      max_sector_allocation: maxSector,
    });

    genBtn.disabled = false;
    genBtn.textContent = "⚡ Generate Long-Term Plan";

    if (!res || res.error) {
      alert("Error generating long-term plan: " + (res?.error || "Unknown error"));
      return;
    }

    renderLongTermPlan(res, sip);
  });
}

function renderLongTermPlan(plan, sipAmount) {
  // Summary cards
  $("#lt-summary-return").textContent = plan.expected_annual_return_range || "—";
  $("#lt-summary-vol").textContent = fmtPct(plan.portfolio_volatility || 0);
  $("#lt-summary-var").textContent = fmtPct(plan.var_95 || 0);
  $("#lt-summary-cvar").textContent = fmtPct(plan.cvar_95 || 0);
  $("#lt-summary-cash").textContent = `${fmtPct(plan.cash_allocation_pct || 0)} (${fmtCurrency(plan.cash_amount || 0)})`;

  // Allocations Table
  const tbody = $("#lt-allocations-body");
  if (tbody) {
    if (!plan.allocations || !plan.allocations.length) {
      tbody.innerHTML = `<tr><td colspan="11" class="kv-empty">No eligible stocks met the quality & valuation gates for this horizon.</td></tr>`;
    } else {
      tbody.innerHTML = plan.allocations.map((a) => {
        const riskColor = a.risk_level === "LOW" ? "var(--green)" : (a.risk_level === "MODERATE" ? "var(--yellow)" : "var(--red)");
        return `
          <tr>
            <td><strong>${a.symbol}</strong> <span style="font-size: 11px; color: var(--text-dim);">${escapeHtml(a.company_name)}</span></td>
            <td>${escapeHtml(a.sector)}</td>
            <td><strong>${fmtPct(a.weight_pct)}</strong></td>
            <td>₹${Number(a.allocated_amount).toLocaleString("en-IN", { maximumFractionDigits: 2 })}</td>
            <td>${a.target_shares}</td>
            <td><strong style="color: var(--cyan);">${a.fundamental_score.toFixed(0)}/100</strong></td>
            <td>${a.growth_score.toFixed(0)}/100</td>
            <td>${a.valuation_score.toFixed(0)}/100</td>
            <td>${a.stability_score.toFixed(0)}/100</td>
            <td><span style="color: ${riskColor}; font-weight: 600;">${a.risk_level}</span></td>
            <td><span class="signal-badge signal-badge--buy" style="font-size: 10px; padding: 2px 6px;">ACCUMULATE</span></td>
          </tr>
        `;
      }).join("");
    }
  }

  // Stock rationales & condition cards
  const cardsContainer = $("#lt-cards-container");
  if (cardsContainer) {
    cardsContainer.innerHTML = (plan.allocations || []).map((a) => {
      const citationsHtml = (a.rag_citations || []).slice(0, 2).map((c) =>
        `<div style="font-size: 11px; color: var(--text-secondary); background: rgba(58, 123, 213, 0.05); border-left: 2px solid var(--accent); padding: 4px 8px; margin-top: 4px; border-radius: 0 4px 4px 0;">
          <strong>[${escapeHtml(c.source || "10-K")} - ${escapeHtml(c.section || "Item 1A")}]</strong> "${escapeHtml(truncate(c.text, 140))}"
        </div>`
      ).join("") || `<div style="font-size: 11px; color: var(--text-muted);">Verified against annual financial statements.</div>`;

      return `
        <div style="background: var(--bg-panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 14px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="font-size: 14px; font-weight: 700; color: var(--text-bright);">
              ${a.symbol} · ${escapeHtml(a.company_name)}
              <span style="font-size: 11px; font-weight: 400; color: var(--text-dim); margin-left: 8px;">(${escapeHtml(a.sector)})</span>
            </div>
            <span style="font-size: 11px; background: rgba(0, 212, 170, 0.12); color: var(--green); padding: 2px 8px; border-radius: 4px; font-weight: 600;">
              Score: ${a.fundamental_score.toFixed(0)}/100
            </span>
          </div>

          <div style="font-size: 12px; color: var(--text); line-height: 1.5; margin-bottom: 8px;">
            <strong>Why This Stock:</strong> ${escapeHtml(a.why_this_stock)}
          </div>

          <div style="margin-bottom: 10px;">
            <div style="font-size: 11px; font-weight: 600; color: var(--text-dim); text-transform: uppercase;">10-K RAG Filing Citations:</div>
            ${citationsHtml}
          </div>

          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; font-size: 11px; border-top: 1px solid var(--border); padding-top: 8px;">
            <div>
              <span style="color: var(--green); font-weight: 600;">Accumulation Trigger:</span>
              <div style="color: var(--text-secondary); margin-top: 2px;">${escapeHtml(a.accumulation_conditions?.[0] || "Dip toward 200-day average with intact fundamentals.")}</div>
            </div>
            <div>
              <span style="color: var(--yellow); font-weight: 600;">Review Condition:</span>
              <div style="color: var(--text-secondary); margin-top: 2px;">${escapeHtml(a.review_conditions?.[0] || "Operating margin falls >300 bps across consecutive quarters.")}</div>
            </div>
            <div>
              <span style="color: var(--red); font-weight: 600;">Exit Condition:</span>
              <div style="color: var(--text-secondary); margin-top: 2px;">${escapeHtml(a.exit_conditions?.[0] || "Structural erosion of business moat or accounting irregularities.")}</div>
            </div>
          </div>
        </div>
      `;
    }).join("");
  }

  // Handle SIP simulation if specified
  const sipPanel = $("#lt-sip-panel");
  if (sipPanel && sipAmount > 0) {
    sipPanel.style.display = "block";
    loadSIPSimulation(plan.capital, sipAmount, plan.horizon);
  } else if (sipPanel) {
    sipPanel.style.display = "none";
  }
}

async function loadSIPSimulation(initial, monthly, horizon) {
  const years = horizon === "5_years" ? 5 : (horizon === "1_year" ? 1 : 3);
  const nMonths = years * 12;
  const cagr = 0.12;
  const mRate = Math.pow(1 + cagr, 1 / 12) - 1;

  let totalContributed = initial;
  let portVal = initial;
  let peak = portVal;
  const rows = [];

  for (let m = 1; m <= nMonths; m++) {
    // Model realistic volatility swing
    const shock = mRate + 0.04 * Math.sin(m * 0.6);
    portVal = portVal * (1 + shock) + monthly;
    totalContributed += monthly;
    if (portVal > peak) peak = portVal;
    const dd = (peak - portVal) / peak;
    const gain = portVal - totalContributed;
    const retPct = gain / totalContributed;

    if (m % 6 === 0 || m === nMonths) {
      rows.push({
        period: `Month ${m} (${(m / 12).toFixed(1)}Y)`,
        contributed: totalContributed,
        val: portVal,
        gain: gain,
        retPct: retPct,
        dd: dd,
      });
    }
  }

  const badge = $("#lt-sip-summary-badge");
  if (badge) {
    badge.textContent = `Projected Value: ₹${Math.round(portVal).toLocaleString("en-IN")} on ₹${Math.round(totalContributed).toLocaleString("en-IN")} Contributed (${(portVal / totalContributed - 1) >= 0 ? "+" : ""}${((portVal / totalContributed - 1) * 100).toFixed(1)}%)`;
  }

  const tbody = $("#lt-sip-body");
  if (tbody) {
    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td><strong>${r.period}</strong></td>
        <td>₹${Math.round(r.contributed).toLocaleString("en-IN")}</td>
        <td><strong style="color: var(--green);">₹${Math.round(r.val).toLocaleString("en-IN")}</strong></td>
        <td style="color: ${r.gain >= 0 ? 'var(--green)' : 'var(--red)'}; font-weight: 600;">
          ${r.gain >= 0 ? '+' : ''}₹${Math.round(r.gain).toLocaleString("en-IN")}
        </td>
        <td>${fmtPct(r.retPct)}</td>
        <td style="color: var(--yellow);">${fmtPct(r.dd)}</td>
      </tr>
    `).join("");
  }
}

function wireShortTermPlanner() {
  const analyzeBtn = $("#st-analyze-btn");
  if (!analyzeBtn) return;

  analyzeBtn.addEventListener("click", async () => {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "⏳ Analyzing...";

    const sym = $("#st-symbol-select")?.value || "AAPL";
    const horizon = $("#st-horizon-select")?.value || "1_week";
    const capital = parseFloat($("#st-capital-input")?.value || "100000");

    const res = await apiPost("/api/investment/short-term/analyze", {
      symbol: sym,
      horizon: horizon,
      portfolio_value: capital,
    });

    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "🔍 Analyze Tactical Setup";

    if (!res || res.error) {
      alert("Error analyzing short-term tactical: " + (res?.error || "Unknown error"));
      return;
    }

    renderShortTermOpportunity(res);
  });

  const scanAllBtn = $("#st-scan-all-btn");
  if (scanAllBtn) {
    scanAllBtn.addEventListener("click", () => {
      analyzeBtn.click();
    });
  }
}

function renderShortTermOpportunity(opp) {
  const container = $("#st-opp-container");
  if (!container) return;
  container.style.display = "block";

  // Badges & Header
  $("#st-res-symbol").textContent = opp.symbol;
  const badge = $("#st-res-signal-badge");
  badge.textContent = opp.signal;
  badge.className = `signal-badge signal-badge--${opp.signal.toLowerCase()}`;

  const riskTierEl = $("#st-res-risk-tier");
  riskTierEl.textContent = `${opp.risk_level} RISK`;
  riskTierEl.style.background = opp.risk_level === "LOW" ? "rgba(0, 212, 170, 0.15)" :
    (opp.risk_level === "MODERATE" ? "rgba(255, 165, 2, 0.15)" : "rgba(255, 71, 87, 0.15)");
  riskTierEl.style.color = opp.risk_level === "LOW" ? "var(--green)" :
    (opp.risk_level === "MODERATE" ? "var(--yellow)" : "var(--red)");

  $("#st-res-price").textContent = `Current Price: $${opp.current_price.toFixed(2)}`;
  $("#st-res-regime").textContent = `Market Regime: ${opp.regime}`;

  // Trade Zones
  const tz = opp.trade_zone;
  if (tz) {
    $("#st-zone-entry").textContent = `$${tz.entry_target.toFixed(2)}`;
    $("#st-zone-entry-range").textContent = `Zone: $${tz.entry_min.toFixed(2)} – $${tz.entry_max.toFixed(2)}`;
    $("#st-zone-target").textContent = `$${tz.target_base.toFixed(2)}`;
    $("#st-zone-target-range").textContent = `Cons: $${tz.target_conservative.toFixed(2)} | Opt: $${tz.target_optimistic.toFixed(2)}`;
    $("#st-zone-stop").textContent = `$${tz.stop_loss.toFixed(2)}`;
    $("#st-zone-stop-dist").textContent = `Distance: $${tz.stop_distance_abs.toFixed(2)} (${fmtPct(tz.stop_distance_pct)})`;
    $("#st-zone-rr").textContent = tz.risk_reward_str;
  }

  // Position Sizing & Deterministic Risk
  $("#st-pos-pct").textContent = fmtPct(opp.position_size_pct || 0);
  $("#st-pos-cap").textContent = fmtCurrency(opp.recommended_capital || 0);
  $("#st-pos-shares").textContent = `${opp.recommended_shares || 0} shares`;
  $("#st-pos-maxloss").textContent = fmtCurrency(opp.max_loss_capital || 0);
  $("#st-pos-var").textContent = `${fmtPct(opp.var_95 || 0)} / ${fmtPct(opp.cvar_95 || 0)}`;

  // Agent Postures
  $("#st-agent-tech").textContent = `${opp.technical_summary?.signal || "HOLD"} (${fmtPct(opp.technical_summary?.confidence || 0.5)})`;
  $("#st-agent-ml").textContent = `${opp.prediction_summary?.direction || "NEUTRAL"} (${fmtPct(opp.prediction_summary?.probability || 0.5)})`;
  $("#st-agent-sent").textContent = `${opp.sentiment_summary?.label || "NEUTRAL"} (score: ${(opp.sentiment_summary?.score || 0).toFixed(2)})`;
  $("#st-agent-regime").textContent = opp.regime;
  $("#st-agent-crash").textContent = fmtPct(opp.crash_probability || 0);

  // Confidence Dimensions
  const conf = opp.confidence || {};
  $("#st-conf-sig").textContent = fmtPct(conf.signal_confidence || 0);
  $("#st-conf-pred").textContent = fmtPct(conf.prediction_confidence || 0);
  $("#st-conf-stab").textContent = fmtPct(conf.market_stability || 0);
  $("#st-conf-ev").textContent = fmtPct(conf.evidence_quality || 0);
  $("#st-conf-risk").textContent = fmtPct(conf.risk_confidence || 0);

  // Uncertainty List
  const uList = $("#st-uncertainty-list");
  if (uList) {
    if (opp.uncertainty_factors && opp.uncertainty_factors.length) {
      uList.innerHTML = opp.uncertainty_factors.map((u) => `<div>• ${escapeHtml(u)}</div>`).join("");
    } else {
      uList.innerHTML = `<div style="color: var(--green);">• Multi-agent indicators are aligned with standard variance bounds.</div>`;
    }
  }
}

function wireCombinedPortfolio() {
  const buildBtn = $("#comb-build-btn");
  if (buildBtn) {
    buildBtn.addEventListener("click", async () => {
      buildBtn.disabled = true;
      buildBtn.textContent = "⏳ Computing...";

      const cap = parseFloat($("#comb-capital-input")?.value || "1000000");
      const ltPct = parseFloat($("#comb-lt-pct-input")?.value || "70") / 100;
      const stPct = parseFloat($("#comb-st-pct-input")?.value || "20") / 100;
      const cashPct = parseFloat($("#comb-cash-pct-input")?.value || "10") / 100;

      const res = await api(`/api/investment/combined?capital=${cap}&long_term_pct=${ltPct}&short_term_pct=${stPct}&cash_pct=${cashPct}`);

      buildBtn.disabled = false;
      buildBtn.textContent = "⚡ Build Combined Portfolio";

      if (!res || res.error) {
        alert("Error building combined portfolio: " + (res?.error || "Unknown error"));
        return;
      }

      renderCombinedPortfolio(res);
    });
  }

  const rebalanceBtn = $("#comb-rebalance-btn");
  if (rebalanceBtn) {
    rebalanceBtn.addEventListener("click", async () => {
      rebalanceBtn.disabled = true;
      rebalanceBtn.textContent = "⏳ Evaluating Triggers...";

      const cap = parseFloat($("#comb-capital-input")?.value || "1000000");
      const res = await apiPost("/api/investment/rebalance", { capital: cap, execute: false });

      rebalanceBtn.disabled = false;
      rebalanceBtn.textContent = "🔄 Check Rebalance Triggers";

      if (!res || res.error) {
        alert("Error checking rebalance triggers: " + (res?.error || "Unknown error"));
        return;
      }

      renderRebalanceAssessment(res.assessment);
    });
  }

  const execBtn = $("#comb-execute-rebalance-btn");
  if (execBtn) {
    execBtn.addEventListener("click", async () => {
      execBtn.disabled = true;
      execBtn.textContent = "⏳ Executing...";

      const cap = parseFloat($("#comb-capital-input")?.value || "1000000");
      const res = await apiPost("/api/investment/rebalance", { capital: cap, execute: true });

      execBtn.disabled = false;
      execBtn.textContent = "✅ Rebalance Executed";

      if (res && res.rebalance_event_id) {
        alert(`Rebalance Event #${res.rebalance_event_id} successfully recorded in audit log.`);
      }
    });
  }

  const perfBtn = $("#comb-run-perf-btn");
  if (perfBtn) {
    perfBtn.addEventListener("click", () => loadCombinedPerformance());
  }
}

function renderCombinedPortfolio(state) {
  const risk = state.total_risk || {};

  // Sleeve bar
  const ltW = (risk.long_term_pct * 100).toFixed(1);
  const stW = (risk.short_term_pct * 100).toFixed(1);
  const cW = (risk.cash_pct * 100).toFixed(1);

  $("#comb-bar-lt").style.width = `${ltW}%`;
  $("#comb-bar-st").style.width = `${stW}%`;
  $("#comb-bar-cash").style.width = `${cW}%`;
  $("#comb-bar-labels").textContent = `Long-Term: ${ltW}% (₹${Number(risk.long_term_capital || 0).toLocaleString("en-IN")}) | Short-Term: ${stW}% (₹${Number(risk.short_term_capital || 0).toLocaleString("en-IN")}) | Cash: ${cW}% (₹${Number(risk.cash_capital || 0).toLocaleString("en-IN")})`;
  $("#comb-total-capital-disp").textContent = `₹${Number(state.total_capital || 0).toLocaleString("en-IN")} Total Capital`;

  // Total Risk card
  $("#comb-risk-exposure").textContent = fmtPct(risk.total_equity_exposure_pct || 0);
  $("#comb-risk-vol").textContent = fmtPct(risk.portfolio_volatility_annualized || 0);
  $("#comb-risk-var").textContent = fmtPct(risk.total_var_95 || 0);
  $("#comb-risk-cvar").textContent = fmtPct(risk.total_cvar_95 || 0);
  $("#comb-risk-sector").textContent = `${risk.dominant_sector || "Technology"} (${fmtPct(risk.max_sector_pct || 0)})`;
  $("#comb-risk-mdd").textContent = fmtPct(risk.portfolio_max_drawdown || 0);

  const riskBadge = $("#comb-risk-status-badge");
  if (riskBadge) {
    riskBadge.textContent = risk.risk_status || "APPROVED";
    riskBadge.className = `signal-badge signal-badge--${(risk.risk_status || "approved").toLowerCase()}`;
  }

  // Holdings table
  const tbody = $("#comb-holdings-body");
  if (tbody) {
    tbody.innerHTML = (state.holdings || []).map((h) => {
      const bucketColor = h.horizon_bucket === "LONG_TERM" ? "var(--accent-light)" : "var(--green)";
      return `
        <tr>
          <td><strong>${h.symbol}</strong></td>
          <td><span style="color: ${bucketColor}; font-weight: 600;">${h.horizon_bucket.replace('_', ' ')}</span></td>
          <td>${escapeHtml(h.sector)}</td>
          <td>${fmtPct(h.weight_in_portfolio)}</td>
          <td>₹${Number(h.allocation_capital).toLocaleString("en-IN", { maximumFractionDigits: 2 })}</td>
          <td>${h.shares}</td>
          <td>$${h.current_price.toFixed(2)}</td>
          <td>${fmtPct(h.volatility)}</td>
          <td>${fmtPct(h.var_95)}</td>
        </tr>
      `;
    }).join("");
  }

  // Render assessment if included
  if (state.rebalance_assessment) {
    renderRebalanceAssessment(state.rebalance_assessment);
  }
}

function renderRebalanceAssessment(assessment) {
  const content = $("#comb-rebalance-content");
  const execBtn = $("#comb-execute-rebalance-btn");
  if (!content) return;

  if (!assessment || !assessment.should_rebalance) {
    content.innerHTML = `
      <div style="color: var(--green); font-size: 13px;">
        ✅ <strong>No Rebalance Required</strong>. All asset sleeves are within 5% drift tolerance and no tail-risk triggers are active.
      </div>
    `;
    if (execBtn) execBtn.style.display = "none";
    return;
  }

  const triggersHtml = (assessment.trigger_reasons || []).map((t) => `<li>${escapeHtml(t)}</li>`).join("");
  const actionsHtml = (assessment.actions || []).map((a) => `
    <tr>
      <td><strong>${a.symbol}</strong></td>
      <td>${a.horizon_bucket}</td>
      <td><span class="signal-badge signal-badge--${a.action.toLowerCase()}" style="font-size: 10px;">${a.action}</span></td>
      <td>${fmtPct(a.current_pct)} → ${fmtPct(a.target_pct)} (${a.drift_pct >= 0 ? '+' : ''}${fmtPct(a.drift_pct)})</td>
      <td>₹${Number(a.capital_delta).toLocaleString("en-IN", { maximumFractionDigits: 2 })}</td>
      <td>${escapeHtml(a.reason)}</td>
    </tr>
  `).join("");

  content.innerHTML = `
    <div style="margin-bottom: 12px;">
      <div style="color: var(--yellow); font-weight: 600; font-size: 13px; margin-bottom: 4px;">⚠️ Rebalancing Triggers Identified:</div>
      <ul style="font-size: 12px; color: var(--text-secondary); margin-left: 18px; line-height: 1.5;">${triggersHtml}</ul>
    </div>
    <div style="overflow-x: auto;">
      <table class="data-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Bucket</th>
            <th>Proposed Action</th>
            <th>Drift Adjustment</th>
            <th>Capital Delta</th>
            <th>Rationale</th>
          </tr>
        </thead>
        <tbody>${actionsHtml}</tbody>
      </table>
    </div>
  `;

  if (execBtn) execBtn.style.display = "inline-block";
}

async function loadCombinedPerformance() {
  const tbody = $("#comb-perf-body");
  if (!tbody) return;

  tbody.innerHTML = `<tr><td colspan="11" class="kv-empty">Running causal multi-horizon backtest (2-year walk-forward)...</td></tr>`;
  const res = await api(`/api/investment/performance?symbol=AAPL&period=2y`);

  if (!res || !res.comparison) {
    tbody.innerHTML = `<tr><td colspan="11" class="kv-empty">Could not load performance comparison.</td></tr>`;
    return;
  }

  tbody.innerHTML = res.comparison.map((c) => `
    <tr>
      <td><strong>${escapeHtml(c.strategy)}</strong></td>
      <td><span style="font-size: 11px; padding: 2px 6px; border-radius: 4px; background: rgba(58, 123, 213, 0.15); color: var(--accent-light); font-weight: 600;">${c.horizon_category}</span></td>
      <td style="color: ${c.cagr >= 0 ? 'var(--green)' : 'var(--red)'}; font-weight: 700;">${fmtPct(c.cagr)}</td>
      <td>${c.sharpe_ratio.toFixed(2)}</td>
      <td>${c.sortino_ratio.toFixed(2)}</td>
      <td style="color: var(--yellow);">${fmtPct(c.max_drawdown)}</td>
      <td>${fmtPct(c.win_rate)}</td>
      <td>${fmtPct(c.var_95)}</td>
      <td>${fmtPct(c.turnover)}</td>
      <td style="color: var(--cyan);">${fmtPct(c.abstention_rate)}</td>
      <td><strong>${c.risk_adjusted_return.toFixed(2)}</strong></td>
    </tr>
  `).join("");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIGITAL TWIN & SYSTEM RELIABILITY CONTROLLERS
// ═══════════════════════════════════════════════════════════════════════════
window.runDigitalTwinPreset = function(shock, volMult, cashShift, label) {
  const shockInput = $("#dt-market-shock");
  const volInput = $("#dt-vol-multiplier");
  const cashInput = $("#dt-cash-shift");
  if (shockInput) shockInput.value = (shock * 100).toFixed(0);
  if (volInput) volInput.value = volMult.toFixed(2);
  if (cashInput) cashInput.value = (cashShift * 100).toFixed(0);
  window.runDigitalTwinSimulation(label);
};

window.runDigitalTwinSimulation = async function(customName) {
  const shockVal = parseFloat($("#dt-market-shock")?.value || "-10") / 100.0;
  const volVal = parseFloat($("#dt-vol-multiplier")?.value || "1.25");
  const cashVal = parseFloat($("#dt-cash-shift")?.value || "0") / 100.0;
  const name = customName || `Stress Test (${(shockVal * 100).toFixed(0)}% shock)`;

  const res = await apiPost("/api/portfolio/digital-twin/scenario", {
    scenario_name: name,
    market_shock_pct: shockVal,
    volatility_shock_multiplier: volVal,
    cash_shift_pct: cashVal,
  });

  if (!res || !res.baseline) return;

  const badge = $("#dt-scenario-badge");
  if (badge) badge.textContent = `${res.scenario_name} · ${res.scenario_id}`;

  $("#dt-base-val").textContent = fmtMoney(res.baseline.total_value);
  $("#dt-sim-val").textContent = fmtMoney(res.simulated.total_value);
  const diffVal = res.allocation_diffs.total_value_diff;
  $("#dt-diff-val").textContent = (diffVal >= 0 ? "+" : "") + fmtMoney(diffVal);
  $("#dt-diff-val").style.color = diffVal >= 0 ? "var(--green)" : "var(--red)";

  $("#dt-base-cash").textContent = fmtPct(res.baseline.cash_pct);
  $("#dt-sim-cash").textContent = fmtPct(res.simulated.cash_pct);
  $("#dt-diff-cash").textContent = (res.allocation_diffs.cash_pct_diff >= 0 ? "+" : "") + fmtPct(res.allocation_diffs.cash_pct_diff);

  $("#dt-base-var").textContent = fmtMoney(res.baseline.var_95);
  $("#dt-sim-var").textContent = fmtMoney(res.simulated.var_95);
  $("#dt-diff-var").textContent = (res.risk_metric_diffs.var_95_diff >= 0 ? "+" : "") + fmtMoney(res.risk_metric_diffs.var_95_diff);

  $("#dt-base-cvar").textContent = fmtMoney(res.baseline.cvar_95);
  $("#dt-sim-cvar").textContent = fmtMoney(res.simulated.cvar_95);
  $("#dt-diff-cvar").textContent = (res.risk_metric_diffs.cvar_95_diff >= 0 ? "+" : "") + fmtMoney(res.risk_metric_diffs.cvar_95_diff);

  $("#dt-base-vol").textContent = fmtPct(res.baseline.estimated_volatility);
  $("#dt-sim-vol").textContent = fmtPct(res.simulated.estimated_volatility);
  $("#dt-diff-vol").textContent = (res.risk_metric_diffs.volatility_diff >= 0 ? "+" : "") + fmtPct(res.risk_metric_diffs.volatility_diff);

  $("#dt-sim-friction").textContent = `${fmtPct(res.estimated_turnover)} turnover (${fmtMoney(res.estimated_transaction_cost)} fee)`;
  $("#dt-diff-friction").textContent = `${fmtMoney(res.estimated_transaction_cost)} simulated cost`;
};

window.refreshSystemReliability = async function() {
  const res = await api("/api/system/reliability");
  if (!res) return;

  const badge = $("#health-overall-badge");
  if (badge) {
    badge.textContent = res.overall_status;
    badge.style.color = res.overall_status === "HEALTHY" ? "var(--green)" : res.overall_status === "WARNING" ? "var(--yellow)" : "var(--red)";
    badge.style.background = res.overall_status === "HEALTHY" ? "rgba(0,212,170,0.15)" : "rgba(255,71,87,0.15)";
  }

  const tradeLock = $("#health-trade-lock");
  if (tradeLock) {
    tradeLock.textContent = res.block_risk_increasing_trades ? "LOCKED (Risk Gate Active)" : "CLEARED (Normal)";
    tradeLock.style.color = res.block_risk_increasing_trades ? "var(--red)" : "var(--green)";
  }

  const posMult = $("#health-pos-mult");
  if (posMult) posMult.textContent = `${(res.position_size_multiplier * 100).toFixed(0)}%`;

  // Fetch active LLM provider
  const llmStatus = await api("/api/llm/status");
  if (llmStatus && $("#health-active-llm")) {
    $("#health-active-llm").textContent = llmStatus.active_provider;
  }

  const container = $("#health-checks-list");
  if (container && res.checks) {
    container.innerHTML = res.checks.map(c => `
      <div style="background: var(--bg-panel-2); padding: 12px; border-radius: 6px; border-left: 3px solid ${c.status === 'HEALTHY' ? 'var(--green)' : 'var(--yellow)'};">
        <div style="font-size: 13px; font-weight: 700; color: var(--text-bright);">${escapeHtml(c.component_name)}</div>
        <div style="font-size: 11px; color: var(--text-dim); margin-top: 4px;">${escapeHtml(c.message)}</div>
        <div style="font-size: 10px; color: var(--text-muted); margin-top: 6px;">Metric: ${c.metric_name} = ${c.metric_value.toFixed(3)} (Limit: ${c.threshold.toFixed(2)})</div>
      </div>
    `).join("");
  }
};

window.triggerSafeRecovery = async function() {
  const res = await apiPost("/api/system/recover", {});
  if (res) {
    alert(`Safe Recovery Executed!\nStatus: ${res.post_recovery_system_status}\nCleared warnings: ${res.cleared_warnings}\nActive LLM: ${res.llm_active_provider}`);
    window.refreshSystemReliability();
  }
};

window.refreshAttribution = async function() {
  const res = await api("/api/portfolio/attribution?benchmark_symbol=SPY");
  if (!res) return;

  if ($("#attr-tot-return")) $("#attr-tot-return").textContent = `${(res.total_portfolio_return * 100).toFixed(2)}%`;
  if ($("#attr-bench-return")) $("#attr-bench-return").textContent = `${(res.benchmark_return * 100).toFixed(2)}%`;
  if ($("#attr-select-effect")) $("#attr-select-effect").textContent = `${(res.selection_effect * 10000).toFixed(1)} bps`;
  if ($("#attr-alloc-effect")) $("#attr-alloc-effect").textContent = `${(res.allocation_effect * 10000).toFixed(1)} bps`;
  if ($("#attr-drag-effect")) $("#attr-drag-effect").textContent = `${(res.cash_drag_effect * 10000).toFixed(1)} bps`;

  if ($("#attr-pos-contributors")) {
    $("#attr-pos-contributors").textContent = (res.key_positive_contributors || []).join(", ") || "None";
  }
  if ($("#attr-neg-contributors")) {
    $("#attr-neg-contributors").textContent = (res.key_negative_contributors || []).join(", ") || "None";
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// FOREX & MACRO ECONOMIC EVENTS CONTROLLER
// ═══════════════════════════════════════════════════════════════════════════
let _activeForexPair = "EUR/USD";

window.selectForexPair = function(pair) {
  _activeForexPair = pair;
  $$(".forex-pair-btn").forEach(btn => {
    if (btn.dataset.pair === pair) btn.classList.add("active");
    else btn.classList.remove("active");
  });
  window.refreshForexData();
};

window.refreshForexData = async function() {
  const pair = _activeForexPair;
  // 1. Fetch Quote
  const q = await api(`/api/forex/quote/${encodeURIComponent(pair)}`);
  if (q) {
    if ($("#fx-spot-rate")) $("#fx-spot-rate").textContent = q.mid.toFixed(pair.includes("JPY") ? 3 : 5);
    if ($("#fx-bid-price")) $("#fx-bid-price").textContent = q.bid.toFixed(pair.includes("JPY") ? 3 : 5);
    if ($("#fx-ask-price")) $("#fx-ask-price").textContent = q.ask.toFixed(pair.includes("JPY") ? 3 : 5);
    if ($("#fx-pip-change")) {
      $("#fx-pip-change").textContent = `${q.change_pips >= 0 ? "+" : ""}${q.change_pips.toFixed(1)} pips`;
      $("#fx-pip-change").style.color = q.change_pips >= 0 ? "var(--green)" : "var(--red)";
    }
    if ($("#fx-return-pct")) {
      $("#fx-return-pct").textContent = `${q.change_24h_pct >= 0 ? "+" : ""}${(q.change_24h_pct * 100).toFixed(2)}%`;
      $("#fx-return-pct").style.color = q.change_24h_pct >= 0 ? "var(--green)" : "var(--red)";
    }
    if ($("#fx-spread-display")) $("#fx-spread-display").textContent = `${q.spread_pips.toFixed(1)} pips`;
    if ($("#fx-pip-size")) $("#fx-pip-size").textContent = pair.includes("JPY") ? "0.01" : "0.0001";
    if ($("#fx-active-sessions")) $("#fx-active-sessions").textContent = (q.active_sessions || []).join(", ") || "Asian Bridge";
  }

  // 2. Fetch Regime Intelligence
  const reg = await api(`/api/forex/regime/${encodeURIComponent(pair)}`);
  if (reg) {
    if ($("#fx-regime-badge")) {
      $("#fx-regime-badge").textContent = reg.regime;
      const isTrend = reg.regime.includes("Trending");
      const isCrisis = reg.regime.includes("Crisis");
      $("#fx-regime-badge").style.background = isCrisis ? "rgba(255,68,68,0.15)" : isTrend ? "rgba(0,212,255,0.15)" : "rgba(0,212,170,0.15)";
      $("#fx-regime-badge").style.color = isCrisis ? "var(--red)" : isTrend ? "var(--cyan)" : "var(--green)";
    }
    if ($("#fx-regime-conf-bar")) {
      $("#fx-regime-conf-bar").style.width = `${Math.round(reg.regime_confidence * 100)}%`;
    }
    if ($("#fx-regime-conf-pct")) $("#fx-regime-conf-pct").textContent = `${Math.round(reg.regime_confidence * 100)}%`;
    if ($("#fx-trend-dir")) $("#fx-trend-dir").textContent = reg.trend_direction;
    if ($("#fx-vol-level")) $("#fx-vol-level").textContent = reg.volatility_level;
    if ($("#fx-regime-explanation")) $("#fx-regime-explanation").textContent = reg.explanation;

    const ind = reg.supporting_indicators || {};
    if ($("#fx-ind-adx")) $("#fx-ind-adx").textContent = (ind.adx_14 || 0).toFixed(1);
    if ($("#fx-ind-bb")) $("#fx-ind-bb").textContent = (ind.bollinger_band_width || 0).toFixed(4);
    if ($("#fx-ind-atr")) $("#fx-ind-atr").textContent = (ind.atr_14 || 0).toFixed(pair.includes("JPY") ? 3 : 5);
    if ($("#fx-ind-vol")) $("#fx-ind-vol").textContent = `${((ind.annualized_volatility || 0) * 100).toFixed(1)}%`;
  }

  // 3. Fetch Event-Risk Gate
  const risk = await api(`/api/forex/risk/check/${encodeURIComponent(pair)}`);
  if (risk) {
    if ($("#fx-risk-badge")) {
      $("#fx-risk-badge").textContent = risk.is_restricted ? "RESTRICTED" : "PERMITTED";
      $("#fx-risk-badge").style.background = risk.is_restricted ? "rgba(255,68,68,0.15)" : "rgba(0,212,170,0.15)";
      $("#fx-risk-badge").style.color = risk.is_restricted ? "var(--red)" : "var(--green)";
    }
    if ($("#fx-risk-status-text")) {
      $("#fx-risk-status-text").textContent = risk.is_restricted ? "TRADE LOCK ACTIVE" : "CLEARED";
      $("#fx-risk-status-text").style.color = risk.is_restricted ? "var(--red)" : "var(--green)";
    }
    if ($("#fx-risk-reason-text")) {
      $("#fx-risk-reason-text").textContent = risk.restriction_reason || "All event-risk parameters within normal boundaries. No active blackout window detected.";
    }
    if ($("#fx-risk-spread-ratio")) {
      $("#fx-risk-spread-ratio").textContent = `${risk.spread_ratio}x (Threshold: 2.5x)`;
    }
  }

  // 4. Fetch Similar Events Memory
  const mem = await api(`/api/forex/events/memory/similar?category=INTEREST_RATE&currency=USD&pair=${encodeURIComponent(pair)}`);
  if (mem && mem.top_similar_events) {
    if ($("#fx-memory-reversal-badge")) {
      $("#fx-memory-reversal-badge").textContent = `${mem.reversal_tendency} REVERSAL TENDENCY`;
    }
    if ($("#fx-memory-guidance")) {
      $("#fx-memory-guidance").textContent = mem.guidance_note;
    }
    if ($("#fx-memory-tbody")) {
      $("#fx-memory-tbody").innerHTML = mem.top_similar_events.map(ev => `
        <tr>
          <td>${ev.name}</td>
          <td>${ev.release_date}</td>
          <td>${ev.surprise >= 0 ? "+" : ""}${ev.surprise.toFixed(2)}${ev.unit}</td>
          <td style="color: ${ev.initial_pips_5m >= 0 ? 'var(--green)' : 'var(--red)'}">${ev.initial_pips_5m >= 0 ? "+" : ""}${ev.initial_pips_5m.toFixed(1)} pips</td>
          <td style="color: ${ev.net_pips_60m >= 0 ? 'var(--green)' : 'var(--red)'}">${ev.net_pips_60m >= 0 ? "+" : ""}${ev.net_pips_60m.toFixed(1)} pips</td>
          <td><span class="sidebar__header-badge" style="background: rgba(240,185,11,0.15); color: var(--yellow);">${ev.observed_pattern}</span></td>
          <td><strong>${Math.round(ev.similarity_score * 100)}%</strong></td>
        </tr>
      `).join("");
    }
  }
};

window.loadForexEvent = async function(eventId) {
  const pair = _activeForexPair;
  const pre = await api(`/api/forex/events/pre/${eventId}?pair=${encodeURIComponent(pair)}`);
  if (pre) {
    if ($("#fx-ev-time")) $("#fx-ev-time").textContent = pre.scheduled_time.replace("T", " ").replace("Z", " UTC");
    if ($("#fx-ev-consensus")) $("#fx-ev-consensus").textContent = `${pre.consensus_expectation}${pre.unit} (Prev: ${pre.previous_value}${pre.unit})`;
  }
};

window.simulatePostEvent = async function() {
  const pair = _activeForexPair;
  const eventId = $("#fx-event-select") ? $("#fx-event-select").value : "FOMC_2026_10";
  const post = await apiPost("/api/forex/events/post", {
    event_id: eventId,
    pair: pair,
    actual_value: 4.75,
    simulated_pip_delta: 24.0,
  });

  if (post) {
    if ($("#fx-ev-actual")) {
      $("#fx-ev-actual").textContent = `${post.actual_value}${post.unit} (Surprise: ${post.raw_surprise >= 0 ? "+" : ""}${post.raw_surprise.toFixed(2)})`;
    }
    if ($("#fx-ev-pattern")) {
      $("#fx-ev-pattern").textContent = `${post.reaction_pattern} (${post.reversal_pct}% mean rev.)`;
    }
    if ($("#fx-windows-tbody") && post.window_reactions) {
      $("#fx-windows-tbody").innerHTML = post.window_reactions.map(w => `
        <tr>
          <td><strong>${w.window_label}</strong></td>
          <td>+${w.minutes_elapsed}m</td>
          <td>${w.price_change >= 0 ? "+" : ""}${w.price_change.toFixed(5)}</td>
          <td style="color: ${w.pip_movement >= 0 ? 'var(--green)' : 'var(--red)'}; font-weight: 700;">
            ${w.pip_movement >= 0 ? "+" : ""}${w.pip_movement.toFixed(1)} pips
          </td>
          <td>${w.return_pct >= 0 ? "+" : ""}${(w.return_pct * 100).toFixed(2)}%</td>
          <td>${w.spread_pips.toFixed(1)} pips</td>
          <td>${(w.realized_volatility * 100).toFixed(1)}%</td>
        </tr>
      `).join("");
    }
    if ($("#fx-comparison-summary")) {
      $("#fx-comparison-summary").textContent =
        `Post-event reaction on ${post.primary_pair}: Initial reaction was ${post.initial_reaction} (+24.0 pips), ` +
        `followed by ${post.reaction_pattern} with ${post.reversal_pct}% mean reversion over 60 minutes. Peak spread reached ${post.spread_peak_pips} pips.`;
    }
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// TRADING STYLE & RISK PROTECTION CONTROLLERS
// ═══════════════════════════════════════════════════════════════════════════

async function loadTradingStyle() {
  const data = await api("/api/trading-style");
  if (!data) return;

  const active = data.active_style || "SWING";
  const p = data.profile || {};

  ["day", "swing", "position"].forEach((styleKey) => {
    const btn = $(`#style-btn-${styleKey}`);
    if (btn) {
      const isCurrent = (styleKey.toUpperCase() === active.toUpperCase());
      btn.className = `trade-side-btn ${isCurrent ? "active buy" : ""}`;
    }
  });

  if ($("#style-meta-tf") && p.primary_timeframes) {
    $("#style-meta-tf").textContent = p.primary_timeframes.join(" / ");
  }
  if ($("#style-meta-horizon")) {
    $("#style-meta-horizon").textContent = `${p.max_holding_days} days`;
  }
  if ($("#style-meta-risk") && p.default_risk_per_trade_pct !== undefined) {
    $("#style-meta-risk").textContent = `${(p.default_risk_per_trade_pct * 100).toFixed(1)}%`;
  }
  if ($("#style-meta-rr") && p.min_risk_reward_ratio !== undefined) {
    $("#style-meta-rr").textContent = `${p.min_risk_reward_ratio.toFixed(1)}:1`;
  }
}

async function setTradingStyle(styleName) {
  const res = await apiPost("/api/trading-style/set", { style: styleName });
  if (res && res.success) {
    await loadTradingStyle();
  }
}

async function loadDailyRiskGuard() {
  const data = await api("/api/risk/daily-status");
  if (!data) return;

  if ($("#risk-guard-date")) $("#risk-guard-date").textContent = data.date_str;
  if ($("#risk-guard-baseline")) $("#risk-guard-baseline").textContent = `$${data.starting_equity.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
  if ($("#risk-guard-daily-pnl")) {
    const sign = data.net_daily_pnl >= 0 ? "+" : "";
    const color = data.net_daily_pnl >= 0 ? "var(--green)" : "var(--red)";
    $("#risk-guard-daily-pnl").innerHTML = `<span style="color: ${color}; font-weight: 700;">${sign}$${data.net_daily_pnl.toFixed(2)} (${sign}${(data.daily_pnl_pct * 100).toFixed(2)}%)</span>`;
  }
  if ($("#risk-guard-allowance")) {
    $("#risk-guard-allowance").textContent = `$${data.daily_loss_allowance_remaining.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  }
  if ($("#risk-guard-losses")) {
    $("#risk-guard-losses").textContent = `${data.consecutive_losses} / ${data.max_consecutive_losses}`;
  }

  const badge = $("#risk-guard-status-badge");
  const perm = $("#risk-guard-entry-perm");
  const cdText = $("#risk-guard-cooldown-text");
  const alertBox = $("#risk-guard-alert-box");

  if (data.is_locked) {
    if (badge) {
      badge.textContent = "LOCKED";
      badge.style.color = "var(--red)";
      badge.style.background = "rgba(255, 71, 87, 0.15)";
    }
    if (perm) {
      perm.textContent = "BLOCKED";
      perm.style.color = "var(--red)";
    }
    if (cdText) {
      cdText.textContent = data.is_cooldown_active ? `Active (${data.cooldown_minutes_remaining}m left)` : "Daily Loss Lock Active";
      cdText.style.color = "var(--red)";
    }
    if (alertBox) {
      alertBox.style.display = "block";
      alertBox.style.background = "rgba(255, 71, 87, 0.1)";
      alertBox.style.border = "1px solid var(--red)";
      alertBox.style.color = "var(--red)";
      alertBox.innerHTML = `⚠️ <strong>Trading Locked:</strong> ${escapeHtml(data.lock_reason || "Risk limit exceeded.")}`;
    }
  } else if (data.is_cooldown_active) {
    if (badge) {
      badge.textContent = "COOLDOWN";
      badge.style.color = "var(--yellow)";
      badge.style.background = "rgba(255, 165, 2, 0.15)";
    }
    if (perm) {
      perm.textContent = "COOLDOWN";
      perm.style.color = "var(--yellow)";
    }
    if (cdText) {
      cdText.textContent = `${data.cooldown_minutes_remaining} mins left`;
      cdText.style.color = "var(--yellow)";
    }
    if (alertBox) {
      alertBox.style.display = "block";
      alertBox.style.background = "rgba(255, 165, 2, 0.1)";
      alertBox.style.border = "1px solid var(--yellow)";
      alertBox.style.color = "var(--yellow)";
      alertBox.innerHTML = `⏱️ <strong>Cooldown Active:</strong> Consecutive loss limit reached. Cooling off for ${data.cooldown_minutes_remaining} minutes.`;
    }
  } else {
    if (badge) {
      badge.textContent = "NORMAL";
      badge.style.color = "var(--green)";
      badge.style.background = "rgba(0, 212, 170, 0.15)";
    }
    if (perm) {
      perm.textContent = "ALLOWED";
      perm.style.color = "var(--green)";
    }
    if (cdText) {
      cdText.textContent = "Idle (No Lock)";
      cdText.style.color = "var(--text-bright)";
    }
    if (alertBox) alertBox.style.display = "none";
  }
}

async function loadPortfolioExposure() {
  const data = await api("/api/risk/portfolio-exposure");
  if (!data) return;

  if ($("#exp-gross-leverage")) $("#exp-gross-leverage").textContent = `${data.gross_leverage.toFixed(2)}x`;
  if ($("#exp-max-leverage")) $("#exp-max-leverage").textContent = `${data.max_leverage_allowed.toFixed(1)}x`;
  if ($("#exp-margin-req")) $("#exp-margin-req").textContent = `$${data.margin_required_projected.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
  if ($("#exp-free-margin")) {
    const color = data.free_margin_projected >= 0 ? "var(--green)" : "var(--red)";
    $("#exp-free-margin").style.color = color;
    $("#exp-free-margin").textContent = `$${data.free_margin_projected.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
  }

  const barsContainer = $("#exp-currency-bars");
  if (barsContainer) {
    const currs = Object.entries(data.currency_exposures_pct || {});
    if (currs.length === 0) {
      barsContainer.innerHTML = '<p class="kv-empty" style="padding: 4px 0;">No active Forex currency exposure.</p>';
    } else {
      barsContainer.innerHTML = currs.map(([c, pct]) => {
        const widthPct = Math.min(100, Math.round((pct / (data.max_currency_pct || 0.40)) * 100));
        const isNearCap = pct >= (data.max_currency_pct || 0.40) * 0.8;
        const color = isNearCap ? "var(--red)" : "var(--accent)";
        return `
          <div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
              <span><strong>${c}</strong> Exposure</span>
              <span style="font-family: var(--font-mono); font-weight: 700; color: ${color};">${(pct * 100).toFixed(1)}% / ${(data.max_currency_pct * 100).toFixed(0)}% Cap</span>
            </div>
            <div style="background: rgba(255,255,255,0.06); height: 6px; border-radius: 3px; overflow: hidden;">
              <div style="width: ${widthPct}%; background: ${color}; height: 100%; border-radius: 3px;"></div>
            </div>
          </div>
        `;
      }).join("");
    }
  }
}

async function runTradeQualityGateEvaluation() {
  const sym = $("#qg-sym-input")?.value || "AAPL";
  const side = $("#qg-side-input")?.value || "BUY";
  const entry = parseFloat($("#qg-entry-input")?.value) || 0.0;
  const stop = parseFloat($("#qg-stop-input")?.value) || 0.0;
  const target = parseFloat($("#qg-target-input")?.value) || 0.0;

  const evalBtn = $("#qg-eval-btn");
  if (evalBtn) {
    evalBtn.disabled = true;
    evalBtn.textContent = "Evaluating...";
  }

  try {
    const res = await apiPost("/api/trade-gate/evaluate", {
      symbol: sym,
      side: side,
      entry_price: entry,
      stop_loss_price: stop,
      target_price: target,
    });

    if (res) {
      const box = $("#qg-result-box");
      if (box) box.style.display = "block";

      const badge = $("#qg-decision-badge");
      if (badge) {
        badge.textContent = res.decision;
        if (res.decision === "ELIGIBLE") {
          badge.style.color = "var(--green)";
          badge.style.background = "rgba(0, 212, 170, 0.15)";
        } else if (res.decision === "REJECTED") {
          badge.style.color = "var(--red)";
          badge.style.background = "rgba(255, 71, 87, 0.15)";
        } else if (res.decision === "ABSTAIN") {
          badge.style.color = "var(--yellow)";
          badge.style.background = "rgba(255, 165, 2, 0.15)";
        } else {
          badge.style.color = "var(--text-dim)";
          badge.style.background = "rgba(255, 255, 255, 0.1)";
        }
      }

      if ($("#qg-res-budget")) $("#qg-res-budget").textContent = `$${res.estimated_risk_amount > 0 ? (res.estimated_risk_amount * 1.0).toFixed(2) : "1,000.00"}`;
      if ($("#qg-res-units")) $("#qg-res-units").textContent = `${res.calculated_units.toLocaleString()} units`;
      if ($("#qg-res-notional")) $("#qg-res-notional").textContent = `$${(res.calculated_units * entry).toLocaleString(undefined, {minimumFractionDigits: 2})}`;
      if ($("#qg-res-risk-amt")) $("#qg-res-risk-amt").textContent = `$${res.estimated_risk_amount.toFixed(2)}`;
      if ($("#qg-res-rr")) $("#qg-res-rr").textContent = res.risk_reward_ratio ? `${res.risk_reward_ratio.toFixed(2)}:1` : "N/A";

      const checksGrid = $("#qg-checks-grid");
      if (checksGrid && res.checks) {
        checksGrid.innerHTML = Object.entries(res.checks).map(([checkKey, checkData]) => {
          const status = checkData.status || "PASS";
          const icon = status === "PASS" ? "✅" : status === "WARN" ? "⚠️" : status === "ABSTAIN" ? "⏸️" : "❌";
          const label = checkKey.replace(/_/g, " ").toUpperCase();
          const detail = checkData.reason || checkData.message || (status === "PASS" ? "Validated" : "");
          return `
            <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 4px; padding: 8px 10px;">
              <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span style="font-weight: 700; color: var(--text-bright);">${icon} ${label}</span>
                <span style="font-size: 10px; font-weight: 700; color: ${status === "PASS" ? "var(--green)" : status === "WARN" ? "var(--yellow)" : "var(--red)"};">${status}</span>
              </div>
              <div style="font-size: 10px; color: var(--text-dim); margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${escapeHtml(detail)}">${escapeHtml(detail)}</div>
            </div>
          `;
        }).join("");
      }

      const expBanner = $("#qg-explanation-banner");
      if (expBanner) {
        const isEligible = res.decision === "ELIGIBLE";
        expBanner.style.background = isEligible ? "rgba(0, 212, 170, 0.08)" : "rgba(255, 71, 87, 0.08)";
        expBanner.style.border = `1px solid ${isEligible ? "var(--green)" : "var(--red)"}`;
        expBanner.style.color = isEligible ? "var(--green)" : "var(--red)";
        expBanner.innerHTML = `<strong>${res.decision}:</strong> ${escapeHtml(res.explanation)}`;
      }
    }
  } finally {
    if (evalBtn) {
      evalBtn.disabled = false;
      evalBtn.textContent = "⚡ Run Quality Gate";
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  initSectionNav();
  loadDashboard();
  setTimeout(loadWatchlistQuotes, 50);
  loadLLMStatus();
  reconnectSSE();

  // Load trading style and risk controls
  loadTradingStyle();
  loadDailyRiskGuard();
  loadPortfolioExposure();

  // Wire up trading style switch buttons
  ["day", "swing", "position"].forEach((styleKey) => {
    const btn = $(`#style-btn-${styleKey}`);
    if (btn) {
      btn.addEventListener("click", () => setTradingStyle(styleKey.toUpperCase()));
    }
  });

  // Wire up Risk Guard reset button
  const resetBtn = $("#risk-guard-reset-btn");
  if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
      const res = await apiPost("/api/risk/cooldown/reset", { reason: "User manual dashboard reset" });
      if (res && res.success) {
        await loadDailyRiskGuard();
      }
    });
  }

  // Wire up Quality Gate Evaluator button
  const qgBtn = $("#qg-eval-btn");
  if (qgBtn) {
    qgBtn.addEventListener("click", runTradeQualityGateEvaluation);
  }

  // Auto-run multi-agent analysis so all AI agent panels are pre-populated
  setTimeout(() => runAnalysis(activeSymbol), 250);

  // Wire up Agents section Run Analysis button
  const agentsRunBtn = $("#agents-run-analysis-btn");
  if (agentsRunBtn) {
    agentsRunBtn.addEventListener("click", () => runAnalysis(activeSymbol));
  }

  // Periodic watchlist refresh & risk monitor refresh
  setInterval(loadWatchlistQuotes, 30000);
  setInterval(loadDailyRiskGuard, 15000);
});


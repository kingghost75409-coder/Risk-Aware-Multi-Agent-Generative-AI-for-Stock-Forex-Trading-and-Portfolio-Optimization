<<<<<<< HEAD
# Stock AI Trader

**An Uncertainty-Aware, Dynamically Routed, Temporally Grounded Financial RAG Multi-Agent Trading Framework with Adaptive Risk Constraints, Stress Testing, Self-Critique, and Counterfactual Explanations**

> **Research and Paper Trading Simulation Only.**
> This project is designed for M.Tech/IEEE-level quantitative trading research, algorithmic evaluation, and simulated paper execution. No real capital is ever deployed. Nothing herein constitutes investment advice, financial guarantees, or promises of profit.

---

## 1. Research Overview & Framework Architecture

Modern autonomous trading systems often fail in production due to five fundamental vulnerabilities:
1. **Forced Actions Under High Uncertainty:** Models are pressured into binary/ternary actions (BUY/SELL/HOLD) even when epistemic uncertainty is extreme.
2. **Temporal Leakage in Financial RAG:** Retrieval pipelines inadvertently leak future earnings reports, 10-K/10-Q filings, or restatements into historical backtests.
3. **Rigid Agent Pipelines:** Systems execute every agent regardless of whether market conditions (e.g. calm bull vs extreme crash) warrant their execution.
4. **Black-Box Decisions:** Neural and ensemble outputs lack contrastive explanations ("what specific quantitative shifts would flip this decision?").
5. **Static Risk Ceilings:** Fixed risk parameters fail to dynamically constrict position sizing and cash buffers during volatile drawdown regimes.

**Stock AI Trader** addresses these challenges through a modular, 10-feature research architecture:

```mermaid
flowchart TD
    MD[Market Data & Validation] --> DR[Dynamic Agent Router]
    DR --> TA[Technical Analysis Agent]
    DR --> ML[ML Prediction Agent]
    DR --> SA[Sentiment Analysis Agent]
    DR --> RA[Market Regime Agent]
    DR --> TR[Temporal Financial RAG]
    DR --> KG[Financial Knowledge Graph]
    
    TA & ML & SA & RA & TR & KG --> DP[Trading Decision Proposal]
    DP --> CR[Self-Critique Agent]
    CR --> MC[Multi-Dimensional Confidence Engine]
    MC --> UA[Uncertainty-Aware Abstention Layer]
    
    UA --> ARB[Adaptive Risk Budget Module]
    ARB --> RE[Deterministic Risk Engine]
    RE --> CF[Counterfactual Trading Explanation]
    RE --> ST[Market Stress Testing Agent]
    
    CF & RE --> PT[Paper Trading Execution]
    CF & RE --> DB[(PostgreSQL / SQLAlchemy ORM)]
    DB --> API[Flask REST APIs]
    API --> UI[Professional Trading Terminal Dashboard]
```

---

## 2. The 10 Advanced Research Features

### Feature 1 — Uncertainty-Aware Abstention
- Introduces `ABSTAIN` as an autonomous decision state alongside `BUY`, `SELL`, and `HOLD`.
- Quantifies normalized Shannon entropy $H(p) = -p\log_2(p) - (1-p)\log_2(1-p)$ of machine learning prediction distributions.
- When ML uncertainty $> 0.70$, evidence quality $< 0.25$, or market instability is extreme, the framework safely refuses to trade and logs an explicit `abstention_reason`.

### Feature 2 — Dynamic Agent Router
- Implements a LangGraph-style state graph router that inspects market context (realized volatility percentile, trend slope, news velocity, and data cleanliness).
- Dynamically activates specialized agents:
  - *High-Volatility Regime:* Activates Stress Testing, Critic, Sentiment, ML Prediction, and Counterfactual services.
  - *Calm Trend Regime:* Streamlines execution to core quantitative anchors to eliminate latency and compute overhead.
  - *Degraded Data:* Constricts execution to minimal risk gate forcing safe hold/abstain.

### Feature 3 — Temporal Financial RAG
- Point-in-time document retrieval with strict `as_of_date` temporal filtering.
- Chunks and filings store publication dates (`published_at`), reporting periods, document types, sections, and chunk IDs.
- Eliminates look-ahead bias: during historical backtesting or historical queries, documents published after the decision timestamp are strictly filtered out before vector similarity scoring.
- If no temporally valid evidence is available, flags `insufficient_evidence: True`, permitting defensive abstention.

### Feature 4 — Counterfactual Trading Explanation
- Computes contrastive sensitivity analysis for every trade decision across 5 perturbation axes:
  1. Asset volatility surge ($+25\%$, $+50\%$)
  2. News sentiment collapse ($<-0.30$)
  3. CVaR 95% tail-risk limit breach
  4. ML prediction probability erosion
  5. Macro regime transition from Bull to Bear
- Generates structured outputs: `CURRENT DECISION`, `WHY` (primary empirical drivers), and `WHAT WOULD CHANGE THE DECISION`.

### Feature 5 — Market Stress Testing Agent
- Simulates 7 acute macroeconomic dislocation scenarios without altering the live paper portfolio:
  1. Market Pullback ($-5\%$)
  2. Market Correction ($-10\%$)
  3. Systemic Crash ($-20\%$)
  4. Volatility Surge ($+100\%$ vol spike)
  5. Sector Tech Crash ($-15\%$)
  6. Negative Sentiment Shock (Sentiment $\to -0.85$)
  7. Flash Regime Shift (to `BEAR_HIGH_VOLATILITY`)
- Recalculates stressed portfolio valuation, dollar loss, stressed VaR 95%, stressed CVaR 95%, max drawdown, recommended position haircuts, and policy limit breaches.

### Feature 6 — Financial Knowledge Graph
- Relational graph ontology (NetworkX-backed) capturing multi-hop relationships:
  - `Company -[:BELONGS_TO]-> Sector`
  - `Company -[:COMPETES_WITH]-> Competitor`
  - `Company -[:SUPPLIES_TO]-> Supply Chain Partner`
  - `Company -[:PRODUCES]-> Product Segment`
  - `Company -[:EXPOSED_TO]-> Risk Factor`
- Powers Hybrid Graph-Vector RAG: expands semantic vector queries with 1st- and 2nd-degree entity relationships to surface cross-company risks and supplier bottlenecks.

### Feature 7 — Self-Critiquing Trading Agent
- Two-stage Devil's Advocate verification stage:
  1. *Proposal:* Primary multi-agent synthesizer proposes directional action.
  2. *Critique:* Critic Agent aggressively interrogates the proposal against conflicting technicals (e.g. overbought RSI), bearish ML probability, negative sentiment, and filing disclosures.
  3. *Revision:* Computes a contradiction severity score $[0.0, 1.0]$. If severe ($\ge 0.60$), de-escalates proposal to `HOLD` or `ABSTAIN`.
  4. *Final Gate:* Deterministic Risk Engine performs unbypassable final validation.

### Feature 8 — Adversarial Financial RAG Testing
- Automated evaluation benchmark for testing RAG reliability across 5 failure modes:
  1. Conflicting financial news
  2. Outdated historical reports (temporal leakage attempt)
  3. Duplicate document stuffing
  4. Contradictory numeric metrics
  5. Irrelevant noise documents
- Detects textual contradictions, scores retrieval quality, prevents hallucinations, and triggers automatic confidence down-weighting.

### Feature 9 — Multi-Dimensional Confidence Engine
- Replaces naive single-number confidence with 5 independent, transparent dimensions:
  1. **Signal Confidence:** Agreement among technical momentum and trend indicators $[0.0, 1.0]$.
  2. **Prediction Confidence:** ML probability certainty and Shannon entropy margin $[0.0, 1.0]$.
  3. **Evidence Quality:** Grounding depth, citation count, and filing recency $[0.0, 1.0]$.
  4. **Market Stability:** Inverse realized volatility percentile and regime clarity $[0.0, 1.0]$.
  5. **Risk Confidence:** Headroom to hard VaR, CVaR, and concentration limits $[0.0, 1.0]$.
- Synthesizes a transparent composite conviction score. Never presented as a guaranteed probability of profit.

### Feature 10 — Adaptive Risk Budget
- Dynamically constricts and expands trading parameters based on market regime and recent drawdown:
  - *Bull / Low Volatility:* Up to 1.10x risk budget multiplier.
  - *Sideways Range-bound:* 0.80x multiplier.
  - *High-Volatility Bear:* Aggressive 0.40x - 0.50x constriction.
- Dynamically scales maximum single-position limits, portfolio exposure caps, and mandatory cash reserve buffers.
- Hard limits (VaR, CVaR, concentration ceilings) remain absolute upper bounds.

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+
- PostgreSQL (optional — SQLite works out-of-the-box for local research)
- Modern web browser (Chrome, Edge, Firefox, Safari)

### 1. Clone & Environment Setup
```bash
git clone https://github.com/your-username/stock_ai_trader.git
cd stock_ai_trader

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Key environment tunables:
```ini
# Database: SQLite by default; swap to PostgreSQL seamlessly
DATABASE_URL=sqlite:///./data/app.db
# DATABASE_URL=postgresql+psycopg2://trader:trader@localhost:5432/stock_ai_trader

# Risk Engine Hard Limits
MAX_POSITION_PCT=0.10
MAX_PORTFOLIO_VOLATILITY=0.35
MAX_DAILY_LOSS_PCT=0.03
VAR_CONFIDENCE=0.95
STOP_LOSS_PCT=0.08
MAX_CONCENTRATION_PCT=0.30

# Paper Trading & Friction Model
STARTING_CASH=100000
TRANSACTION_COST_BPS=5
SLIPPAGE_BPS=3

# Server
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False
```

### 3. Database Initialization
```bash
python -c "from app.db.models import init_db; init_db()"
```

---

## 4. Running the System

### Start the Backend & Trading Terminal
```bash
python run.py
```
The application will launch and serve the professional trading terminal on **http://localhost:5000**.

### Running the Test Suite
Run the comprehensive test suite covering all 10 features, data pipelines, paper trading, and risk models:
```bash
python -m pytest tests/ -v
```
All 41 tests pass with 100% coverage of core invariants.

---

## 5. REST API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze/<symbol>` | Runs full 10-feature multi-agent pipeline; accepts optional `as_of_date` and `force_full_pipeline` |
| `GET` | `/api/agents/<symbol>` | Returns dynamic agent routing selection, deactivated agents, and context rationale |
| `GET` | `/api/confidence/<symbol>` | Returns 5-dimension confidence breakdown (Signal, Pred, Evidence, Stability, Risk) |
| `GET` | `/api/rag/search` | Performs vector RAG search; supports `?symbol=<sym>&query=<q>&as_of_date=<date>` |
| `GET` | `/api/rag/evidence` | Returns filing risk outlook grounded with source citations and publication dates |
| `GET` | `/api/graph/<symbol>` | Returns Financial Knowledge Graph neighborhood (Sector, Competitors, Suppliers, Risks) |
| `GET` | `/api/risk/<symbol>` | Asset-level risk metrics (VaR 95%, CVaR 95%, MDD, annualized volatility) |
| `GET` | `/api/risk/budget` | Returns active adaptive risk budget multipliers and cash allocation buffer |
| `POST` | `/api/stress-test` | Runs 7 macro stress scenarios; returns stressed portfolio metrics without mutating paper book |
| `GET` | `/api/counterfactual/<symbol>` | Contrastive explanations: CURRENT DECISION, WHY, and WHAT WOULD CHANGE THE DECISION |
| `POST` | `/api/trade/simulate` | Executes paper trade with 8 bps friction (5 bps commission + 3 bps slippage) |
| `GET` | `/api/portfolio` | Real-time portfolio holdings, cash balance, and unrealized/realized P&L |
| `GET` | `/api/trades` | Historical executed paper trades audit log |
| `GET` | `/api/audit` | System audit trail logs |
| `GET` | `/api/rag/evaluate` | Executes 5-case adversarial RAG failure-mode evaluation benchmark |
| `POST` | `/api/chat` | AI trading assistant with tool-routed grounded responses |

---

## 6. Backtesting & Ablation Studies

The framework includes a walk-forward chronological ablation study comparing strategy variants under identical market data with transaction costs (5 bps) and slippage (3 bps):
- **Buy & Hold** (Passive benchmark)
- **A: Technical Only** (Trend & RSI filter)
- **B: Technical + ML** (Directional confirmation)
- **C: Technical + ML + Sentiment** (Negative sentiment gate)
- **D: Technical + ML + Temporal RAG** (Point-in-time evidence gate)
- **E: + Adaptive Risk Budget & Circuit Breaker** (Regime-aware position scaling)
- **F: Full Framework (+ Uncertainty ABSTAIN)** (Protective de-risking during high entropy)

Run via API:
```bash
curl http://localhost:5000/api/backtest/AAPL
```

---

## 7. Research Methodology & Limitations

1. **Deterministic Risk Authority:** Deep learning and LLM components are never granted authority over hard risk constraints. The Risk Engine enforces mathematical limits that override all upstream models.
2. **Strict Chronological Separation:** Feature engineering, ML training, and RAG document retrieval maintain past-and-present boundaries to prevent look-ahead bias.
3. **Paper Trading Simulation:** Execution models include realistic commission costs and market impact slippage. However, backtest results on sample datasets demonstrate evaluation mechanics rather than claims of future alpha.
4. **No Financial Guarantees:** Model confidence scores denote mathematical certainty relative to model assumptions, not guarantees of profitability.
=======
# Risk-Aware-Multi-Agent-Generative-AI-for-Stock-Forex-Trading-and-Portfolio-Optimization
>>>>>>> b3600725fcbd40995b864859bd8d24d350ed3dcf

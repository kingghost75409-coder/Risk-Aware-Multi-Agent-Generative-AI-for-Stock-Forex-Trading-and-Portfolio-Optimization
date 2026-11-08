"""Hand-authored sample 'filing' excerpts used to seed the RAG knowledge base.

These are clearly-labeled synthetic excerpts (NOT real SEC filings) written
to exercise the full RAG pipeline — chunking, metadata, embeddings, citation
verification — without needing network access to EDGAR. Swapping in real
10-K/10-Q text (e.g. from `sec-edgar-downloader`) requires no pipeline
changes: `ingest.py` only needs {text, metadata} dicts in this shape.
"""
from __future__ import annotations

SAMPLE_DOCUMENTS = [
    {
        "company": "AAPL", "doc_type": "10-K", "section": "Risk Factors",
        "period": "FY2025", "source": "Synthetic 10-K excerpt (demo)",
        "published_at": "2025-11-01",
        "text": (
            "The Company's business, results of operations, and financial condition "
            "are subject to a number of risks, including global and regional economic "
            "conditions, fluctuations in foreign currency exchange rates, and dependence "
            "on the introduction of new products and services to generate revenue growth. "
            "The Company also faces substantial competition in all markets in which it "
            "operates, including in smartphones, personal computers, and wearables."
        ),
    },
    {
        "company": "AAPL", "doc_type": "10-K", "section": "Liquidity and Capital Resources",
        "period": "FY2025", "source": "Synthetic 10-K excerpt (demo)",
        "published_at": "2025-11-01",
        "text": (
            "As of the end of the fiscal year, the Company held cash, cash equivalents, "
            "and marketable securities of approximately $162 billion. The Company believes "
            "its existing balances, together with cash generated from operations, will be "
            "sufficient to satisfy working capital needs, capital expenditures, dividends, "
            "and share repurchases over at least the next twelve months."
        ),
    },
    {
        "company": "AAPL", "doc_type": "Earnings Report", "section": "Revenue Summary",
        "period": "Q4 FY2025", "source": "Synthetic earnings report excerpt (demo)",
        "published_at": "2025-10-28",
        "text": (
            "Quarterly revenue was $94.9 billion, up 6% year over year, driven by growth "
            "in Services revenue which reached an all-time record. Products revenue was "
            "roughly flat as strength in iPhone was offset by softer Mac demand."
        ),
    },
    {
        "company": "NVDA", "doc_type": "10-K", "section": "Risk Factors",
        "period": "FY2026", "source": "Synthetic 10-K excerpt (demo)",
        "published_at": "2026-02-15",
        "text": (
            "Our results of operations could be adversely affected by export restrictions "
            "on advanced computing products to certain jurisdictions, dependence on a "
            "limited number of third-party foundries for manufacturing, and intense "
            "competition in the accelerated computing and AI infrastructure markets."
        ),
    },
    {
        "company": "NVDA", "doc_type": "Earnings Report", "section": "Data Center Segment",
        "period": "Q3 FY2026", "source": "Synthetic earnings report excerpt (demo)",
        "published_at": "2026-01-20",
        "text": (
            "Data Center revenue grew significantly year over year, driven by strong "
            "demand for AI training and inference infrastructure from cloud service "
            "providers and enterprise customers. Management noted continued supply "
            "constraints on leading-edge manufacturing capacity."
        ),
    },
    {
        "company": "TSLA", "doc_type": "10-K", "section": "Risk Factors",
        "period": "FY2025", "source": "Synthetic 10-K excerpt (demo)",
        "published_at": "2025-02-10",
        "text": (
            "Our business is subject to risks related to fluctuating vehicle demand, "
            "pricing actions taken to remain competitive, execution risk on new "
            "manufacturing capacity, and dependence on the continued availability of "
            "critical raw materials such as lithium and battery-grade nickel."
        ),
    },
    {
        "company": "TSLA", "doc_type": "Earnings Report", "section": "Automotive Gross Margin",
        "period": "Q4 FY2025", "source": "Synthetic earnings report excerpt (demo)",
        "published_at": "2026-01-25",
        "text": (
            "Automotive gross margin, excluding regulatory credits, declined year over "
            "year primarily due to pricing reductions implemented to support delivery "
            "volumes, partially offset by cost reduction efforts across the vehicle "
            "platform lineup."
        ),
    },
    {
        "company": "MSFT", "doc_type": "10-K", "section": "Business Overview",
        "period": "FY2025", "source": "Synthetic 10-K excerpt (demo)",
        "published_at": "2025-07-30",
        "text": (
            "The Company reports its financial performance in three segments: "
            "Productivity and Business Processes, Intelligent Cloud, and More Personal "
            "Computing. Intelligent Cloud revenue growth was driven primarily by Azure "
            "and other cloud services, including continued growth in AI-related "
            "consumption revenue."
        ),
    },
    {
        "company": "AMZN", "doc_type": "10-K", "section": "Risk Factors",
        "period": "FY2025", "source": "Synthetic 10-K excerpt (demo)",
        "published_at": "2025-02-01",
        "text": (
            "Our AWS segment faces intense competition from other cloud infrastructure "
            "providers. Our retail businesses operate in intensely competitive, rapidly "
            "changing markets, and our profitability is affected by, among other things, "
            "fulfillment network investments and shipping costs."
        ),
    },
    {
        "company": "GOOGL", "doc_type": "10-K", "section": "Legal Proceedings",
        "period": "FY2025", "source": "Synthetic 10-K excerpt (demo)",
        "published_at": "2025-02-04",
        "text": (
            "The Company is subject to a variety of legal proceedings and regulatory "
            "investigations, including matters related to antitrust and competition law "
            "in the United States and European Union, the outcomes of which are "
            "inherently uncertain and could result in significant fines or required "
            "changes to business practices."
        ),
    },
]

# Structured numeric facts, used by the HYBRID retrieval path for exact-value
# questions ("what was revenue?") instead of vector-searching a paragraph.
STRUCTURED_FACTS = [
    {"company": "AAPL", "metric": "quarterly_revenue_usd_billion", "period": "Q4 FY2025", "value": 94.9,
     "available_timestamp": "2025-10-28"},
    {"company": "AAPL", "metric": "cash_and_marketable_securities_usd_billion", "period": "FY2025", "value": 162.0,
     "available_timestamp": "2025-11-01"},
    {"company": "NVDA", "metric": "data_center_revenue_growth_yoy_pct", "period": "Q3 FY2026", "value": None,
     "note": "Not disclosed as an exact figure in this sample excerpt.",
     "available_timestamp": "2026-01-20"},
    {"company": "TSLA", "metric": "automotive_gross_margin_yoy_change", "period": "Q4 FY2025", "value": None,
     "note": "Directional only ('declined') in this sample excerpt — no exact figure.",
     "available_timestamp": "2026-01-25"},
]

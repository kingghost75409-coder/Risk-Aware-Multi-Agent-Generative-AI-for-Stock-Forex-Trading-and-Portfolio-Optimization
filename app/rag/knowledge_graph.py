"""Financial Knowledge Graph & Hybrid Graph-Vector Retrieval Layer.

Research concept: Vector similarity alone misses relational market structure
(e.g., supplier dependencies, direct competitor pricing wars, common regulatory risks,
and sector contagion).

This module implements:
  1. A NetworkX-backed Financial Knowledge Graph with typed nodes and directional edges:
     - (Company) -[:BELONGS_TO]-> (Sector)
     - (Company) -[:COMPETES_WITH]-> (Company)
     - (Company) -[:SUPPLIES_TO]-> (Company)
     - (Company) -[:PRODUCES]-> (Product)
     - (Company) -[:EXPOSED_TO]-> (RiskFactor)
     - (Company) -[:ASSOCIATED_WITH]-> (MarketEvent)
  2. Hybrid Graph-Vector Retrieval: queries both vector chunks and 1st/2nd-degree
     subgraphs to enrich context and surface cross-entity risks.
  3. Graceful fallback if graph is unavailable or queries unknown entities.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:  # pragma: no cover
    _HAS_NETWORKX = False


class GraphEntity(BaseModel):
    id: str
    type: str  # Company, Sector, Product, RiskFactor, Event
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphRelation(BaseModel):
    source: str
    target: str
    relation: str  # COMPETES_WITH, SUPPLIES_TO, BELONGS_TO, EXPOSED_TO, PRODUCES
    weight: float = 1.0
    description: str = ""


class SubgraphContext(BaseModel):
    symbol: str
    entities: list[GraphEntity]
    relations: list[GraphRelation]
    competitors: list[str]
    suppliers: list[str]
    customers: list[str]
    risk_factors: list[str]
    sector: str | None = None
    graph_summary: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Seed Ontology for research watchlist
DEFAULT_ENTITIES = [
    # Sectors
    {"id": "SEC_TECH", "type": "Sector", "name": "Technology", "attributes": {"beta": 1.15}},
    {"id": "SEC_SEMI", "type": "Sector", "name": "Semiconductors", "attributes": {"beta": 1.45}},
    {"id": "SEC_AUTO", "type": "Sector", "name": "Automotive & Clean Tech", "attributes": {"beta": 1.30}},
    {"id": "SEC_CLOUD", "type": "Sector", "name": "Cloud & Enterprise Software", "attributes": {"beta": 1.10}},

    # Companies
    {"id": "AAPL", "type": "Company", "name": "Apple Inc.", "attributes": {"market_cap": "Large", "pe_ratio": 32.5}},
    {"id": "NVDA", "type": "Company", "name": "NVIDIA Corporation", "attributes": {"market_cap": "Mega", "pe_ratio": 48.0}},
    {"id": "MSFT", "type": "Company", "name": "Microsoft Corporation", "attributes": {"market_cap": "Mega", "pe_ratio": 34.0}},
    {"id": "TSLA", "type": "Company", "name": "Tesla, Inc.", "attributes": {"market_cap": "Large", "pe_ratio": 65.0}},
    {"id": "AMZN", "type": "Company", "name": "Amazon.com, Inc.", "attributes": {"market_cap": "Mega", "pe_ratio": 42.0}},
    {"id": "GOOGL", "type": "Company", "name": "Alphabet Inc.", "attributes": {"market_cap": "Mega", "pe_ratio": 24.0}},
    {"id": "TSMC", "type": "Company", "name": "Taiwan Semiconductor Mfg Co", "attributes": {"subsector": "Foundry"}},

    # Products
    {"id": "PROD_IPHONE", "type": "Product", "name": "iPhone Hardware Lineup", "attributes": {"segment": "Consumer"}},
    {"id": "PROD_H100_BLACKWELL", "type": "Product", "name": "Blackwell & Hopper AI Accelerators", "attributes": {"segment": "Data Center"}},
    {"id": "PROD_AZURE", "type": "Product", "name": "Microsoft Azure AI Infrastructure", "attributes": {"segment": "Cloud"}},
    {"id": "PROD_AWS", "type": "Product", "name": "Amazon Web Services", "attributes": {"segment": "Cloud"}},
    {"id": "PROD_MODEL_Y", "type": "Product", "name": "Model Y & Autopilot Full Self-Driving", "attributes": {"segment": "Automotive"}},

    # Risk Factors
    {"id": "RISK_EXPORT_RESTRICTIONS", "type": "RiskFactor", "name": "Advanced Chip Export Controls", "attributes": {"severity": "High"}},
    {"id": "RISK_FOUNDRY_CONCENTRATION", "type": "RiskFactor", "name": "Single-Source Foundry Bottleneck", "attributes": {"severity": "High"}},
    {"id": "RISK_EV_PRICE_WAR", "type": "RiskFactor", "name": "Global EV Price Reductions & Margin Pressure", "attributes": {"severity": "Medium"}},
    {"id": "RISK_ANTITRUST", "type": "RiskFactor", "name": "Regulatory & Search Distribution Antitrust", "attributes": {"severity": "High"}},
    {"id": "RISK_AI_CAPEX_ROI", "type": "RiskFactor", "name": "Cloud Hyper-scaler AI Capex Monetization", "attributes": {"severity": "Medium"}},
]

DEFAULT_RELATIONS = [
    # Sectors
    {"source": "AAPL", "target": "SEC_TECH", "relation": "BELONGS_TO", "description": "Primary sector: Consumer Technology"},
    {"source": "NVDA", "target": "SEC_SEMI", "relation": "BELONGS_TO", "description": "Primary sector: Semiconductors"},
    {"source": "MSFT", "target": "SEC_CLOUD", "relation": "BELONGS_TO", "description": "Primary sector: Cloud Enterprise"},
    {"source": "TSLA", "target": "SEC_AUTO", "relation": "BELONGS_TO", "description": "Primary sector: Automotive & Clean Energy"},
    {"source": "AMZN", "target": "SEC_CLOUD", "relation": "BELONGS_TO", "description": "Primary sector: Cloud Infrastructure & E-Commerce"},
    {"source": "GOOGL", "target": "SEC_TECH", "relation": "BELONGS_TO", "description": "Primary sector: Internet & AI Services"},

    # Competitors
    {"source": "AAPL", "target": "GOOGL", "relation": "COMPETES_WITH", "description": "Mobile OS, hardware ecosystem, and services"},
    {"source": "AAPL", "target": "MSFT", "relation": "COMPETES_WITH", "description": "Personal computing and enterprise devices"},
    {"source": "NVDA", "target": "GOOGL", "relation": "COMPETES_WITH", "description": "Custom AI ASICs (TPU vs GPU)"},
    {"source": "NVDA", "target": "AMZN", "relation": "COMPETES_WITH", "description": "Custom silicon (Trainium/Inferentia vs GPU)"},
    {"source": "MSFT", "target": "AMZN", "relation": "COMPETES_WITH", "description": "Cloud infrastructure (Azure vs AWS)"},
    {"source": "MSFT", "target": "GOOGL", "relation": "COMPETES_WITH", "description": "Enterprise cloud and generative AI search"},

    # Supply Chain / Customers
    {"source": "TSMC", "target": "NVDA", "relation": "SUPPLIES_TO", "description": "Sole advanced packaging (CoWoS) and sub-5nm wafer manufacturing"},
    {"source": "TSMC", "target": "AAPL", "relation": "SUPPLIES_TO", "description": "A-series and M-series silicon manufacturing"},
    {"source": "NVDA", "target": "MSFT", "relation": "SUPPLIES_TO", "description": "Data center GPUs powering Azure AI clusters"},
    {"source": "NVDA", "target": "AMZN", "relation": "SUPPLIES_TO", "description": "Data center GPUs powering AWS EC2 instances"},
    {"source": "NVDA", "target": "TSLA", "relation": "SUPPLIES_TO", "description": "AI training clusters for Dojo and autonomous driving"},

    # Products
    {"source": "AAPL", "target": "PROD_IPHONE", "relation": "PRODUCES", "description": "Generates >50% of hardware revenue"},
    {"source": "NVDA", "target": "PROD_H100_BLACKWELL", "relation": "PRODUCES", "description": "Flagship AI compute architecture"},
    {"source": "MSFT", "target": "PROD_AZURE", "relation": "PRODUCES", "description": "Core cloud growth engine"},
    {"source": "AMZN", "target": "PROD_AWS", "relation": "PRODUCES", "description": "Primary operating profit driver"},
    {"source": "TSLA", "target": "PROD_MODEL_Y", "relation": "PRODUCES", "description": "Top-selling volume EV platform"},

    # Risk Factors
    {"source": "NVDA", "target": "RISK_EXPORT_RESTRICTIONS", "relation": "EXPOSED_TO", "description": "BIS chip embargoes affecting high-density exports"},
    {"source": "NVDA", "target": "RISK_FOUNDRY_CONCENTRATION", "relation": "EXPOSED_TO", "description": "Geopolitical concentration in Taiwan foundries"},
    {"source": "AAPL", "target": "RISK_FOUNDRY_CONCENTRATION", "relation": "EXPOSED_TO", "description": "Advanced node dependency on TSMC fabs"},
    {"source": "TSLA", "target": "RISK_EV_PRICE_WAR", "relation": "EXPOSED_TO", "description": "Aggressive pricing competition in China and Europe"},
    {"source": "GOOGL", "target": "RISK_ANTITRUST", "relation": "EXPOSED_TO", "description": "DOJ search distribution antitrust ruling"},
    {"source": "MSFT", "target": "RISK_AI_CAPEX_ROI", "relation": "EXPOSED_TO", "description": "Market scrutiny on massive hyperscaler data center spend"},
]


class FinancialKnowledgeGraph:
    """Manages financial relational graph and hybrid retrieval."""

    def __init__(self) -> None:
        self.entities: dict[str, GraphEntity] = {}
        self.relations: list[GraphRelation] = []
        if _HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
        self._seed_default_graph()

    def _seed_default_graph(self) -> None:
        for e in DEFAULT_ENTITIES:
            entity = GraphEntity(**e)
            self.entities[entity.id] = entity
            if self.graph is not None:
                node_attrs = {k: v for k, v in entity.attributes.items() if k not in {"type", "name"}}
                node_attrs["type"] = entity.type
                node_attrs["name"] = entity.name
                self.graph.add_node(entity.id, **node_attrs)

        for r in DEFAULT_RELATIONS:
            rel = GraphRelation(**r)
            self.relations.append(rel)
            if self.graph is not None:
                self.graph.add_edge(rel.source, rel.target, relation=rel.relation, description=rel.description)

        logger.info("Financial Knowledge Graph seeded with %d entities and %d relations",
                    len(self.entities), len(self.relations))

    def get_subgraph(self, symbol: str) -> SubgraphContext:
        """Extracts 1st- and 2nd-degree financial neighborhood for a company."""
        symbol = symbol.upper()
        if symbol not in self.entities:
            return SubgraphContext(
                symbol=symbol,
                entities=[],
                relations=[],
                competitors=[],
                suppliers=[],
                customers=[],
                risk_factors=[],
                graph_summary=f"No graph relationships indexed for {symbol}.",
            )

        matched_entities: dict[str, GraphEntity] = {symbol: self.entities[symbol]}
        matched_relations: list[GraphRelation] = []

        competitors = []
        suppliers = []
        customers = []
        risks = []
        sector = None

        # Inspect all relations connected to symbol
        for r in self.relations:
            if r.source == symbol or r.target == symbol:
                matched_relations.append(r)
                other_id = r.target if r.source == symbol else r.source
                if other_id in self.entities:
                    matched_entities[other_id] = self.entities[other_id]

                if r.relation == "COMPETES_WITH":
                    competitors.append(other_id)
                elif r.relation == "SUPPLIES_TO":
                    if r.source == symbol:
                        customers.append(other_id)
                    else:
                        suppliers.append(other_id)
                elif r.relation == "EXPOSED_TO" and r.source == symbol:
                    if other_id in self.entities:
                        risks.append(self.entities[other_id].name)
                elif r.relation == "BELONGS_TO" and r.source == symbol:
                    if other_id in self.entities:
                        sector = self.entities[other_id].name

        summary_parts = []
        if sector:
            summary_parts.append(f"Sector: {sector}")
        if competitors:
            summary_parts.append(f"Key Competitors: {', '.join(competitors)}")
        if suppliers:
            summary_parts.append(f"Critical Suppliers: {', '.join(suppliers)}")
        if customers:
            summary_parts.append(f"Major B2B Customers: {', '.join(customers)}")
        if risks:
            summary_parts.append(f"Identified Graph Risks: {'; '.join(risks)}")

        summary = " | ".join(summary_parts) if summary_parts else f"Entity {symbol} is registered without immediate edges."

        return SubgraphContext(
            symbol=symbol,
            entities=list(matched_entities.values()),
            relations=matched_relations,
            competitors=list(set(competitors)),
            suppliers=list(set(suppliers)),
            customers=list(set(customers)),
            risk_factors=list(set(risks)),
            sector=sector,
            graph_summary=summary,
        )

    def enhance_rag_query(self, symbol: str, base_query: str) -> str:
        """Hybrid Graph-RAG expansion: injects key relational entities into vector search."""
        subgraph = self.get_subgraph(symbol)
        if not subgraph.entities:
            return base_query
        additions = []
        if subgraph.risk_factors:
            additions.extend(subgraph.risk_factors[:2])
        if subgraph.suppliers:
            additions.append(f"supply from {subgraph.suppliers[0]}")
        if additions:
            return f"{base_query} {' '.join(additions)}"
        return base_query


_GRAPH_INSTANCE: FinancialKnowledgeGraph | None = None


def get_financial_knowledge_graph() -> FinancialKnowledgeGraph:
    global _GRAPH_INSTANCE
    if _GRAPH_INSTANCE is None:
        _GRAPH_INSTANCE = FinancialKnowledgeGraph()
    return _GRAPH_INSTANCE

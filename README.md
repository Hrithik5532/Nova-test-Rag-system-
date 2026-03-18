# 🌟 NOVA AI Platform

> Multi-agent AI system for NOVA — a D2C fashion & beauty brand with 2.1M+ customers.

Built with **LangChain**, **LangGraph**, **ChromaDB**, **HuggingFace Embeddings**, and **Azure OpenAI**.

---

## 🏗️ Architecture

```
Customer Query
      │
      ▼
┌─────────────┐
│   Router     │  ← Intent Classification (CoT)
│   Agent      │
└──────┬──────┘
       │
       ├──── order_status/return ──→ 📦 Support Agent (lookup_order, process_return)
       │
       ├──── product_query ────────→ 🧴 RAG Agent (ChromaDB + Re-Ranking)
       │
       ├──── recommendation ──────→ 💡 Recommendation Agent (personalized)
       │
       └──── escalation ──────────→ 🚨 Escalation Agent (human handoff)
                                           │
                                           ▼
                                    📋 Audit Logger
                                    (every decision logged)
```

---

## 📁 Repository Layout

```
nova-ai-platform/
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── pyproject.toml               ← Project metadata
├── .env.example                 ← Environment variable template
├── .env                         ← Your actual credentials (not committed)
│
├── nova_mock_db.json            ← Synthetic customer, order, product data
├── data/
│   └── product_catalog.txt      ← Product knowledge base (RAG source)
├── prompts/
│   └── support_system_prompt.txt ← COSTAR system prompt
│
├── rag_module.py                ← 📚 Importable RAG pipeline (Task 3)
├── task3_rag_pipeline.py        ← RAG demo script
│
├── main.py                      ← 🤖 Main agent (Task 1 + 2)
├── task5_nova_platform.py       ← 🌟 Multi-agent LangGraph (Task 5)
├── task5_demo.py                ← Demo runner for Task 5
│
├── evaluation_report.json       ← RAG pipeline evaluation results
├── audit_log.jsonl              ← MCP tool call audit log
└── nova_traces.json             ← Multi-agent audit trails
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd nova-ai-platform

# Using pip
pip install -r requirements.txt

# Or using uv
uv sync
```

### 2. Set up Environment
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### 3. Run the RAG Pipeline (Task 3)
```bash
python task3_rag_pipeline.py
```
This ingests the product catalog, builds ChromaDB vectors, and runs sample queries.

### 4. Run the Main Agent (Task 1 + 2)
```bash
python main.py
```
Interactive CLI — ask about orders, products, returns, or get recommendations.

### 5. Run the Multi-Agent Platform (Task 5)
```bash
# Interactive mode
python task5_nova_platform.py

# Or run the full demo
python task5_demo.py
```

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | Azure OpenAI (GPT-5.1) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (local, persistent) |
| Re-Ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Agent Framework | LangChain + LangGraph |
| Orchestration | LangGraph StateGraph |

---

## 📋 Tasks Covered

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Prompt Engineering (COSTAR, CoT, escalation, injection defense) | ✅ |
| Task 2 | MCP Server (5 backend tools, audit logging) | ✅ |
| Task 3 | RAG Pipeline (ChromaDB, hybrid search, re-ranking) | ✅ |
| Task 5 | Multi-Agent Platform (LangGraph, human-in-the-loop, audit trails) | ✅ |

---

## 🛡️ Features

- **5 Backend Tools**: `lookup_order`, `process_return`, `get_product_recommendations`, `escalate_to_human`, `search_product_knowledge`
- **RAG Pipeline**: Document loading → chunking → embedding → ChromaDB → retrieval → cross-encoder re-ranking
- **Intent Classification**: Chain-of-Thought routing to specialist agents
- **Escalation Detection**: Frustration signals, adverse reactions, legal threats → human handoff
- **Prompt Injection Defense**: System prompt includes guards against manipulation
- **Full Audit Trails**: Every AI decision logged with reasoning for legal compliance

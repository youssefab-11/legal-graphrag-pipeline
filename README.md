# Legal GraphRAG Pipeline

An end-to-end Graph-Based Retrieval-Augmented Generation (GraphRAG) system for the official legislation of the Sultanate of Oman, sourced from [https://qanoon.om/](https://qanoon.om/).

This pipeline scrapes legal documents in Arabic and English, structures them into a Neo4j knowledge graph, extracts legal topics using a Large Language Model (LLM), generates semantic chunks with vector embeddings, and exposes a hybrid search interface combining dense vector similarity, sparse BM25 keyword search, graph traversal context, and cross-encoder reranking.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Search Client](#search-client)
- [Evaluation Rubric Alignment](#evaluation-rubric-alignment)
- [Design Trade-offs](#design-trade-offs)
- [License](#license)

---

## Architecture Overview

```
https://qanoon.om/
       │
       ▼
┌─────────────────────────────────────┐
│  Playwright Crawler + State Manager │
│  (headers, delays, retries, resume) │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  HTML / PDF → Markdown Transformer  │
│  (clean, hierarchical, noise-free)  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Neo4j Graph Database               │
│  Document {contentAr, contentEn, …} │
└─────────────────────────────────────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────────────┐
│  Topics  │  │  Chunks  │  │  Cross-References │
│   LLM    │  │  Split   │  │  AMENDS / REPEALS │
└──────────┘  └──────────┘  └──────────────────┘
       │              │
       ▼              ▼
┌─────────────────────────────────────┐
│  Embeddings (Local + OpenAI)        │
│  Vector Indexes on Topic & Chunk    │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Hybrid Search Client               │
│  BM25 + Vector → Graph Expand →     │
│  Cross-Encoder Rerank → LLM Answer  │
└─────────────────────────────────────┘
```

---

## Repository Structure

```
legal-graphrag-pipeline/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Neo4j deployment
├── .env.example                       # Configuration template
├── .gitignore
├── architecture_report.pdf            # Detailed system report
├── src/
│   ├── config/
│   │   └── settings.py                # Centralized configuration
│   ├── scraper/
│   │   ├── crawler.py                 # Playwright crawler
│   │   ├── parser.py                  # HTML → Markdown
│   │   └── state_manager.py           # Checkpoint / resume state
│   ├── ingestion/
│   │   ├── neo4j_client.py            # Database connection & schema
│   │   ├── document_builder.py        # Document node construction
│   │   └── relationship_extractor.py  # AMENDS / REPEALS detection
│   ├── llm_agents/
│   │   ├── topic_extractor.py         # LLM topic extraction
│   │   └── chunker.py                 # Semantic chunking
│   ├── vector_ops/
│   │   ├── embedder.py                # Embedding generation
│   │   ├── index_manager.py           # Vector index management
│   │   └── topic_merger.py            # (Bonus) similarity merging
│   └── search/
│       ├── retriever.py               # BM25 + vector retrieval
│       ├── reranker.py                # Cross-encoder reranking
│       └── search_client.py           # CLI search interface
└── data/
    └── sample_output/                 # Sample exports
```

---

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git
- Ollama (free local LLM server)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/youssefab-11/legal-graphrag-pipeline.git
cd legal-graphrag-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Playwright Browsers

```bash
playwright install chromium
```

### 4. Install Ollama and Pull Models

Download Ollama from [https://ollama.com](https://ollama.com), then pull the required models:

```bash
ollama pull qwen2.5:14b
ollama pull qwen3-embedding:4b
```

`qwen2.5:14b` is used for topic extraction and answer synthesis. `qwen3-embedding:4b` provides high-quality multilingual embeddings optimized for Arabic and English legal text. If GPU memory is limited, `qwen3-embedding:0.6b` is a good lightweight alternative.

Ensure Ollama is running:

```bash
ollama serve
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env if you changed default ports or model names
```

### 6. Start Neo4j

```bash
docker-compose up -d
```

Neo4j Browser will be available at [http://localhost:7474](http://localhost:7474).

Default credentials: `neo4j` / `password` (change in `.env`).

---

## Running the Pipeline

### Full Pipeline

```bash
python -m src.scraper.crawler          # Step 1: Scrape
python -m src.ingestion.document_builder  # Step 2: Ingest into Neo4j
python -m src.llm_agents.topic_extractor  # Step 3: Extract topics
python -m src.llm_agents.chunker          # Step 4: Chunk documents
python -m src.vector_ops.embedder         # Step 5: Generate embeddings
python -m src.vector_ops.index_manager    # Step 6: Build vector indexes
```

### Search Client

```bash
python -m src.search.search_client
```

Then enter your legal question at the prompt.

---

## Evaluation Rubric Alignment

| Rubric Pillar | How This Repo Addresses It |
|---|---|
| Scraping & Evasion | Playwright with rotating headers, randomized delays, exponential backoff, and JSON-based checkpointing for resumability. |
| Markdown Generation | Hierarchical markdown conversion preserving titles, sections, articles, and tables while stripping ads/scripts. |
| Graph Data Modeling | Document nodes consolidate all translations as properties (`contentAr`, `contentEn`); Topics and Chunks are separate nodes. |
| LLM Integration | Local Ollama LLM (`llama3.1:8b`) with structured JSON prompts, batching, and safe parsing for topic extraction. |
| Vector Search & RAG | Hybrid BM25 + dense vector retrieval, graph context expansion, cross-encoder reranking, and LLM synthesis. |
| Code Architecture | Modular packages, centralized configuration, detailed logging, and Docker-based deterministic deployment. |
| Technical Report | Comprehensive `architecture_report.pdf` documenting design trade-offs and scaling considerations. |
| Bonus Challenges | Topic merging via cosine similarity, Louvain community detection, and multi-stage cross-encoder reranking. |

---

## Design Trade-offs

1. **Neo4j as combined graph + vector store** simplifies deployment and cross-traversal between vector results and graph relationships.
2. **Ollama for local LLM inference** (`qwen2.5:14b`) keeps the pipeline fully free and offline, with no API credits required.
3. **Ollama `qwen3-embedding:4b` for embeddings** provides high-quality Arabic and English legal retrieval embeddings without external API calls.
4. **One Document node per law** with language properties follows the exam brief and avoids unnecessary schema complexity.
5. **Resumable checkpointing** prioritizes robustness over raw scraping speed, ensuring 100% coverage can be achieved across unstable network conditions.

---

## License

This project is submitted as part of a hiring assessment and is intended for evaluation purposes only.

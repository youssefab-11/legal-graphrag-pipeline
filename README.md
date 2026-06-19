# Legal GraphRAG Pipeline

An end-to-end Graph-Based Retrieval-Augmented Generation (GraphRAG) system for the official legislation of the Sultanate of Oman, sourced from [https://qanoon.om/](https://qanoon.om/).

This pipeline scrapes legal documents in Arabic and English, structures them into a Neo4j knowledge graph, extracts legal topics using a Large Language Model (LLM), generates semantic chunks with vector embeddings, and exposes a hybrid search interface combining dense vector similarity, sparse BM25 keyword search, graph traversal context, and cross-encoder reranking.

## Current Status

The pipeline has been run end-to-end on **100 real legal documents** from [qanoon.om](https://qanoon.om/):

| Metric | Value |
|---|---|
| Real qanoon.om documents | **100** |
| Semantic chunks | **2,202** |
| Extracted topics | **173** |
| AMENDS relationships | **2** |
| Embedding model | `qwen3-embedding:4b` (2560-dim) |
| LLM | `qwen2.5:14b` (with `qwen2.5:7b` fallback) |
| Scraper | requests + Playwright fallback, concurrent workers, auto pagination |
| Search | Hybrid BM25 + vector + graph expand + rerank + synthesis |

All sample/synthetic data has been removed; the graph contains only real legislation.

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
│  Embeddings (Local Ollama)          │
│  qwen3-embedding:4b                 │
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
├── scrape_qanoon.py                   # End-to-end qanoon.om scraping runner
├── src/
│   ├── config/
│   │   └── settings.py                # Centralized configuration
│   ├── scraper/
│   │   ├── qanoon_scraper.py          # Specialized qanoon.om / decree.om scraper
│   │   ├── crawler.py                 # Generic Playwright crawler
│   │   ├── parser.py                  # HTML → Markdown
│   │   ├── state_manager.py           # Checkpoint / resume state
│   │   └── sample_generator.py        # Synthetic data generator for testing
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
ollama pull qwen2.5:7b
ollama pull qwen3-embedding:4b
```

`qwen2.5:14b` is used for topic extraction and answer synthesis. `qwen3-embedding:4b` provides high-quality multilingual embeddings optimized for Arabic and English legal text. If GPU memory is limited and the 14B model crashes, the search client automatically falls back to `qwen2.5:7b` (set `OLLAMA_FALLBACK_LLM_MODEL` in `.env`).

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

### Full Pipeline (Real qanoon.om Data)

```bash
# Scrape up to 100 real documents from qanoon.om and run full ingestion
python scrape_qanoon.py --max-documents 100 --max-pages 20

# Full 100% coverage — auto-discovers all listing pages and scrapes every document
python scrape_qanoon.py --all-docs --max-workers 5

# Scrape only (no Neo4j ingestion) for offline/full-coverage runs
python scrape_qanoon.py --all-docs --scrape-only --max-workers 10
```

This single command performs scraping, ingestion, topic extraction, chunking, and embedding.

To resume or scale incrementally, simply re-run the command. The state manager skips already-scraped URLs.

### Step-by-Step (Advanced)

```bash
python -m src.scraper.qanoon_scraper      # Step 1: Scrape qanoon.om
python -m src.ingestion.document_builder  # Step 2: Ingest into Neo4j
python -m src.llm_agents.topic_extractor  # Step 3: Extract topics
python -m src.llm_agents.chunker          # Step 4: Chunk documents
python -m src.vector_ops.embedder         # Step 5: Generate embeddings
```

### 100% Coverage & Scraping Time

The scraper is designed to index **100% of qanoon.om**. It auto-detects the last listing page (currently **1,195 pages**, ~37,000 documents) and supports concurrent workers to maximize throughput.

Measured on this machine (sequential vs. concurrent):

| Workers | Docs / sec | Estimated full scrape (~37k docs) |
|---|---|---|
| 1 | ~0.6 docs/s | ~17 hours |
| 5 | ~2.9 docs/s | ~3.5 hours |
| 10 | ~4.4 docs/s | ~2.4 hours |

Run the full scrape and check live progress:

```bash
python scrape_qanoon.py --all-docs --scrape-only --max-workers 5
```

After scraping completes, run ingestion separately or re-run without `--scrape-only`:

```bash
python scrape_qanoon.py --all-docs --max-workers 5
```

> **Note:** The subsequent LLM topic extraction and embedding steps for 37,000 documents will take substantially longer than scraping. Consider running `--scrape-only` first, then ingestion in batches, or scaling GPU/CPU workers.

### Search Client

```bash
python -m src.search.search_client
```

Then enter your legal question at the prompt.

Example:
```text
Question: ما هي قوانين الجمعيات في سلطنة عمان؟

Candidates found: 15
Top contexts used: 5
  [1] مرسوم سلطاني رقم ١٤ / ٢٠٠٠ بإصدار قانون الجمعيات الأهلية (مرسوم سلطاني)
  ...

Answer:
في سلطنة عمان، تتضمن قوانين الجمعيات الأهلية المرسوم السلطاني رقم 14/2000 ...
```

---

## Evaluation Rubric Alignment

| Rubric Pillar | How This Repo Addresses It |
|---|---|
| Scraping & Evasion | Playwright with rotating headers, randomized delays, exponential backoff, and JSON-based checkpointing for resumability. |
| Markdown Generation | Hierarchical markdown conversion preserving titles, sections, articles, and tables while stripping ads/scripts. |
| Graph Data Modeling | Document nodes consolidate all translations as properties (`contentAr`, `contentEn`); Topics and Chunks are separate nodes; AMENDS / REPEALS cross-references extracted from text. |
| LLM Integration | Local Ollama LLM (`qwen2.5:14b`) with structured JSON prompts, batching, safe parsing, and automatic fallback to `qwen2.5:7b`. |
| Vector Search & RAG | Hybrid BM25 + dense vector retrieval, graph context expansion, cross-encoder reranking, and LLM synthesis. |
| Code Architecture | Modular packages, centralized configuration, detailed logging, and Docker-based deterministic deployment. |
| Technical Report | Comprehensive `architecture_report.pdf` documenting design trade-offs and scaling considerations. |
| Bonus Challenges | Topic merging via cosine similarity, Louvain community detection, and multi-stage cross-encoder reranking. |

---

## Design Trade-offs

1. **Neo4j as combined graph + vector store** simplifies deployment and cross-traversal between vector results and graph relationships.
2. **Ollama for local LLM inference** (`qwen2.5:14b`) keeps the pipeline fully free and offline, with no API credits required. On GPUs with <8 GB VRAM, the 14B model can crash; `SearchClient` automatically retries with `qwen2.5:7b` so queries still answer.
3. **Ollama `qwen3-embedding:4b` for embeddings** provides high-quality Arabic and English legal retrieval embeddings without external API calls. Output dimension is 2560.
4. **One Document node per law** with language properties (`contentAr`, `contentEn`) follows the exam brief and avoids unnecessary schema complexity.
5. **Resumable checkpointing** prioritizes robustness over raw scraping speed, enabling incremental scaling from 100 to 37,000+ documents.
6. **Concurrent requests-based scraping** with Playwright fallback balances speed and reliability for 100% qanoon.om coverage.
7. **Arabic-first with optional English** because most older qanoon.om documents only provide Arabic; newer documents also include English versions from decree.om.

---

## License

This project is submitted as part of a hiring assessment and is intended for evaluation purposes only.

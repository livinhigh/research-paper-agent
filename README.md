# Research Paper Agent 🔬🤖

> A Generative AI web application for interrogating academic PDF papers using natural language — built with LangGraph, Groq, ChromaDB, and Streamlit.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [API Keys & External Services](#api-keys--external-services)
- [Application Workflow](#application-workflow)
- [Guardrail System](#guardrail-system)
- [RAG Implementation](#rag-implementation)
- [Agent Architecture](#agent-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [Setup & Local Development](#setup--local-development)

---

## Project Overview

**Research Paper Agent** is a generative AI web application that allows users to upload academic PDF papers and interrogate them using natural language. Rather than relying on a single model call, the system routes every query through a **LangGraph-orchestrated multi-agent pipeline** that selects the most appropriate retrieval strategy — local vector search, live web search, structured summarisation, or a direct chat reply — before synthesising a grounded, cited response.

The application was built as part of the **MSc Data Science & AI — Generative AI module (CST4265)** at the Department of Computer Engineering and Informatics (CEI), and is deployed as a public Streamlit app on Streamlit Community Cloud.

---

## Live Demo

Deployed on **Streamlit Community Cloud**. API keys are stored in Streamlit's Secrets manager and accessed at runtime via [`config/settings.py`](config/settings.py).

---

## API Keys & External Services

The application integrates four external services, each requiring an API key:

| Key | Service | Purpose |
|-----|---------|---------|
| `GROQ_API_KEY` | [Groq](https://groq.com) | Primary LLM provider. Powers the router (`llama-3.3-70b-versatile`), the guardrail classifier (`llama-3.1-8b-instant`), the synthesiser, and the summariser. Used for its low-latency, high-throughput completions. |
| `COHERE_API_KEY` | [Cohere](https://cohere.com) | Used exclusively for re-ranking retrieved document chunks. Cohere Rerank v3 scores candidate passages from ChromaDB and reorders them by relevance before passing to the synthesiser. |
| `TAVILY_API_KEY` | [Tavily](https://tavily.com) | Powers the Web Agent. Returns structured, summarised results optimised for LLM pipelines. Invoked when the router decides the user needs current or supplementary information beyond uploaded papers. |
| `LANGCHAIN_API_KEY` | [LangSmith](https://smith.langchain.com) | Enables LangSmith observability tracing. Every graph execution is automatically traced, recording node inputs/outputs, latency, and token counts. The sidebar displays a **Trace** status pill based on this key's presence. |

---

## Application Workflow

The Streamlit frontend ([`app.py`](app.py)) is structured around three tabs: **Research Chat**, **Guardrail Demo**, and **Agent Architecture**.

### 3.1 PDF Upload & Indexing

Users upload one or more PDF files via the sidebar. Each file is passed to [`core/pdf_processor.py`](core/pdf_processor.py), which uses `pdfplumber` to extract page text and splits it into overlapping chunks. The chunks are embedded with the HuggingFace `all-MiniLM-L6-v2` sentence transformer model (running locally — no API key required) and persisted to a ChromaDB collection via [`core/vector_store.py`](core/vector_store.py).

Upon successful indexing, [`core/suggestions.py`](core/suggestions.py) sends the first chunk of the newly uploaded paper to the Groq LLM and generates **three suggested questions**, surfaced as clickable pill buttons beneath the chat history.

### 3.2 Query Submission

The user types a question (or clicks a suggested question). The message is appended to the session's LangChain message history (stored in `st.session_state` via [`memory/session_memory.py`](memory/session_memory.py)) and the full conversation context is formatted into a string for the agent.

### 3.3 Agent Execution

[`agents/research_agent.py`](agents/research_agent.py) exposes a `run_agent()` function that accepts the user input, formatted chat history, and the list of currently indexed sources. It invokes the compiled LangGraph state machine, passing a shared `AgentState` dict through each node. The final state contains the route taken, any retrieved contexts, and the final synthesised answer.

### 3.4 Response Rendering

The app renders the answer with a **colour-coded route badge** (e.g., `RAG · Papers`, `Web Search`, `RAG + Web`) and expanders showing the raw retrieved context from ChromaDB and/or Tavily. If the guardrail node blocked the request, a red shield badge is shown and no retrieval is attempted.

---

## Guardrail System

All user inputs pass through a **two-layer safety gate** implemented in [`guardrails/prompt_guard.py`](guardrails/prompt_guard.py) before any LLM or retrieval call is made.

### Layer 1 — Pattern Matching (instant)

A compiled regular-expression scan checks the input against known prompt-injection and jailbreak keywords, including phrases such as:
- `ignore previous instructions`
- `pretend you are`
- `DAN mode`
- `jailbreak`
- `override`

A hard maximum input length of **4,000 characters** is also enforced. No API call is needed — zero latency, zero cost.

### Layer 2 — LLM Classifier (~0.5 s)

If Layer 1 passes, the input is sent to `llama-3.1-8b-instant` on Groq with a system prompt instructing it to classify whether the message is:
- **(a)** a genuine academic research question
- **(b)** free from subtle injection attempts that pattern matching would miss

The classifier returns a structured verdict with a boolean `is_safe` flag and a `reason` string. Only inputs that pass **both layers** proceed to the router.

> The **Guardrail Demo** tab lets users test arbitrary inputs against both layers in real time, with latency measurements displayed for each layer. Preset attack examples include direct injection, persona hijacking, off-topic requests, and subtle social-engineering attempts.

---

## RAG Implementation

Retrieval-Augmented Generation is the primary knowledge source for queries about uploaded papers.

### 5.1 Document Processing

[`core/pdf_processor.py`](core/pdf_processor.py) uses `pdfplumber` to extract raw text page-by-page. The text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`, with each chunk retaining metadata including the **source filename** and **page number** — surfaced later in source citations.

### 5.2 Vector Store — ChromaDB

[`core/vector_store.py`](core/vector_store.py) wraps ChromaDB, a lightweight, file-persisted vector database that runs entirely in-process with no external server. Embeddings are generated using `all-MiniLM-L6-v2`, producing **384-dimensional vectors** on CPU without a GPU or API key.

The vector store exposes four operations:

| Operation | Description |
|-----------|-------------|
| `add_documents` | Index new chunks |
| `list_indexed_sources` | Show sidebar list |
| `delete_source` | Remove a specific paper's chunks |
| `clear_collection` | Wipe everything on "Clear All Papers" |

### 5.3 Retrieval & Reranking

When the router selects the `rag` or `both` route, the RAG agent performs a **two-stage retrieval**:

1. **Stage 1 — Semantic search**: ChromaDB returns the top-k most similar chunks using cosine similarity.
2. **Stage 2 — Cohere Rerank v3**: Candidate chunks are passed to the Cohere reranking API, which uses a cross-encoder model to re-score each passage against the query. Only the top-ranked passages are passed to the synthesiser.

> This two-stage approach is important because embedding-based retrieval optimises for semantic similarity, whereas cross-encoder reranking captures finer-grained relevance — including negation and specificity.

---

## Agent Architecture

The agent pipeline is implemented as a **LangGraph `StateGraph`** — a directed acyclic graph of nodes where each node is a Python function that reads from and writes to a shared `AgentState` TypedDict. The graph has **six nodes** connected by conditional edges.

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│  GUARDRAIL NODE                             │
│  Layer 1: pattern match (instant)           │
│  Layer 2: LLM classifier (llama-3.1-8b)     │
└───────────────────┬─────────────────────────┘
                    │ PASS
                    ▼
┌─────────────────────────────────────────────┐
│  ROUTER NODE  (llama-3.3-70b)               │
│  Decides: rag / web / both / summarize / chat│
└────┬──────────────┬───────────────┬─────────┘
     │              │               │
     ▼              ▼               ▼
┌─────────┐  ┌───────────┐  ┌────────────┐
│RAG AGENT│  │ WEB AGENT │  │ SUMMARIZER │
│ChromaDB │  │  Tavily   │  │  Groq LLM  │
│+ Cohere │  │  Search   │  │            │
│reranker │  │           │  │            │
└────┬────┘  └─────┬─────┘  └─────┬──────┘
     │             │               │
     └──────┬──────┘               │
            ▼                      │
┌───────────────────────┐          │
│   SYNTHESIZER NODE    │          │
│   llama-3.3-70b       │          │
│   Merges all context  │          │
└───────────┬───────────┘          │
            │                      │
            └──────────┬───────────┘
                       ▼
              Response to User
              (LangSmith trace)
```

### Node Descriptions

| Node | Description |
|------|-------------|
| **Guardrail Node** | Entry point for every query. Runs the two-layer safety check. On failure, sets `guard_passed=False` and short-circuits the graph. On success, passes control to the Router Node. |
| **Router Node** | Uses `llama-3.3-70b-versatile` to classify intent into one of five routes: `rag`, `web`, `both`, `summarize`, or `chat`. The router prompt includes the list of currently indexed source filenames. |
| **RAG Agent Node** | Activated for `rag` and `both` routes. Embeds the query, queries ChromaDB for top-k candidates, and passes them through Cohere Rerank v3. Writes `rag_context` to state. |
| **Web Agent Node** | Activated for `web` and `both` routes. Calls the Tavily search API and returns structured results (title, URL, summary snippets). Writes `web_context` to state. |
| **Summariser Node** | Activated only for the `summarize` route. Retrieves all indexed chunks from ChromaDB and produces a structured summary (background, methods, results, conclusions) using `llama-3.3-70b-versatile`. Writes directly to `final_answer`, bypassing the Synthesiser. |
| **Synthesiser Node** | Final step for `rag`, `web`, and `both` routes. Receives all available contexts, the original query, and conversation history. Produces a grounded answer citing specific passages and noting which source each claim comes from. |

### Session Memory

Conversation history is maintained in `st.session_state` as a list of LangChain `HumanMessage` and `AIMessage` objects ([`memory/session_memory.py`](memory/session_memory.py)). On each turn, the last N messages are serialised into a plain-text string and injected into both the router and synthesiser prompts, enabling **multi-turn follow-up questions**. The **Clear Chat** button resets this list.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit 1.45.0 — tab layout, sidebar uploader, chat interface, status pills |
| **Orchestration** | LangGraph 0.4.1 — StateGraph with conditional edges; LangChain 0.3.25 |
| **LLM Provider** | Groq API — `llama-3.3-70b-versatile` (router, synthesiser, summariser); `llama-3.1-8b-instant` (guardrail classifier) |
| **Embeddings** | `sentence-transformers` `all-MiniLM-L6-v2` via `langchain-huggingface` (local, CPU) |
| **Vector Database** | ChromaDB 0.6.3 — file-persisted, in-process |
| **Reranking** | Cohere Rerank v3 via `cohere` 5.13.12 |
| **Web Search** | Tavily Python SDK 0.5.0 |
| **PDF Parsing** | `pdfplumber` 0.11.5 + `pypdf` 5.4.0 |
| **Observability** | LangSmith 0.3.42 — automatic trace of every graph run |
| **Deployment** | Streamlit Community Cloud (`packages.txt` for system deps; secrets for API keys) |

---

## Project Structure

```
research-paper-agent/
├── app.py                   # Streamlit UI (tabs: Chat, Guardrail Demo, Architecture)
├── requirements.txt         # Python dependencies
├── packages.txt             # System-level packages for Streamlit Cloud
├── agents/
│   ├── __init__.py
│   ├── research_agent.py    # LangGraph StateGraph definition & run_agent()
│   └── tools/
│       ├── __init__.py
│       ├── rag_tool.py      # RAG retrieval + Cohere reranking
│       ├── summarizer_tool.py
│       └── web_search_tool.py
├── config/
│   ├── __init__.py
│   └── settings.py          # API key loading from environment / st.secrets
├── core/
│   ├── __init__.py
│   ├── embeddings.py        # HuggingFace embedding model setup
│   ├── pdf_processor.py     # PDF → LangChain Document chunks
│   ├── vector_store.py      # ChromaDB CRUD operations
│   └── suggestions.py       # LLM-generated suggested questions
├── guardrails/
│   ├── __init__.py
│   └── prompt_guard.py      # Two-layer safety check (_pattern_check, _llm_check)
└── memory/
    ├── __init__.py
    └── session_memory.py    # LangChain message history helpers
```

---

## Key Design Decisions

### Local Embeddings over API Embeddings
Using `all-MiniLM-L6-v2` locally eliminates an API dependency and removes per-token embedding costs, which would accumulate quickly with large PDFs. The model is small enough to run on CPU within Streamlit Cloud's memory limits.

### Two-Stage Retrieval
Combining ChromaDB's approximate nearest-neighbour search with Cohere's cross-encoder reranker gives better precision than either approach alone, at the cost of one extra API call per RAG query.

### Lightweight Guardrail Model
Using `llama-3.1-8b-instant` (rather than the 70b model) for safety classification keeps latency under 0.5 s while still catching nuanced attacks. The 70b model is reserved for reasoning-heavy tasks.

### LangGraph over a Simple Chain
Using a state machine rather than a linear chain allows clean separation of concerns, easy addition of new agent nodes, and conditional routing without deeply nested `if/else` logic.

### Streamlit Session State for Memory
Rather than a database or external memory store, conversation history lives in `st.session_state`. This is appropriate for a single-user demo deployment and avoids additional infrastructure.

---

## Setup & Local Development

### Prerequisites

- Python 3.10+
- API keys for Groq, Cohere, Tavily, and LangSmith

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/research-paper-agent.git
cd research-paper-agent

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.streamlit/secrets.toml` file with your API keys:

```toml
GROQ_API_KEY = "your-groq-api-key"
COHERE_API_KEY = "your-cohere-api-key"
TAVILY_API_KEY = "your-tavily-api-key"
LANGCHAIN_API_KEY = "your-langsmith-api-key"
```

Or set them as environment variables:

```bash
export GROQ_API_KEY="your-groq-api-key"
export COHERE_API_KEY="your-cohere-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
export LANGCHAIN_API_KEY="your-langsmith-api-key"
```

### Run the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Academic Context

| Field | Detail |
|-------|--------|
| **Module** | CST4265 — GenAI and LLMs |
| **Department** | Computer Engineering and Informatics (CEI) |
| **Programme** | MSc Data Science & AI |
| **Student** | Ivan Joseph Thomas (M01093025) |
| **Module Leader** | Dr. Ivan Reznikov |
| **Submission Date** | 23/04/2026 |

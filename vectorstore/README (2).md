# 🤖 Agentic RAG Platform

An **Agentic Retrieval-Augmented Generation (RAG)** system for querying uploaded PDF documents using hybrid retrieval, cross-encoder reranking, corrective query rewriting, relevance grading, and web-search fallback.

Unlike a basic RAG pipeline that performs a single retrieval followed by generation, this project uses a **LangGraph-based agentic workflow** that dynamically decides whether to search uploaded documents, search the web, or answer using the local LLM.

The system also detects poor retrieval, rewrites the query, retries document retrieval, and falls back to web search when the indexed documents cannot answer the question.

---

## ✨ Key Features

* 📄 PDF document ingestion and indexing
* 🔎 Hybrid semantic + keyword retrieval
* 🧠 FAISS vector search
* 🔤 BM25 keyword retrieval
* 🔀 Reciprocal Rank Fusion (RRF)
* 🎯 Cross-encoder reranking
* 🤖 LangGraph-based agentic workflow
* ✅ LLM-based relevance grading
* 🔁 Corrective query rewriting
* 🌐 Tavily web-search fallback
* 🛡️ Grounded answer generation
* ⚡ Batch document relevance grading
* 🚦 Fast deterministic query routing
* 🦙 Local LLM inference using Ollama + Llama 3.2
* 🚀 FastAPI backend
* 💻 Streamlit user interface
* 📊 RAGAS evaluation support
* 🐳 Docker support

---

# 🏗️ Architecture

The platform supports three primary query paths:

```text
                         User Query
                             │
                             ▼
                       Query Router
                    ┌────────┼────────┐
                    │        │        │
                    ▼        ▼        ▼
               Documents    Web    LLM Only
                    │        │        │
                    ▼        │        │
             Hybrid Retrieval
                    │        │        │
                    ▼        │        │
             Relevance Grading
                    │        │        │
              ┌─────┴─────┐  │        │
              │           │  │        │
           Relevant    Irrelevant     │
              │           │           │
              ▼           ▼           │
           Generate   Rewrite Query   │
                          │            │
                          ▼            │
                       Retrieve        │
                          │            │
                          ▼            │
                        Grade          │
                          │            │
                  Maximum retries?     │
                     │        │        │
                    No       Yes       │
                     │        │        │
                  Rewrite     ▼        │
                         Web Search ◄───┘
                              │
                              ▼
                         Generation
                              │
                              ▼
                            Answer
```

---

# 🔎 Retrieval Pipeline

The document retrieval system combines multiple retrieval techniques instead of relying entirely on vector similarity.

```text
                     User Query
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      Semantic Search          Keyword Search
           FAISS                    BM25
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
             Reciprocal Rank Fusion
                         │
                         ▼
               Candidate Documents
                         │
                         ▼
              Cross-Encoder Reranker
                         │
                         ▼
                  Top Contexts
```

### FAISS

FAISS performs dense semantic retrieval using document embeddings.

This allows the system to retrieve passages that are conceptually related to the query even when the exact query keywords are absent.

### BM25

BM25 provides lexical retrieval and is useful when exact terminology, names, or technical phrases appear in the documents.

### Reciprocal Rank Fusion

Results from FAISS and BM25 are combined using **Reciprocal Rank Fusion (RRF)**.

This provides a balance between:

* semantic similarity
* exact keyword matching

### Cross-Encoder Reranking

The fused candidates are reranked using a cross-encoder model.

This performs a deeper query-document relevance comparison before selecting the final context supplied to the RAG agent.

---

# 🤖 Agentic RAG Workflow

The system is implemented as a state machine using **LangGraph**.

The main nodes are:

```text
route_query
retrieve
grade_documents
rewrite_query
web_search
generate
```

Each node performs a specific part of the reasoning/retrieval pipeline.

---

## 1. Query Routing

Queries are classified into three categories.

### Document Query

Example:

```text
According to the uploaded document, how has Shakespeare
been adapted to local cultural contexts?
```

Route:

```text
documents → retrieval → grading → generation
```

### Current / Real-Time Query

Example:

```text
What is NVIDIA's stock price today?
```

Route:

```text
web → Tavily → generation
```

### General Knowledge Query

Example:

```text
Explain gradient descent.
```

Route:

```text
llm_only → generation
```

A lightweight deterministic router is used for obvious document and current-information queries, avoiding an unnecessary LLM routing call and reducing latency.

---

# 🔁 Corrective RAG

A standard RAG system may generate an answer even when retrieval produces irrelevant context.

This project adds a **Corrective RAG loop**.

After retrieval, the retrieved chunks are graded for relevance.

```text
Retrieve
   │
   ▼
Grade Documents
   │
   ├── Relevant ─────────────► Generate
   │
   └── Not Relevant
             │
             ▼
        Rewrite Query
             │
             ▼
          Retrieve
             │
             ▼
           Grade
```

The query can be rewritten up to a configured maximum number of attempts.

If relevant document evidence still cannot be found, the system can trigger web-search fallback.

---

# 🌐 Web Search Fallback

When the indexed documents cannot provide sufficient information, the agent can use **Tavily Search**.

Example:

```text
According to the uploaded Shakespeare document,
what does it say about NVIDIA GPU architecture?
```

The retrieval system recognizes that the document does not contain the requested information.

Instead of hallucinating an answer from unrelated document chunks, the agent can:

1. Attempt corrective retrieval
2. Determine that document evidence is unavailable
3. Search the web
4. Clearly distinguish web information from document information

---

# 🛡️ Grounded Generation

An important improvement made to the pipeline is separation between:

```text
original_query
```

and

```text
query
```

`original_query` always contains the user's actual question.

`query` is a working retrieval query and may be modified during corrective rewriting.

For example:

```text
Original question
       │
       ▼
original_query  ──────────────────────────────┐
                                              │
Working query                                 │
       │                                      │
       ▼                                      │
Rewrite → Retrieve → Rewrite → Retrieve       │
                                              │
                                              ▼
                                      Final Generation
                                              │
                                              ▼
                                Answer original question
```

This prevents the final LLM from accidentally answering a rewritten search query instead of the user's actual question.

---

## Document vs Web Grounding

The generator also distinguishes between:

```text
Document evidence
```

and:

```text
Web evidence
```

If document retrieval fails and web fallback is triggered, the model is explicitly instructed that the web results **did not come from the uploaded document**.

This prevents answers such as:

```text
The Shakespeare document discusses NVIDIA Ampere architecture...
```

when the document contains no such information.

Instead, the expected behavior is:

```text
The uploaded document does not contain information about
NVIDIA GPU architecture.

Web information:
...
```

---

# ⚡ Performance Optimizations

Several optimizations were introduced to reduce local inference latency.

## Batch Relevance Grading

An earlier implementation graded every retrieved chunk using a separate LLM call.

For five chunks:

```text
Chunk 1 → LLM
Chunk 2 → LLM
Chunk 3 → LLM
Chunk 4 → LLM
Chunk 5 → LLM
```

The optimized implementation sends all retrieved chunks to the model in a **single grading request**:

```text
Chunks 1–5
    │
    ▼
Single LLM Call
    │
    ▼
Relevant chunk indices
```

This substantially reduces unnecessary local LLM inference.

---

## Fast Query Routing

The original router used an LLM call to classify every question.

The optimized router handles obvious cases using lightweight deterministic routing.

For example:

```text
"uploaded PDF" → documents
"according to the document" → documents

"today" → web
"latest" → web
"current" → web
```

This removes another unnecessary LLM call from common query paths.

---

# 🧪 Example Results

### Document-grounded query

**Question**

```text
According to the uploaded document, how has Shakespeare
been adapted to local cultural contexts?
```

Example observed execution:

```text
Latency: 5833 ms
Rewrites: 0
Web used: No
```

The system retrieved relevant passages from the uploaded document and generated a document-grounded response.

---

### Corrective RAG query

**Question**

```text
According to the uploaded document about Shakespeare
adaptations, what does it say about NVIDIA GPU architecture?
```

Example observed execution:

```text
Latency: 23962 ms
Rewrites: 2
Web used: Yes
```

The agent correctly determined that the uploaded Shakespeare document did not contain NVIDIA GPU architecture information and avoided presenting unrelated information as document evidence.

> Performance numbers are local example runs and can vary depending on hardware, model state, retrieval workload, and network latency.

---

# 📊 RAG Evaluation

The project includes support for evaluating the RAG pipeline using **RAGAS**.

Evaluation dimensions include:

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall

This allows retrieval and generation quality to be measured systematically rather than relying only on manual inspection.

> The evaluation framework is integrated into the project. Benchmark scores should be generated on a representative evaluation dataset before making quantitative quality claims.

---

# 🧰 Tech Stack

| Component           | Technology             |
| ------------------- | ---------------------- |
| Agent orchestration | LangGraph              |
| LLM                 | Llama 3.2              |
| Local inference     | Ollama                 |
| Vector retrieval    | FAISS                  |
| Keyword retrieval   | BM25                   |
| Fusion              | Reciprocal Rank Fusion |
| Reranking           | Cross-Encoder          |
| Embeddings          | BGE-M3                 |
| Web search          | Tavily                 |
| Backend             | FastAPI                |
| Frontend            | Streamlit              |
| RAG evaluation      | RAGAS                  |
| Containerization    | Docker                 |
| Language            | Python                 |

---

# 📁 Project Structure

```text
Agentic-RAG-Platform/
│
├── app/
│   ├── agent.py
│   ├── api.py
│   └── main.py
│
├── vectorstore/
│   ├── store.py
│   └── index/
│
├── evaluation/
│
├── tests/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

Generated vector indexes are excluded from Git version control and can be rebuilt locally from uploaded documents.

---

# 🚀 Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/shashankkachouli/Agentic-RAG-Platform.git
cd Agentic-RAG-Platform
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Install Ollama and pull the Llama model:

```bash
ollama pull llama3.2
```

Verify:

```bash
ollama list
```

---

## 5. Configure environment variables

Create a `.env` file based on `.env.example`.

```env
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
TAVILY_API_KEY=your_tavily_api_key
```

Never commit the real `.env` file or API keys to GitHub.

---

## 6. Start FastAPI

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

The backend runs at:

```text
http://localhost:8000
```

---

## 7. Start Streamlit

Open another terminal and run:

```bash
streamlit run app/main.py
```

Upload a PDF from the sidebar and click:

```text
Index Document
```

Then start querying the document.

---

# 🔄 End-to-End Flow

```text
Upload PDF
    │
    ▼
Extract & Chunk
    │
    ▼
Generate Embeddings
    │
    ├─────────────► FAISS
    │
    └─────────────► BM25
                       │
                       ▼
User Question → Query Router
                       │
             ┌─────────┼──────────┐
             │         │          │
         Documents     Web     LLM Only
             │         │          │
             ▼         │          │
       Hybrid Search   │          │
             │         │          │
             ▼         │          │
            RRF        │          │
             │         │          │
             ▼         │          │
      Cross-Encoder    │          │
             │         │          │
             ▼         │          │
      Relevance Grade  │          │
             │         │          │
        ┌────┴────┐    │          │
        │         │    │          │
     Relevant   Poor   │          │
        │         │    │          │
        │      Rewrite │          │
        │         │    │          │
        │      Retrieve│          │
        │         │    │          │
        │    Still Poor│          │
        │         │    │          │
        │         ▼    ▼          │
        │      Tavily Search      │
        │             │           │
        └─────────────┼───────────┘
                      ▼
              Grounded Generation
                      │
                      ▼
                    Answer
```

---

# 🎯 Design Goals

The project focuses on four major problems in production-oriented RAG systems:

### 1. Retrieval Quality

Combining semantic and lexical retrieval improves candidate discovery compared with relying on a single retrieval method.

### 2. Retrieval Failure

Corrective RAG allows the system to recognize poor retrieval and retry with a rewritten query.

### 3. Hallucination Reduction

Document-grounded prompts prevent the model from treating unsupported information as document evidence.

### 4. Fresh Information

Web fallback allows the agent to handle information that is missing from indexed documents or requires current sources.

---

# 🔮 Future Improvements

Potential improvements include:

* Streaming LLM responses
* Asynchronous web retrieval
* Persistent document collections
* Multi-document filtering
* Retrieval caching
* Query-result caching
* Faster relevance grading
* Smaller specialized router/grader models
* Expanded RAGAS evaluation dataset
* Retrieval ablation experiments
* Docker deployment
* Cloud deployment
* Conversation-aware retrieval
* Citation verification
* Observability and tracing

---

# 👨‍💻 Author

**Shashank Kachouli**

GitHub: `shashankkachouli`

---

## ⭐ Project Summary

This project demonstrates an end-to-end **Agentic RAG architecture** combining:

**Hybrid Retrieval + RRF + Cross-Encoder Reranking + Corrective RAG + LangGraph + Web Search + Grounded Generation + RAG Evaluation**

The primary goal is to build a RAG system that does more than retrieve and generate: it can **route queries, evaluate retrieval quality, correct failed retrieval, use external information when appropriate, and preserve source grounding throughout the answer-generation process**.

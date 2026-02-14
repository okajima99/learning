## Minimal RAG System (Local LLM + Retrieval)

This project is a **learning-oriented implementation of a Retrieval-Augmented Generation (RAG) system**
built from scratch using a local LLM and a simple retrieval pipeline.

The goal is to understand **how retrieval, prompt construction, and generation are connected**
at the implementation level, rather than to build a production-grade RAG service.

### Motivation
This project was created to:
- Understand the end-to-end flow of a RAG system (ingestion → retrieval → generation)
- Implement document indexing and similarity-based retrieval manually
- Connect a local LLM to external knowledge via a lightweight API

### Scope
- Document ingestion and preprocessing
- Vector-based retrieval logic
- Simple RAG API (query → retrieve → generate)
- Client-side request flow to a local LLM backend

### Notes / Limitations
- Designed for small-scale, local experimentation
- Retrieval quality and latency are not optimized
- Not intended for large datasets or production deployment

This project serves as a **foundation-level experiment** to clarify
why modern RAG systems rely on careful system design, indexing strategies,
and scalable infrastructure in real-world applications.

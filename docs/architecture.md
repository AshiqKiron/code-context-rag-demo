# System Architecture: CodeContext-RAG

## Overview
This system transforms a generic Large Language Model (LLM) into a domain-specific coding assistant by injecting proprietary product context via Retrieval-Augmented Generation (RAG).

## Pipeline Flow

1. **Ingestion Layer**: 
   - Raw code snippets from `data/mock_codebase.py` are ingested.
   - `CharacterTextSplitter` breaks code into semantic chunks (200 tokens) to preserve function logic.

2. **Embedding Layer**: 
   - We use `sentence-transformers/all-MiniLM-L6-v2` (a free, lightweight model) to convert code chunks into vector embeddings.
   - These vectors capture the semantic meaning of the code (e.g., "admin login" vs "user login").

3. **Vector Store**: 
   - `FAISS` (Facebook AI Similarity Search) stores these embeddings for fast, local retrieval.
   - This allows the system to find relevant code snippets in milliseconds without an external database.

4. **Retrieval & Generation**: 
   - When a user asks a question, the query is embedded and searched against the FAISS index.
   - The top-k relevant code snippets are retrieved.
   - These snippets are injected into the LLM's prompt as "Context," forcing the model to follow specific product rules.

5. **Evaluation Gate**: 
   - Outputs are compared against `data/golden_eval_set.json` to measure accuracy and context adherence before being considered "production-ready."

# Code Directory

This directory contains all Python scripts for the Document-Grounded QA project.

## Files

| File | Description |
|------|-------------|
| `simple-baseline.py` | BM25 retrieval baseline - ranks documents by lexical similarity |
| `strong-baseline.py` | LLM-only baseline - uses Llama 3.1 8B via Groq API |
| `evaluation.py` | Evaluates baseline outputs (retrieval, citation, LLM-judge metrics) |
| `extension1.py` | Extension 1: RAG system (BM25 retrieval + LLM answer generation) |
| `extension2.py` | Extension 2: RAG + Cross-Encoder reranking |
| `extension3.py` | Extension 3: RAG + LLM-based citation extraction |
| `extension4.py` | Extension 4: Hybrid retrieval (BM25 + dense embeddings) |
| `extension5.py` | Extension 5: Document chunking (sliding window retrieval) |
| `extension6.py` | Extension 6: Answer verification (retry on failed verification) |
| `evaluate_extension1.py` | Evaluates Extension 1 outputs |
| `evaluate_extension2.py` | Evaluates Extension 2 outputs |
| `evaluate_extension3.py` | Evaluates Extension 3 outputs |
| `evaluate_extension4.py` | Evaluates Extension 4 outputs |
| `evaluate_extension5.py` | Evaluates Extension 5 outputs |
| `evaluate_extension6.py` | Evaluates Extension 6 outputs |
| `config.py` | Path configuration for project directories |

## Prerequisites

Install dependencies using the requirements file:
```bash
pip install -r code/requirements.txt
```

Or install individually:
```bash
pip install rank-bm25 nltk groq sentence-transformers pandas numpy
```

## Environment Setup

Set your Groq API key:
```bash
export GROQ_API_KEY="your-key-here"
```

Or the scripts will prompt you interactively.

## Running the Code

**Important:** All scripts should be run from the **project root directory** (parent of `code/`), not from within the `code/` directory.

### Baselines

```bash
cd /path/to/cis5300_project

# Simple baseline (BM25 retrieval)
python3 code/simple-baseline.py train    # Options: train, dev, test

# Strong baseline (LLM-only)
python3 code/strong-baseline.py train

# Evaluate baselines
python3 code/evaluation.py train
```

### Extension 1: RAG (BM25 + LLM)

Combines BM25 retrieval with LLM answer generation.

```bash
python3 code/extension1.py train
python3 code/evaluate_extension1.py train
```

### Extension 2: Cross-Encoder Reranking

Adds cross-encoder reranking using `ms-marco-MiniLM-L-6-v2`. Retrieves top-30 with BM25, reranks to top-5.

```bash
python3 code/extension2.py train
python3 code/evaluate_extension2.py train
```

### Extension 3: LLM-based Citation Extraction

Replaces word-overlap citation extraction with LLM-based semantic identification of supporting sentences.

```bash
python3 code/extension3.py train
python3 code/evaluate_extension3.py train
```

### Extension 4: Hybrid Retrieval

Combines BM25 (lexical) with sentence-transformer dense embeddings (semantic) for retrieval. Uses `all-MiniLM-L6-v2` for embeddings.

```bash
python3 code/extension4.py train
python3 code/evaluate_extension4.py train
```

### Extension 5: Document Chunking

Operates on document chunks instead of full documents. Documents are split into overlapping windows of sentences (default: 8 sentences, 2 overlap), allowing more precise retrieval when relevant information is localized.

```bash
python3 code/extension5.py train
python3 code/evaluate_extension5.py train
```

### Extension 6: Answer Verification

Builds on Extension 5 (chunking) and adds answer verification. After generating an answer, the system verifies if it's supported by the document. If verification fails, it tries the next-ranked document (up to 3 attempts).

```bash
python3 code/extension6.py train
python3 code/evaluate_extension6.py train
```

## Output Files

All output files are saved to the `output/` directory:

| File Pattern | Description |
|--------------|-------------|
| `{split}_simple_output.json` | Simple baseline predictions |
| `{split}_strong_output.json` | Strong baseline predictions |
| `{split}_extension{N}_output.json` | Extension N predictions |
| `{split}_evaluation.json` | Baseline evaluation results |
| `{split}_extension{N}_evaluation.json` | Extension N evaluation results |

## Script Details

### simple-baseline.py
- Uses BM25Okapi for document retrieval from 787-document corpus
- Extracts top-3 sentences by word overlap as answer snippet
- Outputs retrieved documents and evidence sentence IDs

### strong-baseline.py
- Queries Llama 3.1 8B directly via Groq API (no retrieval)
- Temperature: 0.7, max_tokens: 128
- Does not output retrieved docs or evidence sentences

### extension1.py
- Combines BM25 retrieval with LLM generation
- Retrieves top-5 documents, uses top-1 as context for LLM
- LLM generates 2-4 sentence answer grounded in document
- Word-overlap extraction for citations

### extension2.py
- Adds cross-encoder reranking stage after BM25
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`
- BM25 retrieves top-30, cross-encoder reranks to top-5
- Significantly improves Recall@1 (54% → 79%)

### extension3.py
- Same retrieval as Extension 2 (BM25 + reranking)
- Replaces word-overlap citation with LLM-based extraction
- LLM identifies which sentences semantically support the answer
- Significantly improves Citation F1 (0.29 → 0.58)

### extension4.py
- Hybrid retrieval: combines BM25 scores with dense embedding similarity
- Uses `all-MiniLM-L6-v2` for document embeddings
- Alpha parameter controls BM25 vs semantic weight (default 0.5)
- Best retrieval performance (Recall@1: 83%)

### extension5.py
- Document chunking: splits documents into overlapping sentence windows
- Default: 8 sentences per chunk, 2 sentence overlap
- BM25 + cross-encoder reranking operates on chunks, not full documents
- Uses LLM-based citation extraction (same as Extension 3)
- Deduplicates chunks to return unique parent documents
- Achieves 100% Recall@5 on train split

### extension6.py
- Builds on Extension 5 (chunking + BM25 + cross-encoder reranking)
- Adds answer verification step after answer generation
- LLM checks if answer is SUPPORTED, PARTIAL, or NOT_SUPPORTED by document
- If verification fails, tries next-ranked document (up to 3 attempts)
- Uses LLM-based citation extraction (same as Extension 3)
- Returns verification status and attempt count in output
- Achieves best combined score (0.71) on train split

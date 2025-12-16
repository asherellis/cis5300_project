# CIS 5300 Term Project: A Grounded Q&A Assistant for Intelligence Analysts

## Motivation

Intelligence analysts today face a difficult trade-off. They can comb through lengthy reports and documents manually—a time-consuming process—or query large language models for quick summaries that often hallucinate details and lack verifiable sourcing. What's missing is a middle ground: a tool that delivers fast, trustworthy answers with transparent citations.

This project aims to fill that gap by building a retrieval-augmented generation (RAG) system that provides concise, accurate answers while showing precisely where in the source documents the information originated. This approach provides both the efficiency and accountability that analytical work demands.

## Project Overview

This project implements a document-grounded question answering system that:
1. Retrieves relevant documents from a corpus of 787 documents (news articles + CIA World Factbook)
2. Generates answers grounded in retrieved documents using Llama 3.1 8B (via Groq API)
3. Cites specific sentence IDs as evidence

## Team Members
- Vincent Lin
- Risha Kumar
- Chih Yu Tsai
- Asher Ellis

## Acknowledgments
Thanks to Anirudh Bharadwaj for his guidance as our TA and project advisor!

## Project Structure

```
cis5300_project/
├── report.pdf               # Final report (PDF)
├── README.md                # This file
├── code/                    # All Python scripts
│   ├── simple-baseline.py   # BM25 retrieval baseline
│   ├── strong-baseline.py   # LLM-only baseline (Llama 3.1 8B via Groq)
│   ├── evaluation.py        # Evaluation script for baselines
│   ├── extension1.py        # Extension 1: RAG (BM25 + LLM generation)
│   ├── extension2.py        # Extension 2: RAG + Cross-Encoder Reranking
│   ├── extension3.py        # Extension 3: RAG + LLM-based Citation Extraction
│   ├── extension4.py        # Extension 4: Hybrid Retrieval (BM25 + Dense)
│   ├── extension5.py        # Extension 5: Document Chunking
│   ├── extension6.py        # Extension 6: Answer Verification
│   ├── evaluate_extension*.py
│   ├── config.py            # Path configuration
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Code usage instructions
├── data/                    # Training/dev/test data
│   ├── questions/           # Question-answer pairs (JSONL)
│   ├── corpus/              # Document corpus (787 documents)
│   └── README.md            # Data description
├── output/                  # Model predictions and evaluation results
│   └── README.md            # Output description
├── figures/                 # Figures for report
│   ├── architecture.png     # System architecture diagram
│   ├── results_comparison.png
│   ├── improvement_progression.png
│   └── sample_qa.png
└── docs/                    # Source files
    ├── report.tex           # LaTeX source for report
    ├── refs.bib             # Bibliography references
    └── presentation.pptx    # Presentation slides (editable)
```

## Quick Start

### Prerequisites
```bash
pip install -r code/requirements.txt
```

### Set API Key
```bash
export GROQ_API_KEY="your-groq-api-key"
```

### Run Baselines
```bash
cd cis5300_project

# Simple baseline (BM25 retrieval)
python3 code/simple-baseline.py train

# Strong baseline (LLM-only)
python3 code/strong-baseline.py train

# Evaluate baselines
python3 code/evaluation.py train
```

### Run Extensions
```bash
# Extension 1: RAG (BM25 + LLM)
python3 code/extension1.py train
python3 code/evaluate_extension1.py train

# Extension 2: RAG + Cross-Encoder Reranking
python3 code/extension2.py train
python3 code/evaluate_extension2.py train

# Extension 3: RAG + LLM-based Citation Extraction
python3 code/extension3.py train
python3 code/evaluate_extension3.py train

# Extension 4: Hybrid Retrieval (BM25 + Dense Embeddings)
python3 code/extension4.py train
python3 code/evaluate_extension4.py train

# Extension 5: Document Chunking
python3 code/extension5.py train
python3 code/evaluate_extension5.py train

# Extension 6: Answer Verification
python3 code/extension6.py train
python3 code/evaluate_extension6.py train
```

## Evaluation Metrics

1. **Retrieval Metrics**: Recall@1, Recall@5
2. **Citation Metrics**: Precision, Recall, F1 (exact sentence ID matching)
3. **Answer Quality**: LLM-as-Judge (1-5 rubric score)
4. **Combined Score**: λ × answer_score + (1-λ) × evidence_score (λ=0.5)

## Results Summary (Train Set - 24 Questions)

| Model | Recall@1 | Recall@5 | Citation F1 | Answer Score | Combined |
|-------|----------|----------|-------------|--------------|----------|
| Simple (BM25) | 54.17% | 62.50% | 0.241 | N/A | N/A |
| Strong (LLM) | 0% | 0% | 0% | 2.58/5 | 0.30 |
| Extension 1 (RAG) | 54.17% | 62.50% | 0.241 | 3.12/5 | 0.48 |
| Extension 2 (Rerank) | 79.17% | 91.67% | 0.291 | 3.75/5 | 0.58 |
| Extension 3 (LLM Cite) | 79.17% | 91.67% | 0.576 | 3.83/5 | 0.70 |
| Extension 4 (Hybrid) | 83.33% | 95.83% | 0.528 | 3.79/5 | 0.65 |
| Extension 5 (Chunking) | 75.00% | **100%** | 0.616 | 3.67/5 | 0.69 |
| **Extension 6 (Verify)** | 75.00% | **100%** | **0.632** | **3.88/5** | **0.71** |

### Key Improvements

- **Extension 2** (Cross-encoder reranking): Recall@1 improved 54% → 79% (+46% relative)
- **Extension 3** (LLM citations): Citation F1 improved 0.29 → 0.58 (+98% relative)
- **Extension 4** (Hybrid retrieval): Recall@5 improved 92% → 96%
- **Extension 5** (Document chunking): Recall@5 reached 100% (perfect retrieval in top-5)
- **Extension 6** (Answer verification): Best combined score (0.71), reduces hallucination via retry

## Extensions Implemented

1. **Extension 1: RAG** - Combines BM25 retrieval with LLM answer generation
2. **Extension 2: Cross-Encoder Reranking** - Reranks BM25 results using `ms-marco-MiniLM-L-6-v2`
3. **Extension 3: LLM-based Citation Extraction** - Uses LLM to identify supporting sentences semantically
4. **Extension 4: Hybrid Retrieval** - Combines BM25 (lexical) + sentence-transformers (semantic) embeddings
5. **Extension 5: Document Chunking** - Splits documents into overlapping sentence windows for more precise retrieval
6. **Extension 6: Answer Verification** - Verifies answers are grounded in documents, retries with next doc if not

## Milestones

- [x] Milestone 0: Proposal
- [x] Milestone 1: Literature Review + Data Collection
- [x] Milestone 2: Evaluation + Baselines
- [x] Milestone 3: Extension 1 (RAG)
- [x] Milestone 4: Extensions 2-6 + Final Report + Presentation

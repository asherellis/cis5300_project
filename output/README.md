# Output Directory

This directory contains model predictions and evaluation results.

## Files

### Predictions (JSON format)

| File | Model | Description |
|------|-------|-------------|
| `{split}_simple_output.json` | Simple Baseline | BM25 retrieval predictions |
| `{split}_strong_output.json` | Strong Baseline | LLM-only predictions |
| `{split}_extension1_output.json` | Extension 1 | RAG (BM25 + LLM) predictions |
| `{split}_extension2_output.json` | Extension 2 | RAG + Reranking predictions |
| `{split}_extension3_output.json` | Extension 3 | RAG + LLM Citation predictions |
| `{split}_extension4_output.json` | Extension 4 | Hybrid Retrieval predictions |
| `{split}_extension5_output.json` | Extension 5 | Document Chunking predictions |
| `{split}_extension6_output.json` | Extension 6 | Answer Verification predictions |

### Evaluation Results

| File | Description |
|------|-------------|
| `{split}_evaluation.json` | Combined evaluation for simple & strong baselines |
| `{split}_extension1_evaluation.json` | Extension 1 evaluation |
| `{split}_extension2_evaluation.json` | Extension 2 evaluation |
| `{split}_extension3_evaluation.json` | Extension 3 evaluation |
| `{split}_extension4_evaluation.json` | Extension 4 evaluation |
| `{split}_extension5_evaluation.json` | Extension 5 evaluation |
| `{split}_extension6_evaluation.json` | Extension 6 evaluation |

## Prediction Format

### Simple Baseline Output
```json
{
  "q001": {
    "question": "According to the article...",
    "answer": "The article states that...",
    "retrieved_docs": [
      {"doc_id": "cnn_dailymail__abc123", "score": 12.34, "rank": 1},
      ...
    ],
    "evidence_sentences": ["S13", "S14", "S15"]
  }
}
```

### Strong Baseline Output
```json
{
  "q001": {
    "question": "According to the article...",
    "answer": "The LLM-generated answer..."
  }
}
```

### Extension Outputs
Same format as Simple Baseline with both retrieval results and LLM-generated answers.

## Running Evaluation

From the project root directory:

```bash
# Evaluate baselines
python3 code/evaluation.py train

# Evaluate extensions
python3 code/evaluate_extension1.py train
python3 code/evaluate_extension2.py train
python3 code/evaluate_extension3.py train
python3 code/evaluate_extension4.py train
python3 code/evaluate_extension5.py train
python3 code/evaluate_extension6.py train
```

## Evaluation Metrics

### Retrieval Metrics
- **Recall@1**: % of questions where correct doc is ranked first
- **Recall@5**: % of questions where correct doc is in top 5

### Citation Metrics
- **Precision**: Correct sentences / Predicted sentences
- **Recall**: Correct sentences / Gold sentences
- **F1**: Harmonic mean of precision and recall

### Answer Quality (LLM-as-Judge)
- **Answer Score**: 1-5 rubric score (normalized to 0-1)
- **Evidence Score**: Word overlap between gold and predicted sentences
- **Combined Score**: 0.5 × answer_score + 0.5 × evidence_score

---

## Current Results

### Train Split (24 questions)

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

### Dev Split (3 questions)

| Model | Recall@1 | Recall@5 | Citation F1 | Answer Score | Combined |
|-------|----------|----------|-------------|--------------|----------|
| Simple (BM25) | 66.67% | 66.67% | 0.167 | N/A | N/A |
| Extension 1 (RAG) | 66.67% | 66.67% | 0.167 | 3.33/5 | 0.49 |
| Extension 2 (Rerank) | 66.67% | 66.67% | 0.167 | 4.33/5 | 0.60 |
| Extension 3 (LLM Cite) | 66.67% | 66.67% | 0.553 | 4.00/5 | 0.66 |
| **Extension 4 (Hybrid)** | 66.67% | 66.67% | 0.553 | **4.33/5** | **0.69** |
| Extension 5 (Chunking) | 66.67% | 66.67% | 0.333 | 3.00/5 | 0.44 |
| Extension 6 (Verify) | 66.67% | 66.67% | 0.370 | 3.33/5 | 0.51 |

### Test Split (3 questions)

| Model | Recall@1 | Recall@5 | Citation F1 | Answer Score | Combined |
|-------|----------|----------|-------------|--------------|----------|
| Simple (BM25) | 33.33% | 66.67% | 0.042 | N/A | N/A |
| Extension 1 (RAG) | 33.33% | 66.67% | 0.042 | 2.33/5 | 0.29 |
| Extension 2 (Rerank) | 66.67% | 66.67% | 0.083 | 4.33/5 | 0.54 |
| **Extension 3 (LLM Cite)** | 66.67% | 66.67% | **0.452** | **4.33/5** | **0.71** |
| Extension 4 (Hybrid) | 66.67% | 66.67% | 0.519 | 3.33/5 | 0.61 |
| Extension 5 (Chunking) | 66.67% | **100%** | 0.319 | 3.33/5 | 0.48 |
| Extension 6 (Verify) | 66.67% | **100%** | 0.319 | 2.67/5 | 0.41 |

---

## Key Findings

1. **Extension 2 (Reranking)** dramatically improves retrieval: Recall@1 jumps from 54% to 79% on train (+46%)
2. **Extension 3 (LLM Citations)** dramatically improves citation quality: F1 improves from 0.29 to 0.58 (+98%)
3. **Extension 4 (Hybrid)** achieves best single-doc retrieval: 83% Recall@1, 96% Recall@5
4. **Extension 5 (Chunking)** achieves perfect Recall@5 (100%) on train and test splits
5. **Extension 6 (Verification)** achieves best combined score (0.71 on train) and reduces hallucination via retry
6. Extensions 5 and 6 achieve 100% Recall@5 on test split, ensuring correct document is always in top-5

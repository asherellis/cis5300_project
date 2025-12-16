# Data Directory

This directory contains all data for the Document-Grounded QA project.

## Directory Structure

```
data/
├── questions/           # Question-answer pairs
│   ├── train_final_jsonl.txt      # Training set (24 questions)
│   ├── dev_final_jsonl.txt        # Development set (3 questions)
│   ├── test_final_jsonl.txt       # Test set (3 questions)
│   └── all_examples_final.txt     # All examples combined
└── corpus/                     # Document corpus (787 documents)
    ├── gold/                      # Gold documents (13 .txt files with JSON content)
    ├── abisee/                    # CNN/DailyMail corpus (300 files)
    ├── hugginglearners/           # Russia-Ukraine corpus (217 files)
    └── cia-world-factbook/        # CIA World Factbook corpus (257 files)
```

## Corpus Organization

The corpus consists of **gold documents** and **corpus documents**:

### Gold Documents (13 .txt files containing JSON)
These are the documents directly referenced by the 30 QA pairs. They are stored in the `gold/` subfolder as `.txt` files containing JSON-structured data with sentence-level annotations for citation extraction.

| Source | Count | Example Filename |
|--------|-------|------------------|
| CNN/DailyMail | 3 | `gold/cnn_dailymail__506e1baad13bc8b50b0f1db98518da52aea1c40c.txt` |
| Russia-Ukraine | 5 | `gold/russia-ukraine-conflict_2022-06-11T12_01_17Z.txt` |
| CIA World Factbook | 5 | `gold/the-world-factbook-by-cia__Korea_North_history.txt` |

### Corpus Documents (774 files, plain text)
These are additional documents from each source that form the retrieval corpus. The system must find the correct gold document among these.

| Subfolder | Source | Count |
|-----------|--------|-------|
| `abisee/` | CNN/DailyMail (HuggingFace) | 300 |
| `hugginglearners/` | Russia-Ukraine conflict articles | 217 |
| `cia-world-factbook/` | CIA World Factbook entries | 257 |

## Data Statistics

| Split | Questions | Unique Gold Docs |
|-------|-----------|------------------|
| Train | 24 | 13 |
| Dev | 3 | 3 |
| Test | 3 | 3 |
| **Total** | **30** | **13** |

Note: Multiple questions can reference the same gold document.

## Question Format (JSONL)

Each line in the JSONL files is a JSON object with the following fields:

```json
{
  "doc_id": "gold/cnn_dailymail__506e1baad13bc8b50b0f1db98518da52aea1c40c",
  "source_dataset": "cnn_dailymail",
  "source": "506e1baad13bc8b50b0f1db98518da52aea1c40c",
  "question_type": "specific_fact_retrieval",
  "question": "According to the article, what controversies...",
  "answer": "The article states that Griffin Bell's...",
  "evidence_sentences": ["S13", "S14", "S15"],
  "edge_case_type": "precise_fact_retrieval",
  "rubric": {
    "description": "Rubric assessing retrieval of specific controversies...",
    "scale": {
      "1": "Vague, incorrect, or fabricated...",
      "2": "Includes one fragment...",
      "3": "Mentions at least one...",
      "4": "Accurately mentions both...",
      "5": "Fully correct..."
    }
  }
}
```

## Document Formats

### Gold Documents (JSON)
Gold documents use structured JSON format with sentence IDs for citation:

```json
{
  "doc_id": "gold/the-world-factbook-by-cia__Korea_North_history",
  "sentences": [
    {"sid": "S1", "text": "First sentence of the document."},
    {"sid": "S2", "text": "Second sentence of the document."}
  ]
}
```

### Corpus Documents (Plain Text)
Corpus documents in subfolders are plain text files used for retrieval.

## Question Types

| Type | Description |
|------|-------------|
| `specific_fact_retrieval` | Retrieve specific facts from the document |
| `broad_analytical_overview` | Analyze multiple aspects of a topic |
| `precise_fact_retrieval` | Extract exact numerical/procedural facts |
| `unanswerable_from_source` | Questions that cannot be answered from the document |
| `analysis` | Analytical questions requiring synthesis |

## Data Sources

- **CNN/DailyMail**: HuggingFace `abisee/cnn_dailymail` dataset
- **Russia-Ukraine Conflict**: HuggingFace `hugginglearners/russia-ukraine-conflict-articles`
- **CIA World Factbook**: Processed entries from CIA World Factbook

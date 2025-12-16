import argparse
import sys
import json
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

_punkt_param = PunktParameters()
_punkt_param.abbrev_types = set([
    'dr', 'vs', 'mr', 'mrs', 'ms', 'prof', 'inc', 'u.s', 'e.g', 'i.e', 
    'etc', 'fig', 'al', 'gen', 'col', 'jr', 'sr', 'rev', 'hon', 'esq',
    'ltd', 'co', 'corp', 'approx', 'appt', 'dept', 'est', 'min', 'max',
    'a.m', 'p.m', 'e.t', 'no', 'pp', 'op', 'vol', 'ed', 'eds', 'st'
])
custom_sent_tokenize = PunktSentenceTokenizer(_punkt_param).tokenize

DEFAULT_THRESHOLD = 0.0


def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_documents(doc_dir):
    doc_texts = []
    doc_ids = []
    doc_structures = []

    import os

    if not os.path.exists(doc_dir):
        print(
            f"Warning: Document directory {doc_dir} not found. Using placeholder.",
            file=sys.stderr,
        )
        return doc_texts, doc_ids, doc_structures

    skip_files = {'train_final.txt', 'dev_final.txt', 'test_final.txt',
                   'train_final_jsonl.txt', 'dev_final_jsonl.txt', 'test_final_jsonl.txt',
                   'all_examples_final.txt'}
    
    for root, _, files in os.walk(doc_dir):
        for filename in files:
            if filename in skip_files:
                continue
            if filename.endswith(".txt") or filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, doc_dir)
                doc_id = rel_path.replace(os.sep, "/")
                doc_id = doc_id.replace(".txt", "").replace(".json", "")
                try:
                    structure = None
                    if filename.endswith(".json"):
                        with open(filepath, "r") as f:
                            doc_data = json.load(f)
                            text = doc_data.get("text", "") or doc_data.get(
                                "content", ""
                            )
                            if 'sentences' in doc_data:
                                structure = doc_data  
                    else:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        try:
                            doc_data = json.loads(content)
                            if 'sentences' in doc_data:
                                sentences = [s.get('text', '') for s in doc_data['sentences'] if s.get('text')]
                                text = ' '.join(sentences)
                                structure = doc_data 
                            else:
                                text = content
                        except json.JSONDecodeError:
                            text = content

                    if text:
                        doc_texts.append(text)
                        doc_ids.append(doc_id)
                        doc_structures.append(structure)
                except Exception as e:
                    print(
                        f"Warning: Could not load {rel_path}: {e}",
                        file=sys.stderr,
                    )

    return doc_texts, doc_ids, doc_structures


def tokenize(text):
    return word_tokenize(text.lower())


def build_bm25_index(doc_texts):
    tokenized_corpus = [tokenize(text) for text in doc_texts]
    return BM25Okapi(tokenized_corpus)


def extract_snippet(text, question, max_chars=500):
    sentences = custom_sent_tokenize(text)
    if not sentences:
        snippet = text[:max_chars]
        return snippet, []

    query_tokens = tokenize(question)
    sentence_scores = []

    for i, sent in enumerate(sentences):
        sent_tokens = tokenize(sent)
        if sent_tokens:
            matches = sum(1 for token in query_tokens if token in sent_tokens)
            sentence_scores.append((i, sent, matches))

    if not sentence_scores:
        snippet = " ".join(sentences[:2])[:max_chars]
        return snippet, list(range(min(2, len(sentences))))

    sentence_scores.sort(key=lambda x: x[2], reverse=True)
    num_sentences = min(3, len(sentence_scores))
    top_sentences_info = sentence_scores[:num_sentences]
    top_sentences = [sent for _, sent, _ in top_sentences_info]
    top_sentence_indices = [idx for idx, _, _ in top_sentences_info]
    snippet = " ".join(top_sentences)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    return snippet, sorted(top_sentence_indices)


def extract_sentence_ids(question, doc_structure, doc_text):
    """
    Extract sentence IDs from document based on query relevance.
    For structured JSON: uses existing sentence IDs (e.g., 'S1', 'S2')
    For plain text: generates sentence IDs (e.g., 'S1', 'S2', 'S3')
    Returns list of sentence IDs.
    """
    query_tokens = tokenize(question)
    
    # Handle documents with sentence IDs in gold set
    if doc_structure and 'sentences' in doc_structure:
        sentences = doc_structure['sentences']
        if not sentences:
            return []
        
        sentence_scores = []
        for sent_obj in sentences:
            sid = sent_obj.get('sid', '')
            text = sent_obj.get('text', '')
            if not sid or not text:
                continue
            
            sent_tokens = tokenize(text)
            if sent_tokens:
                matches = sum(1 for token in query_tokens if token in sent_tokens)
                sentence_scores.append((sid, matches))
        
        if not sentence_scores:
            return [s.get('sid') for s in sentences[:3] if s.get('sid')]
        
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        num_sentences = min(3, len(sentence_scores))
        return [sid for sid, _ in sentence_scores[:num_sentences]]
    
    # Handle remaining documents - generate sentence IDs
    else:
        sentences = custom_sent_tokenize(doc_text)
        if not sentences:
            return []
        
        sentence_scores = []
        for i, sent in enumerate(sentences):
            sent_tokens = tokenize(sent)
            if sent_tokens:
                matches = sum(1 for token in query_tokens if token in sent_tokens)
                sentence_scores.append((f"S{i+1}", matches))
        
        if not sentence_scores:
            return [f"S{i+1}" for i in range(min(3, len(sentences)))]
        
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        num_sentences = min(3, len(sentence_scores))
        return [sid for sid, _ in sentence_scores[:num_sentences]]


def simple_baseline_predict(
    question,
    bm25_index,
    doc_texts,
    doc_ids,
    doc_structures,
    k=5,
    score_threshold=DEFAULT_THRESHOLD,
):
    if not doc_texts:
        return {
            "answer": "I cannot answer this question as no documents are available in the corpus.",
            "citations": [],
            "evidence_sentences": [],
        }

    query_tokens = tokenize(question)
    scores = bm25_index.get_scores(query_tokens)

    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:k]

    top_score = scores[top_indices[0]] if top_indices else 0.0

    if top_score < score_threshold:
        return {
            "answer": "I cannot answer this question based on the available documents. No sufficiently relevant documents were found.",
            "citations": [],
            "evidence_sentences": [],
        }

    answer_entries = []
    aggregated_answer = None
    evidence_sentences = [] 

    for rank, idx in enumerate(top_indices):
        doc_id = doc_ids[idx]
        doc_text = doc_texts[idx]
        doc_structure = doc_structures[idx] if idx < len(doc_structures) else None
        
        snippet, sentence_ids = extract_snippet(doc_text, question)
        if snippet:
            entry_answer = snippet
        else:
            entry_answer = doc_text[:500]
        citation = {
            "doc_id": doc_id,
            "score": float(scores[idx]),
        }
        if sentence_ids:
            citation["sentence_ids"] = sentence_ids
        answer_entries.append({"answer": entry_answer, "citation": citation})
        
        if rank == 0:
            aggregated_answer = entry_answer
            evidence_sentences = extract_sentence_ids(question, doc_structure, doc_text)

    retrieved_docs = [
        {
            "doc_id": doc_ids[idx],
            "score": float(scores[idx]),
            "rank": rank + 1
        }
        for rank, idx in enumerate(top_indices)
    ]
    
    return {
        "answer": aggregated_answer or "",
        "answers": answer_entries,
        "retrieved_docs": retrieved_docs,
        "evidence_sentences": evidence_sentences,
    }


parser = argparse.ArgumentParser(
    description="Run the simple BM25 retrieval baseline."
)
parser.add_argument(
    "split",
    nargs="?",
    default="train",
    choices=["train", "dev", "test"],
    help="Which split to run (train, dev, or test). Default: train",
)
parser.add_argument(
    "--score-threshold",
    type=float,
    default=DEFAULT_THRESHOLD,
    help="BM25 score floor for accepting a document.",
)
args = parser.parse_args()
split = args.split
input_file = f"data/questions/{split}_final_jsonl.txt"
doc_dir = "data/corpus"
output_file = f"output/{split}_simple_output.json"
score_threshold = args.score_threshold
questions = load_jsonl(input_file)
doc_texts, doc_ids, doc_structures = load_documents(doc_dir)
bm25_index = build_bm25_index(doc_texts)
predictions = {}
print("\n" + "=" * 60 + "\n")
for idx, item in enumerate(questions):
    question = item.get("question", "")
    if not question:
        continue
    question_id = item.get("question_id", f"q{idx+1:03d}")
    question_split = item.get("split", "unknown")
    question_type = item.get("question_type", "")
    prediction = simple_baseline_predict(
        question,
        bm25_index,
        doc_texts,
        doc_ids,
        doc_structures,
        k=5,
        score_threshold=score_threshold,
    )
    predictions[question_id] = {
        "question": question,
        "answer": prediction.get("answer", ""),
        "answers": prediction.get("answers", []),
        "retrieved_docs": prediction.get("retrieved_docs", []),
        "evidence_sentences": prediction.get("evidence_sentences", []),
    }
    print("Q:", question)
    print("A:", prediction.get("answer", ""))
    print("-" * 40)
with open(output_file, "w") as f:
    json.dump(predictions, f, indent=2)

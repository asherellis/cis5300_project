import argparse
import sys
import json
import os
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from groq import Groq

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

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("="*60)
    print("Submit Groq API Key")
    print("\n1. Go to: https://console.groq.com/keys")
    print("2. Click 'Create API Key'")
    print("3. Copy key and paste it below\n")
    groq_api_key = input("Enter your Groq API key: ").strip()
    if not groq_api_key:
        print("No API key provided. Exiting.")
        exit(1)

client = Groq(api_key=groq_api_key)
model_name = "llama-3.1-8b-instant"

def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_documents(doc_dir):
    doc_texts = []
    doc_ids = []
    doc_structures = []
    import os

    if not os.path.exists(doc_dir):
        print(f"Warning: Document directory {doc_dir} not found. Using placeholder.", file=sys.stderr)
        return doc_texts, doc_ids, doc_structures

    skip_files = {'train_final.txt', 'dev_final.txt', 'test_final.txt',
                   'train_final_jsonl.txt', 'dev_final_jsonl.txt', 'test_final_jsonl.txt',
                   'all_examples_final.txt'}
    
    print("Loading documents from corpus...", file=sys.stderr)
    file_count = 0
    loaded_count = 0
    
    for root, _, files in os.walk(doc_dir):
        for filename in files:
            if filename in skip_files:
                continue
            if filename.endswith(".txt") or filename.endswith(".json"):
                file_count += 1
                if file_count % 100 == 0:
                    print(f"  Processed {file_count} files, loaded {loaded_count} documents...", file=sys.stderr)
                
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, doc_dir)
                doc_id = rel_path.replace(os.sep, "/")
                doc_id = doc_id.replace(".txt", "").replace(".json", "")
                try:
                    structure = None
                    if filename.endswith(".json"):
                        with open(filepath, "r") as f:
                            doc_data = json.load(f)
                            text = doc_data.get("text", "") or doc_data.get("content", "")
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
                        doc_structures.append(structure)  # None if not structured
                        loaded_count += 1
                except Exception as e:
                    print(f"Warning: Could not load {rel_path}: {e}", file=sys.stderr)

    print(f"Loaded {loaded_count} documents from {file_count} files.", file=sys.stderr)
    return doc_texts, doc_ids, doc_structures

def tokenize(text):
    return word_tokenize(text.lower())

def build_bm25_index(doc_texts):
    tokenized_corpus = [tokenize(text) for text in doc_texts]
    return BM25Okapi(tokenized_corpus)

def extract_sentence_ids(question, doc_structure, doc_text):
    """
    Extract sentence IDs from document based on query relevance.
    For structured JSON: uses existing sentence IDs (e.g., 'S1', 'S2')
    For plain text: generates sentence IDs (e.g., 'S1', 'S2', 'S3')
    Returns list of sentence IDs.
    """
    query_tokens = tokenize(question)
    
    # Handle structured JSON documents with sentence annotations
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
    
    # Handle plain text documents - generate sentence IDs
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

def truncate_text(text, max_tokens=1500):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens]) + "..."

def build_rag_prompt(question, doc_text):
    doc_text = truncate_text(doc_text, max_tokens=1500)
    prompt = f"""You are a document-grounded question answering system. Answer the question based ONLY on the provided document.

Document:
{doc_text}

Question: {question}

Instructions:
- Answer the question in two to four complete sentences based on the document.
- If the document does not contain enough information to answer the question, say "The document does not provide sufficient information to answer this question."
- Do not include any notes, disclaimers, bullet points, or follow-up offers.
- Give only the answer.

Answer:"""
    return prompt

def query_groq(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.7,
            max_tokens=256,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return "Error generating answer"

def rag_predict(
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
            "retrieved_docs": [],
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
            "retrieved_docs": [],
            "evidence_sentences": [],
        }

    retrieved_docs = [
        {
            "doc_id": doc_ids[idx],
            "score": float(scores[idx]),
            "rank": rank + 1,
            "text": doc_texts[idx],
            "structure": doc_structures[idx] if idx < len(doc_structures) else None
        }
        for rank, idx in enumerate(top_indices)
    ]

    # Generate answer using LLM from top-ranked document
    top_doc_text = retrieved_docs[0]['text']
    prompt = build_rag_prompt(question, top_doc_text)
    answer = query_groq(prompt)

    # Extract citation sentences from top-ranked document
    top_doc = retrieved_docs[0]
    evidence_sentences = extract_sentence_ids(question, top_doc["structure"], top_doc["text"])

    return {
        "answer": answer,
        "retrieved_docs": [
            {
                "doc_id": doc["doc_id"],
                "score": doc["score"],
                "rank": doc["rank"]
            }
            for doc in retrieved_docs
        ],
        "evidence_sentences": evidence_sentences,
    }

parser = argparse.ArgumentParser(
    description="Run Extension 1: RAG (BM25 retrieval + LLM generation)."
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
output_file = f"output/{split}_extension1_output.json"
score_threshold = args.score_threshold

questions = load_jsonl(input_file)
print(f"Loaded {len(questions)} questions.", file=sys.stderr)

doc_texts, doc_ids, doc_structures = load_documents(doc_dir)
print(f"Building BM25 index...", file=sys.stderr)
bm25_index = build_bm25_index(doc_texts)
print(f"BM25 index built. Ready to process questions.", file=sys.stderr)

predictions = {}
print("\n" + "=" * 60 + "\n")
print(f"Running Extension 1: RAG Baseline")
print(f"Using top-1 document for answer generation")
print(f"Processing {len(questions)} questions...")
print("=" * 60 + "\n")

for idx, item in enumerate(questions):
    # Progress indicator
    if (idx + 1) % 5 == 0:
        print(f"Progress: {idx + 1}/{len(questions)} questions processed...", file=sys.stderr)
    question = item.get("question", "")
    if not question:
        continue
    question_id = item.get("question_id", f"q{idx+1:03d}")
    
    prediction = rag_predict(
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
        "retrieved_docs": prediction.get("retrieved_docs", []),
        "evidence_sentences": prediction.get("evidence_sentences", []),
    }
    
    print(f"Q{idx+1}:", question[:80] + "..." if len(question) > 80 else question)
    print(f"A:", prediction.get("answer", "")[:100] + "..." if len(prediction.get("answer", "")) > 100 else prediction.get("answer", ""))
    print("-" * 40)

with open(output_file, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\nPredictions saved to: {output_file}")


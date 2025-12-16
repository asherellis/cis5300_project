import argparse
import sys
import json
import os
import re
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from groq import Groq
from sentence_transformers import CrossEncoder
import numpy as np

cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_model = CrossEncoder(cross_encoder_name)


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
MAX_VERIFICATION_ATTEMPTS = 3

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
                        doc_structures.append(structure)
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

def BM25_topN(question, bm25_index, doc_texts, N=30):
    tokenized_question = tokenize(question)
    scores = bm25_index.get_scores(tokenized_question)
    top_ids = np.argsort(scores)[::-1][:N]
    top_texts = [doc_texts[i] for i in top_ids]
    return top_ids, top_texts, scores

def rerank_encoder(question, top_ids, top_texts, k=5):
    input_pairs = [(question, t) for t in top_texts]
    new_scores = rerank_model.predict(input_pairs)
    sorted_by_reranking = list(zip(top_ids, top_texts, new_scores))
    sorted_by_reranking.sort(key=lambda x: x[2], reverse=True)
    top_retrieved = sorted_by_reranking[:k]
    reranked_ids = [ids for ids, _, _ in top_retrieved]
    reranked_scores = [float(score) for _, _, score in top_retrieved]
    return reranked_ids, reranked_scores


def get_sentences_with_ids(doc_structure, doc_text):
    if doc_structure and 'sentences' in doc_structure:
        collect = []
        for sentence in doc_structure['sentences']:
            sid, text = sentence.get('sid'), sentence.get('text')
            if sid and text:
                collect.append((sid, text))
        return collect
    sents = custom_sent_tokenize(doc_text)
    return [(f"S{i+1}", sent) for i, sent in enumerate(sents) if sent.strip()]


def chunk_sentences(sentence_with_ids, window=8, overlap=2):
    if not sentence_with_ids:
        return []
    chunks = []
    step_size = max(1, window - overlap)
    for i in range(0, len(sentence_with_ids), step_size):
        elem = sentence_with_ids[i:i+window]
        if elem:
            chunks.append(elem)
        if i + window >= len(sentence_with_ids):
            break
    return chunks

def load_documents_chunked(doc_dir, window=8, overlap=2):
    doc_texts, doc_ids, doc_structures = load_documents(doc_dir)
    chunk_texts = []
    chunk_ids = []
    chunk_sent_lists = []

    for dt, did, ds in zip(doc_texts, doc_ids, doc_structures):
        sentence_list = get_sentences_with_ids(ds, dt)
        for c in chunk_sentences(sentence_list, window=window, overlap=overlap):
            chunk_texts.append(" ".join(t for _, t in c))
            chunk_ids.append(did)
            chunk_sent_lists.append(c)

    return chunk_texts, chunk_ids, chunk_sent_lists


def extract_sentence_ids_with_chunk(question, sent_list, top_k=3):
    """Fallback word-overlap method for citation extraction."""
    q = tokenize(question)
    scored = []
    for sid, text in sent_list:
        if not sid or not text:
            continue
        tokenized_text = tokenize(text)
        matches = sum(1 for tok in q if tok in tokenized_text)
        scored.append((sid, matches))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:top_k]]


def extract_sentence_ids_llm_from_chunk(question, answer, sent_list):
    """
    Use LLM to identify which sentences from the chunk best support the answer.
    This replaces word-overlap with semantic understanding.
    """
    if not sent_list:
        return []

    # Build the sentence list for the prompt
    sentence_list_str = "\n".join([f"{sid}: {text}" for sid, text in sent_list])

    prompt = f"""You are a citation extraction system. Given a question, an answer, and a list of sentences from a source document, identify which sentences contain information that supports the answer.

QUESTION:
{question}

ANSWER:
{answer}

DOCUMENT SENTENCES:
{sentence_list_str}

INSTRUCTIONS:
1. Read the answer carefully and identify the key claims/facts it contains.
2. Find sentences from the document that directly support these claims.
3. Return ONLY the sentence IDs (e.g., S1, S5, S12) that contain supporting evidence.
4. Return 1-5 sentence IDs, prioritizing the most relevant ones.
5. If no sentences support the answer, return "NONE".

OUTPUT FORMAT:
Return only the sentence IDs separated by commas (e.g., "S3, S7, S12").
Do not include any explanation or other text.

SUPPORTING SENTENCE IDs:"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.0,
            max_tokens=50,
        )
        result = response.choices[0].message.content.strip()

        # Parse the response to extract sentence IDs
        if result.upper() == "NONE" or not result:
            return []

        # Extract sentence IDs using regex
        sentence_ids = re.findall(r'S\d+', result, re.IGNORECASE)
        sentence_ids = [sid.upper() for sid in sentence_ids]

        # Validate that extracted IDs exist in our sentence list
        valid_ids = {sid for sid, _ in sent_list}
        sentence_ids = [sid for sid in sentence_ids if sid in valid_ids]

        # Return up to 5 unique IDs
        seen = set()
        unique_ids = []
        for sid in sentence_ids:
            if sid not in seen:
                seen.add(sid)
                unique_ids.append(sid)
                if len(unique_ids) >= 5:
                    break

        # Fallback to word-overlap if LLM returned nothing useful
        if not unique_ids:
            return extract_sentence_ids_with_chunk(question, sent_list, top_k=3)

        return unique_ids

    except Exception as e:
        print(f"LLM citation extraction error: {e}", file=sys.stderr)
        return extract_sentence_ids_with_chunk(question, sent_list, top_k=3)


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


def verify_answer(question, answer, doc_text):
    """
    Verify whether the generated answer is supported by the document.
    Returns: (status, rationale) where status is 'SUPPORTED', 'NOT_SUPPORTED', or 'PARTIAL'
    """
    doc_text = truncate_text(doc_text, max_tokens=1200)

    prompt = f"""You are a fact-checking system. Determine whether the given answer is supported by the document.

DOCUMENT:
{doc_text}

QUESTION: {question}

ANSWER TO VERIFY: {answer}

INSTRUCTIONS:
1. Check if each claim in the answer can be found in or directly inferred from the document.
2. Return one of these verdicts:
   - SUPPORTED: All claims in the answer are backed by the document
   - NOT_SUPPORTED: The answer contains claims that are not in the document or contradicts it
   - PARTIAL: Some claims are supported but others are not

OUTPUT FORMAT:
Return ONLY a single word: SUPPORTED, NOT_SUPPORTED, or PARTIAL
Do not include any explanation.

VERDICT:"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.0,
            max_tokens=20,
        )
        result = response.choices[0].message.content.strip().upper()

        # Parse the response
        if "SUPPORTED" in result and "NOT" not in result:
            return "SUPPORTED"
        elif "NOT_SUPPORTED" in result or "NOT SUPPORTED" in result:
            return "NOT_SUPPORTED"
        elif "PARTIAL" in result:
            return "PARTIAL"
        else:
            # Default to SUPPORTED if unclear (conservative)
            return "SUPPORTED"

    except Exception as e:
        print(f"Verification error: {e}", file=sys.stderr)
        return "SUPPORTED"  # Default to accepting on error


def rag_predict_with_verification(
    question,
    bm25_index,
    chunk_texts,
    chunk_ids,
    chunk_sent_lists,
    k=5,
    score_threshold=DEFAULT_THRESHOLD,
    max_attempts=MAX_VERIFICATION_ATTEMPTS,
):
    """RAG prediction with answer verification and retry logic."""
    if not chunk_texts:
        return {
            "answer": "I cannot answer this question as no documents are available in the corpus.",
            "retrieved_docs": [],
            "evidence_sentences": [],
            "verification_status": "N/A",
            "verification_attempts": 0,
        }

    N = max(k * 10, 50)
    cand_id, cand_text, scores = BM25_topN(question, bm25_index, chunk_texts, N)

    if len(cand_id) == 0:
        return {
            "answer": "I cannot answer this question as no documents are available in the corpus.",
            "retrieved_docs": [],
            "evidence_sentences": [],
            "verification_status": "N/A",
            "verification_attempts": 0,
        }

    top_score = scores[cand_id[0]] if len(cand_id) > 0 else 0.0

    if top_score < score_threshold:
        return {
            "answer": "I cannot answer this question based on the available documents. No sufficiently relevant documents were found.",
            "retrieved_docs": [],
            "evidence_sentences": [],
            "verification_status": "N/A",
            "verification_attempts": 0,
        }

    rerank_pool = min(len(cand_id), 30)
    reranked_ids, reranked_scores = rerank_encoder(question, cand_id, cand_text, k=rerank_pool)

    # Build list of unique documents (deduplicated by doc_id)
    chosen = []
    seen_docs = set()
    for r, chunk_idx in enumerate(reranked_ids):
        did = chunk_ids[chunk_idx]
        if did in seen_docs:
            continue
        seen_docs.add(did)
        chosen.append((chunk_idx, float(reranked_scores[r])))
        if len(chosen) >= k:
            break

    if not chosen:
        return {
            "answer": "I cannot answer this question based on the available documents. No sufficiently relevant documents were found.",
            "retrieved_docs": [],
            "evidence_sentences": [],
            "verification_status": "N/A",
            "verification_attempts": 0,
        }

    retrieved_docs = [
        {
            "doc_id": chunk_ids[chunk_idx],
            "score": score,
            "rank": rank + 1,
        }
        for rank, (chunk_idx, score) in enumerate(chosen)
    ]

    # Try up to max_attempts documents with verification
    best_answer = None
    best_status = None
    best_chunk_idx = None
    best_doc_index = 0  # Track which position in retrieved_docs was used
    attempts = 0

    for attempt_idx, (chunk_idx, score) in enumerate(chosen[:max_attempts]):
        attempts += 1
        chunk_text = chunk_texts[chunk_idx]

        # Generate answer
        prompt = build_rag_prompt(question, chunk_text)
        answer = query_groq(prompt)

        # Skip verification for "cannot answer" responses
        if "does not provide sufficient information" in answer.lower() or "cannot answer" in answer.lower():
            if best_answer is None:
                best_answer = answer
                best_status = "N/A"
                best_chunk_idx = chunk_idx
                best_doc_index = attempt_idx
            continue

        # Verify answer
        status = verify_answer(question, answer, chunk_text)

        # If supported, use this answer
        if status == "SUPPORTED":
            best_answer = answer
            best_status = status
            best_chunk_idx = chunk_idx
            best_doc_index = attempt_idx
            break

        # Track best partial answer
        if best_answer is None or (best_status == "NOT_SUPPORTED" and status == "PARTIAL"):
            best_answer = answer
            best_status = status
            best_chunk_idx = chunk_idx
            best_doc_index = attempt_idx

    # Use best answer found
    if best_answer is None:
        best_answer = "I cannot provide a verified answer based on the available documents."
        best_status = "NOT_SUPPORTED"
        best_chunk_idx = chosen[0][0]
        best_doc_index = 0

    # Extract citations using LLM-based method (like Extension 3)
    evidence_sentences = extract_sentence_ids_llm_from_chunk(
        question, best_answer, chunk_sent_lists[best_chunk_idx]
    )

    return {
        "answer": best_answer,
        "retrieved_docs": retrieved_docs,
        "evidence_sentences": evidence_sentences,
        "verification_status": best_status,
        "verification_attempts": attempts,
        "answer_doc_index": best_doc_index,  # Index in retrieved_docs of doc used for answer
    }


parser = argparse.ArgumentParser(
    description="Run Extension 6: RAG with Document Chunking + Answer Verification."
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
parser.add_argument("--chunk-window", type=int, default=8)
parser.add_argument("--chunk-overlap", type=int, default=2)
parser.add_argument("--max-attempts", type=int, default=MAX_VERIFICATION_ATTEMPTS,
                    help="Maximum verification attempts before accepting best answer.")

args = parser.parse_args()

split = args.split
input_file = f"data/questions/{split}_final_jsonl.txt"
doc_dir = "data/corpus"
output_file = f"output/{split}_extension6_output.json"
score_threshold = args.score_threshold

questions = load_jsonl(input_file)
print(f"Loaded {len(questions)} questions.", file=sys.stderr)

chunk_texts, chunk_ids, chunk_sent_lists = load_documents_chunked(
    doc_dir, window=args.chunk_window, overlap=args.chunk_overlap
)
print(f"Building BM25 index over {len(chunk_texts)} chunks...", file=sys.stderr)
bm25_index = build_bm25_index(chunk_texts)
print(f"BM25 index built. Ready to process questions.", file=sys.stderr)

predictions = {}
verification_stats = {"SUPPORTED": 0, "PARTIAL": 0, "NOT_SUPPORTED": 0, "N/A": 0}
total_attempts = 0

print("\n" + "=" * 60 + "\n")
print(f"Running Extension 6: RAG with Document Chunking + Answer Verification")
print(f"Max verification attempts per question: {args.max_attempts}")
print(f"Processing {len(questions)} questions...")
print("=" * 60 + "\n")

for idx, item in enumerate(questions):
    if (idx + 1) % 5 == 0:
        print(f"Progress: {idx + 1}/{len(questions)} questions processed...", file=sys.stderr)
    question = item.get("question", "")
    if not question:
        continue
    question_id = item.get("question_id", f"q{idx+1:03d}")

    prediction = rag_predict_with_verification(
        question,
        bm25_index,
        chunk_texts,
        chunk_ids,
        chunk_sent_lists,
        k=5,
        score_threshold=score_threshold,
        max_attempts=args.max_attempts,
    )

    predictions[question_id] = {
        "question": question,
        "answer": prediction.get("answer", ""),
        "retrieved_docs": prediction.get("retrieved_docs", []),
        "evidence_sentences": prediction.get("evidence_sentences", []),
        "verification_status": prediction.get("verification_status", "N/A"),
        "verification_attempts": prediction.get("verification_attempts", 0),
        "answer_doc_index": prediction.get("answer_doc_index", 0),
    }

    # Track stats
    status = prediction.get("verification_status", "N/A")
    verification_stats[status] = verification_stats.get(status, 0) + 1
    total_attempts += prediction.get("verification_attempts", 0)

    print(f"Q{idx+1}:", question[:70] + "..." if len(question) > 70 else question)
    print(f"A:", prediction.get("answer", "")[:90] + "..." if len(prediction.get("answer", "")) > 90 else prediction.get("answer", ""))
    print(f"Verification: {status} (attempts: {prediction.get('verification_attempts', 0)})")
    print("-" * 40)

with open(output_file, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\nPredictions saved to: {output_file}")
print(f"\nVerification Statistics:")
print(f"  SUPPORTED: {verification_stats['SUPPORTED']}")
print(f"  PARTIAL: {verification_stats['PARTIAL']}")
print(f"  NOT_SUPPORTED: {verification_stats['NOT_SUPPORTED']}")
print(f"  N/A: {verification_stats['N/A']}")
print(f"  Average attempts per question: {total_attempts / len(questions):.2f}")

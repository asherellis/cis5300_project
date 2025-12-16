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

# get top rankers
def BM25_topN(question, bm25_index, doc_texts, N=30):
    tokenized_question = tokenize(question)
    scores = bm25_index.get_scores(tokenized_question)

    top_ids = np.argsort(scores)[::-1][:N]
    top_texts = [doc_texts[i] for i in top_ids]

    return top_ids, top_texts, scores
    
# Add the rerank function - will take smaller pool from BM25_topN
def rerank_encoder(question, top_ids, top_texts, k=5):
    input_pairs = [(question, t) for t in top_texts]
    new_scores = rerank_model.predict(input_pairs)

    sorted_by_reranking = list(zip(top_ids, top_texts, new_scores))
    sorted_by_reranking.sort(key = lambda x:x[2], reverse = True)
    top_retrieved = sorted_by_reranking[:k]

    # Return both IDs and cross-encoder scores
    reranked_ids = [ids for ids, _, _ in top_retrieved]
    reranked_scores = [float(score) for _, _, score in top_retrieved]

    return reranked_ids, reranked_scores



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


# return list of (sid, text) tuples for the whole document
def get_sentences_with_ids(doc_structure, doc_text):
    if doc_structure and 'sentences' in doc_structure:
        collect = []
        for sentence in doc_structure['sentences']:
            sid, text = sentence.get('sid'), sentence.get('text')
            if sid and text:
                collect.append((sid, text))
        return collect
    # this is the fallback
    sents = custom_sent_tokenize(doc_text)
    return [(f"S{i+1}", sent) for i, sent in enumerate(sents) if sent.strip()]


# use sliding window to chunk over list of (sid, text) tuples
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
    # reuse the existing load_documents
    doc_texts, doc_ids, doc_structures = load_documents(doc_dir)
    chunk_texts = []
    chunk_ids = []
    chunk_sent_lists = []

    for dt, did, ds in zip(doc_texts, doc_ids, doc_structures):
        sentence_list = get_sentences_with_ids(ds, dt)
        for c in chunk_sentences(sentence_list, window=window, overlap=overlap):
            chunk_texts.append(" ".join(t for _, t in c))
            chunk_ids.append(did)
            chunk_sent_lists.append(c) # the whole chunks
    
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
# Now it's chunk level
def rag_predict(
    question,
    bm25_index,
    chunk_texts,
    chunk_ids,
    chunk_sent_lists,
    k=5,
    score_threshold=DEFAULT_THRESHOLD,
):
    if not chunk_texts:
        return {
            "answer": "I cannot answer this question as no documents are available in the corpus.",
            "retrieved_docs": [],
            "evidence_sentences": [],
        }

    # query_tokens = tokenize(question)
    # scores = bm25_index.get_scores(query_tokens)
    # top_indices = sorted(
    #     range(len(scores)), key=lambda i: scores[i], reverse=True
    # )[:k]
    # modify N since we want the candidate pool to be larger
    N = max(k * 10, 50)
    cand_id, cand_text, scores = BM25_topN(question, bm25_index, chunk_texts, N)
    # top_score = scores[top_indices[0]] if top_indices else 0.0

    if len(cand_id) == 0:
        return {
            "answer": "I cannot answer this question as no documents are available in the corpus.",
            "retrieved_docs": [],
            "evidence_sentences": [],
        }
    
    top_score = scores[cand_id[0]] if len(cand_id) > 0 else 0.0

    if top_score < score_threshold:
        return {
            "answer": "I cannot answer this question based on the available documents. No sufficiently relevant documents were found.",
            "retrieved_docs": [],
            "evidence_sentences": [],
        }
    
    
    rerank_pool = min(len(cand_id), 30)
    reranked_ids, reranked_scores = rerank_encoder(question, cand_id, cand_text, k=rerank_pool)

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
        }

    retrieved_docs = [
        {
            "doc_id": chunk_ids[chunk_idx],
            "score": score,  # Use cross-encoder score, not BM25
            "rank": rank + 1,
        }
        for rank, (chunk_idx, score) in enumerate(chosen)
    ]

    # Generate answer using LLM from top-ranked document
    top_chunk_idx, _ = chosen[0]
    top_chunk_text = chunk_texts[top_chunk_idx]
    prompt = build_rag_prompt(question, top_chunk_text)
    answer = query_groq(prompt)

    # Extract citation sentences using LLM-based method (like Extension 3)
    evidence_sentences = extract_sentence_ids_llm_from_chunk(question, answer, chunk_sent_lists[top_chunk_idx])

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "evidence_sentences": evidence_sentences,
    }

parser = argparse.ArgumentParser(
    description="Run Extension 5: RAG (BM25 retrieval + Cross-Encoder Reranking + Document Chunking + LLM generation)."
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

args = parser.parse_args()

split = args.split
input_file = f"data/questions/{split}_final_jsonl.txt"
doc_dir = "data/corpus"
output_file = f"output/{split}_extension5_output.json"
score_threshold = args.score_threshold

questions = load_jsonl(input_file)
print(f"Loaded {len(questions)} questions.", file=sys.stderr)

# doc_texts, doc_ids, doc_structures = load_documents(doc_dir)
chunk_texts, chunk_ids, chunk_sent_lists = load_documents_chunked(doc_dir, window=args.chunk_window, overlap=args.chunk_overlap)
# print(f"Building BM25 index...", file=sys.stderr)
print(f"Building BM25 index over {len(chunk_texts)} chunks...", file=sys.stderr)
bm25_index = build_bm25_index(chunk_texts)
print(f"BM25 index built. Ready to process questions.", file=sys.stderr)

predictions = {}
print("\n" + "=" * 60 + "\n")
print(f"Running Extension 5: RAG Baseline (with rerank function and doc chunking)")
print(f"Using top-1 chunk for answer generation")
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
        chunk_texts,
        chunk_ids,
        chunk_sent_lists,
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


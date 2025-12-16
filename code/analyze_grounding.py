import argparse
import json
import os
import sys
from groq import Groq

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


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_document(doc_id, doc_dir="data/corpus"):
    """Load document text by ID."""
    doc_id_clean = doc_id.replace('/', os.sep)

    # Build list of possible paths
    possible_paths = [
        os.path.join(doc_dir, doc_id_clean + '.txt'),
        os.path.join(doc_dir, doc_id_clean + '.json'),
    ]

    # If doc_id doesn't include subdirectory, search in all subdirs
    if os.sep not in doc_id_clean and '/' not in doc_id:
        for subdir in os.listdir(doc_dir):
            subdir_path = os.path.join(doc_dir, subdir)
            if os.path.isdir(subdir_path):
                possible_paths.append(os.path.join(subdir_path, doc_id_clean + '.txt'))
                possible_paths.append(os.path.join(subdir_path, doc_id_clean + '.json'))

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                try:
                    doc_data = json.loads(content)
                    if 'sentences' in doc_data:
                        return ' '.join(s.get('text', '') for s in doc_data['sentences'])
                    return doc_data.get('text', '') or doc_data.get('content', '') or content
                except json.JSONDecodeError:
                    return content
            except:
                continue
    return None


def truncate_text(text, max_tokens=1200):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens]) + "..."


def verify_answer(question, answer, doc_text):
    """Check if answer is supported by document."""
    if not doc_text:
        return "NO_DOC"

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

        if "SUPPORTED" in result and "NOT" not in result:
            return "SUPPORTED"
        elif "NOT_SUPPORTED" in result or "NOT SUPPORTED" in result:
            return "NOT_SUPPORTED"
        elif "PARTIAL" in result:
            return "PARTIAL"
        else:
            return "UNKNOWN"
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return "ERROR"


def analyze_extension(extension_num, split="train"):
    """Analyze grounding for a specific extension's outputs."""
    output_file = f"output/{split}_extension{extension_num}_output.json"

    if not os.path.exists(output_file):
        print(f"File not found: {output_file}")
        return None

    predictions = load_json(output_file)

    results = {
        "extension": extension_num,
        "total": 0,
        "SUPPORTED": 0,
        "PARTIAL": 0,
        "NOT_SUPPORTED": 0,
        "NO_DOC": 0,
        "ERROR": 0,
        "UNKNOWN": 0,
        "per_question": []
    }

    print(f"\n{'='*60}")
    print(f"Analyzing Extension {extension_num} ({split} split)")
    print(f"{'='*60}")

    for qid, pred in predictions.items():
        results["total"] += 1

        question = pred.get("question", "")
        answer = pred.get("answer", "")
        retrieved_docs = pred.get("retrieved_docs", [])

        # Skip if no answer or "cannot answer" type responses
        if not answer or "cannot answer" in answer.lower() or "does not provide sufficient" in answer.lower():
            results["per_question"].append({
                "question_id": qid,
                "status": "SKIPPED",
                "reason": "No answer or declined to answer"
            })
            continue

        # Get the document that was actually used for the answer
        if not retrieved_docs:
            status = "NO_DOC"
        else:
            # Extension 6 stores answer_doc_index (exact index in retrieved_docs)
            # Fall back to 0 for other extensions or old Extension 6 outputs
            doc_index = pred.get("answer_doc_index", 0)

            # Make sure index is valid
            if doc_index >= len(retrieved_docs):
                doc_index = 0

            doc_id = retrieved_docs[doc_index].get("doc_id", "")
            doc_text = load_document(doc_id)
            status = verify_answer(question, answer, doc_text)

        results[status] = results.get(status, 0) + 1
        results["per_question"].append({
            "question_id": qid,
            "status": status,
            "answer_preview": answer[:80] + "..." if len(answer) > 80 else answer
        })

        # Progress indicator
        print(f"  {qid}: {status}")

    # Calculate percentages
    answered = results["SUPPORTED"] + results["PARTIAL"] + results["NOT_SUPPORTED"]
    if answered > 0:
        results["supported_pct"] = results["SUPPORTED"] / answered * 100
        results["partial_pct"] = results["PARTIAL"] / answered * 100
        results["not_supported_pct"] = results["NOT_SUPPORTED"] / answered * 100

    return results


def print_summary(all_results):
    """Print comparison summary across extensions."""
    print(f"\n{'='*60}")
    print("GROUNDING ANALYSIS SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Extension':<12} {'Supported':<12} {'Partial':<12} {'Not Supported':<15} {'Grounded %':<12}")
    print("-" * 63)

    for ext_num, results in sorted(all_results.items()):
        if results is None:
            continue
        supported = results.get("SUPPORTED", 0)
        partial = results.get("PARTIAL", 0)
        not_supported = results.get("NOT_SUPPORTED", 0)
        total_checked = supported + partial + not_supported

        if total_checked > 0:
            grounded_pct = (supported + partial) / total_checked * 100
        else:
            grounded_pct = 0

        print(f"Extension {ext_num:<2} {supported:<12} {partial:<12} {not_supported:<15} {grounded_pct:.1f}%")

    print()


parser = argparse.ArgumentParser(description="Analyze grounding across extensions")
parser.add_argument("--extensions", type=str, default="1,2,3,4,5,6",
                    help="Comma-separated list of extension numbers to analyze")
parser.add_argument("--split", type=str, default="train",
                    choices=["train", "dev", "test"])
parser.add_argument("--output", type=str, default=None,
                    help="Output file for results JSON")

args = parser.parse_args()

extensions = [int(x.strip()) for x in args.extensions.split(",")]

all_results = {}
for ext_num in extensions:
    results = analyze_extension(ext_num, args.split)
    all_results[ext_num] = results

print_summary(all_results)

if args.output:
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {args.output}")

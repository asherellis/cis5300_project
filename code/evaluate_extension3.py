import argparse
import json
import os
import sys

def load_jsonl(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_predictions(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_document(doc_id, doc_dir):
    doc_id_clean = doc_id.replace('/', os.sep)
    possible_paths = [
        os.path.join(doc_dir, doc_id_clean + '.txt'),
        os.path.join(doc_dir, doc_id_clean + '.json'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    doc_data = json.loads(content)
                    if 'sentences' in doc_data:
                        return doc_data
                    return None
                except json.JSONDecodeError:
                    return None
            except Exception as e:
                continue
    
    return None

def evaluate_retrieval(questions, predictions):
    results = {
        'recall_at_1': 0,
        'recall_at_5': 0,
        'total': len(questions),
        'per_question': []
    }
    
    for i, question in enumerate(questions):
        qid = f"q{i+1:03d}"
        gold_doc_id = question.get('doc_id', '')
        
        pred = predictions.get(qid, {})
        retrieved_docs = pred.get('retrieved_docs', [])
        
        recall_at_1 = 0
        if retrieved_docs and retrieved_docs[0]['doc_id'] == gold_doc_id:
            recall_at_1 = 1
            results['recall_at_1'] += 1
        
        recall_at_5 = 0
        if any(d['doc_id'] == gold_doc_id for d in retrieved_docs[:5]):
            recall_at_5 = 1
            results['recall_at_5'] += 1
        
        results['per_question'].append({
            'question_id': qid,
            'gold_doc_id': gold_doc_id,
            'retrieved_top_1': retrieved_docs[0]['doc_id'] if retrieved_docs else None,
            'recall_at_1': recall_at_1,
            'recall_at_5': recall_at_5
        })
    
    results['recall_at_1_pct'] = (results['recall_at_1'] / results['total'] * 100) if results['total'] > 0 else 0
    results['recall_at_5_pct'] = (results['recall_at_5'] / results['total'] * 100) if results['total'] > 0 else 0
    
    return results

def evaluate_citation(questions, predictions):
    results = {
        'total_questions': len(questions),
        'questions_with_evidence': 0,
        'precision_sum': 0,
        'recall_sum': 0,
        'f1_sum': 0,
        'per_question': []
    }
    
    for i, question in enumerate(questions):
        qid = f"q{i+1:03d}"
        gold_sentences = set(question.get('evidence_sentences', []))
        gold_doc_id = question.get('doc_id', '')
        
        if not gold_sentences:
            continue
        
        results['questions_with_evidence'] += 1
        
        pred = predictions.get(qid, {})
        pred_sentences = set(pred.get('evidence_sentences', []))
        
        if pred_sentences:
            intersection = gold_sentences & pred_sentences
            precision = len(intersection) / len(pred_sentences)
            recall = len(intersection) / len(gold_sentences)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0
        
        results['precision_sum'] += precision
        results['recall_sum'] += recall
        results['f1_sum'] += f1
        
        results['per_question'].append({
            'question_id': qid,
            'gold_doc_id': gold_doc_id,
            'gold_sentences': sorted(list(gold_sentences)),
            'pred_sentences': sorted(list(pred_sentences)),
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    if results['questions_with_evidence'] > 0:
        results['avg_precision'] = results['precision_sum'] / results['questions_with_evidence']
        results['avg_recall'] = results['recall_sum'] / results['questions_with_evidence']
        results['avg_f1'] = results['f1_sum'] / results['questions_with_evidence']
    else:
        results['avg_precision'] = 0
        results['avg_recall'] = 0
        results['avg_f1'] = 0
    
    return results

def compute_evidence_score(gold_sentences, pred_sentences, doc_structure):
    if not doc_structure or 'sentences' not in doc_structure:
        gold = set(gold_sentences)
        pred = set(pred_sentences)
        if not gold:
            return 1.0 if not pred else 0.0
        intersection = gold & pred
        return len(intersection) / len(gold) if gold else 0.0
    
    sid_to_text = {s['sid']: s['text'] for s in doc_structure['sentences'] if 'sid' in s and 'text' in s}
    
    gold_words = []
    for sid in gold_sentences:
        if sid in sid_to_text:
            gold_words.extend(sid_to_text[sid].lower().split())
    
    pred_words = []
    for sid in pred_sentences:
        if sid in sid_to_text:
            pred_words.extend(sid_to_text[sid].lower().split())
    
    if not gold_words:
        return 1.0 if not pred_words else 0.0
    
    gold_set = set(gold_words)
    pred_set = set(pred_words)
    overlap = gold_set & pred_set
    
    return len(overlap) / len(gold_set) if gold_set else 0.0

def evaluate_llm_judge(questions, predictions, doc_dir, lambda_weight=0.5):
    try:
        from groq import Groq
    except ImportError:
        print("Error: 'groq' package not installed. Run: pip install groq", file=sys.stderr)
        return None
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("="*60)
        print("Submit Groq API Key")
        print("\n1. Go to: https://console.groq.com/keys")
        print("2. Click 'Create API Key'")
        print("3. Copy key and paste it below")
        print("\n(Or press Enter to skip LLM-as-Judge evaluation)\n")
        groq_api_key = input("Enter your Groq API key: ").strip()
        if not groq_api_key:
            print("Skipping LLM-as-Judge evaluation (no API key provided).", file=sys.stderr)
            return None
    
    client = Groq(api_key=groq_api_key)
    model_name = "llama-3.1-8b-instant"
    
    results = {
        'total_questions': len(questions),
        'questions_with_rubrics': 0,
        'answer_score_sum': 0,
        'evidence_score_sum': 0,
        'combined_score_sum': 0,
        'lambda_weight': lambda_weight,
        'per_question': []
    }
    
    questions_with_rubrics = [q for q in questions if 'rubric' in q]
    total_rubrics = len(questions_with_rubrics)
    
    if total_rubrics == 0:
        print("No questions with rubrics found. Skipping LLM-as-Judge evaluation.", file=sys.stderr)
        return None
    
    print(f"\nRunning LLM-as-Judge evaluation on {total_rubrics} questions...", file=sys.stderr)
    print("This may take several minutes...", file=sys.stderr)
    
    for i, question in enumerate(questions):
        qid = f"q{i+1:03d}"
        
        if 'rubric' not in question:
            continue
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i + 1}/{len(questions)} questions processed...", file=sys.stderr)
        
        results['questions_with_rubrics'] += 1
        
        gold_answer = question.get('answer', '')
        gold_sentences = set(question.get('evidence_sentences', []))
        rubric = question.get('rubric', {})
        question_text = question.get('question', '')
        gold_doc_id = question.get('doc_id', '')
        
        pred = predictions.get(qid, {})
        model_answer = pred.get('answer', '')
        pred_sentences = set(pred.get('evidence_sentences', []))
        
        prompt = f"""You are an evaluator. You must score a model's answer from 1 to 5 using the rubric.

QUESTION:
{question_text}

GOLD ANSWER:
{gold_answer}

RUBRIC DESCRIPTION:
{rubric.get("description", "")}

RUBRIC SCALE:
{json.dumps(rubric.get("scale", {}), indent=2)}

MODEL ANSWER:
{model_answer}

INSTRUCTIONS:
1. Read the rubric carefully.
2. Compare the model answer ONLY against the article content implied by the gold answer & rubric (do not bring external facts).
3. Choose a single integer score from 1 to 5.
4. Respond with a pure JSON object of the form:
{{
  "score": <integer between 1 and 5>,
  "rationale": "<short explanation>"
}}
No extra text."""
        
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a strict grading assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            
            content = resp.choices[0].message.content.strip()
            
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = content[start:end]
                    parsed = json.loads(json_str)
                else:
                    parsed = {"score": 0, "rationale": "Failed to parse LLM response"}
            
            answer_score_raw = parsed.get("score", 0)
            rationale = parsed.get("rationale", "")
            
        except Exception as e:
            error_msg = str(e)
            answer_score_raw = 0
            rationale = f"Error: {error_msg}"
        
        answer_score_norm = answer_score_raw / 5.0
        
        doc_structure = load_document(gold_doc_id, doc_dir) if gold_doc_id else None
        evidence_score = compute_evidence_score(gold_sentences, pred_sentences, doc_structure)
        
        combined_score = lambda_weight * answer_score_norm + (1.0 - lambda_weight) * evidence_score
        
        results['answer_score_sum'] += answer_score_raw
        results['evidence_score_sum'] += evidence_score
        results['combined_score_sum'] += combined_score
        
        results['per_question'].append({
            'question_id': qid,
            'answer_score_raw': answer_score_raw,
            'answer_score_normalized': answer_score_norm,
            'answer_rationale': rationale,
            'evidence_score': evidence_score,
            'combined_score': combined_score,
            'gold_sentences': sorted(list(gold_sentences)),
            'pred_sentences': sorted(list(pred_sentences)),
        })
    
    if results['questions_with_rubrics'] > 0:
        results['avg_answer_score'] = results['answer_score_sum'] / results['questions_with_rubrics']
        results['avg_evidence_score'] = results['evidence_score_sum'] / results['questions_with_rubrics']
        results['avg_combined_score'] = results['combined_score_sum'] / results['questions_with_rubrics']
    else:
        results['avg_answer_score'] = 0
        results['avg_evidence_score'] = 0
        results['avg_combined_score'] = 0
    
    return results

def print_results(eval_type, results):
    print(f"\n{'='*60}")
    print(f"{eval_type.upper()} EVALUATION RESULTS")
    print(f"{'='*60}")
    
    if eval_type == "retrieval":
        print(f"\nTotal Questions: {results['total']}")
        print(f"\nRecall@1: {results['recall_at_1']}/{results['total']} = {results['recall_at_1_pct']:.2f}%")
        print(f"Recall@5: {results['recall_at_5']}/{results['total']} = {results['recall_at_5_pct']:.2f}%")
        
        print(f"\n{'-'*60}")
        print("Per-Question Breakdown:")
        print(f"{'-'*60}")
        for item in results['per_question']:
            r1 = "✓" if item['recall_at_1'] else "✗"
            r5 = "✓" if item['recall_at_5'] else "✗"
            print(f"{item['question_id']}: R@1={r1} R@5={r5}")
            print(f"  Gold: {item['gold_doc_id']}")
            print(f"  Top1: {item['retrieved_top_1']}")
    
    elif eval_type == "citation":
        print(f"\nTotal Questions: {results['total_questions']}")
        print(f"Questions with Evidence: {results['questions_with_evidence']}")
        
        if results['questions_with_evidence'] > 0:
            print(f"\n--- OVERALL CITATION METRICS (all questions) ---")
            print(f"Average Precision: {results['avg_precision']:.4f}")
            print(f"Average Recall: {results['avg_recall']:.4f}")
            print(f"Average F1: {results['avg_f1']:.4f}")
            
            print(f"\n{'-'*60}")
            print("Per-Question Breakdown:")
            print(f"{'-'*60}")
            for item in results['per_question']:
                print(f"{item['question_id']}: P={item['precision']:.2f} R={item['recall']:.2f} F1={item['f1']:.2f}")
                print(f"  Gold: {item['gold_sentences']}")
                print(f"  Pred: {item['pred_sentences']}")
    
    elif eval_type == "llm_judge":
        print(f"\nTotal Questions: {results['total_questions']}")
        print(f"Questions with Rubrics: {results['questions_with_rubrics']}")
        print(f"Lambda Weight (answer vs evidence): {results['lambda_weight']:.2f}")
        
        if results['questions_with_rubrics'] > 0:
            print(f"\n--- OVERALL SCORES ---")
            print(f"Average Answer Score (1-5): {results['avg_answer_score']:.2f}")
            print(f"Average Evidence Score (0-1): {results['avg_evidence_score']:.4f}")
            print(f"Average Combined Score (0-1): {results['avg_combined_score']:.4f}")
            
            print(f"\n{'-'*60}")
            print("Per-Question Breakdown:")
            print(f"{'-'*60}")
            for item in results['per_question']:
                print(f"{item['question_id']}: Answer={item['answer_score_raw']}/5 Evidence={item['evidence_score']:.4f} Combined={item['combined_score']:.4f}")
                print(f"  Rationale: {item['answer_rationale']}")
                print(f"  Gold Sentences: {item['gold_sentences']}")
                print(f"  Pred Sentences: {item['pred_sentences']}")
                print()

parser = argparse.ArgumentParser(
    description="Evaluate Extension 3 (RAG + Reranking + LLM Citations) predictions."
)
parser.add_argument(
    "split",
    nargs="?",
    default="train",
    choices=["train", "dev", "test"],
    help="Which split to evaluate (train, dev, or test). Default: train"
)

args = parser.parse_args()
split = args.split

questions_file = f"data/questions/{split}_final_jsonl.txt"
extension3_output = f"output/{split}_extension3_output.json"

if not os.path.exists(extension3_output):
    print(f"Error: Output file not found: {extension3_output}", file=sys.stderr)
    sys.exit(1)

print(f"\nEvaluating Extension 3 (RAG + Reranking + LLM Citations) on {split} split...")
print(f"Questions file: {questions_file}")
print(f"Predictions file: {extension3_output}\n")

questions = load_jsonl(questions_file)
print(f"Loaded {len(questions)} questions\n")

predictions = load_predictions(extension3_output)
print(f"Loaded predictions for {len(predictions)} questions\n")

results = {}

retrieval_results = evaluate_retrieval(questions, predictions)
results['retrieval'] = retrieval_results
print_results("retrieval", retrieval_results)

citation_results = evaluate_citation(questions, predictions)
results['citation'] = citation_results
print_results("citation", citation_results)

llm_results = evaluate_llm_judge(questions, predictions, "data/corpus", lambda_weight=0.5)
if llm_results:
    results['llm_judge'] = llm_results
    print_results("llm_judge", llm_results)

all_evaluation = {'rag_extension3': results}

output_file = f"output/{split}_extension3_evaluation.json"
with open(output_file, 'w') as f:
    json.dump(all_evaluation, f, indent=2)
print(f"{'='*60}")
print(f"Evaluation results saved to: {output_file}")
print(f"{'='*60}\n")

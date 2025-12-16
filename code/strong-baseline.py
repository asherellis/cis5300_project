import argparse
import sys
import pandas as pd
import os
import json
from groq import Groq

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("="*60)
    print("Submit Goq API Key")
    print("\n1. Go to: https://console.groq.com/keys")
    print("2. Click 'Create API Key'")
    print("3. Copy key and paste it below\n")
    groq_api_key = input("Enter your Groq API key: ").strip()
    if not groq_api_key:
        print("No API key provided. Exiting.")
        exit(1)

client = Groq(api_key=groq_api_key)
model_name = "llama-3.1-8b-instant"

parser = argparse.ArgumentParser(
    description="Run the strong LLM baseline."
)
parser.add_argument(
    "split",
    nargs="?",
    default="train",
    choices=["train", "dev", "test"],
    help="Which split to run (train, dev, or test). Default: train",
)
args = parser.parse_args()
split = args.split
input_file = f"data/questions/{split}_final_jsonl.txt"
output_file = f"output/{split}_strong_output.json"

from io import StringIO
with open(input_file, "r") as f:
    text = f.read().replace("NaN", "null")
df = pd.read_json(StringIO(text), lines=True)

# df = pd.read_json(input_file, lines=True)
questions = df['question'].tolist()
question_ids = df['question_id'].tolist() if 'question_id' in df.columns else [f"q{i+1:03d}" for i in range(len(questions))]

def build_prompt(question: str):
    return (f"Question:\n{question}\n"
             "Answer the question in two to four complete sentences.\n"
             "Do not include any notes, disclaimers, bullet points, or follow-up offers.\n"
             "Give only the answer.\n"
             "Answer:")

def query_groq(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.7,
            max_tokens=128,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nError: {e}")
        return "Error generating answer"

prompts = [build_prompt(q) for q in questions]
answers_list = []
for prompt in prompts:
    answer = query_groq(prompt)
    answers_list.append(answer)

predictions = {}
for question_id, question, answer in zip(question_ids, questions, answers_list):
    predictions[question_id] = {
        "question": question,
        "answer": answer
    }

with open(output_file, "w") as f:
    json.dump(predictions, f, indent=2)

print("\n" + "=" * 60 + "\n")
for q, a in zip(questions, answers_list):
    print("Q:", q)
    print("A:", a)
    print("-" * 40)


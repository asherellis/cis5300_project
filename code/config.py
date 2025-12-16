
import os

# Get the project root (parent of code directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
NLP_DATA_DIR = os.path.join(DATA_DIR, "questions")

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

def get_questions_file(split):
    """Get path to questions file for a given split."""
    return os.path.join(NLP_DATA_DIR, f"{split}_final_jsonl.txt")

def get_output_file(split, model_type):
    """Get path to output file for a given split and model type."""
    return os.path.join(OUTPUT_DIR, f"{split}_{model_type}_output.json")

def get_evaluation_file(split):
    """Get path to evaluation results file."""
    return os.path.join(OUTPUT_DIR, f"{split}_evaluation.json")

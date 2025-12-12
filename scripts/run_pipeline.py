import argparse
from src.pipeline.pipeline import run_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--conversation", required=True)
parser.add_argument("--context", required=True)
args = parser.parse_args()

result = run_evaluation(args.conversation, args.context)
print(result)

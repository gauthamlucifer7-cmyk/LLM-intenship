import json
import glob
from src.pipeline.pipeline import run_evaluation

files = glob.glob("data/batch/*.json")

for f in files:
    ctx = "data/sample_context_vectors.json"
    print("Processing:", f)
    result = run_evaluation(f, ctx)
    print(result)
  

import json
from src.llm.models import pricing

def compute_cost(tokens_prompt, tokens_completion, model="gpt-4.1"):
    p = pricing[model]
    cost = (tokens_prompt / 1000) * p["prompt_per_1k"] + \
           (tokens_completion / 1000) * p["completion_per_1k"]
    return round(cost, 6)
  

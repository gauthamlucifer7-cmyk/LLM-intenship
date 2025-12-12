from src.llm.client import LLMClient
import json

def score_factual_accuracy(response, context_chunks):
    judge_prompt = f"""
Evaluate accuracy of the assistant response against context (0–1).

Context: {json.dumps(context_chunks)}
Response: {response}

Return only a number.
"""
    model = LLMClient(model="gpt-4.1-mini")
    result = model.chat([{"role": "user", "content": judge_prompt}])

    try:
        return float(result.choices[0].message["content"])
    except:
        return 0.0
      

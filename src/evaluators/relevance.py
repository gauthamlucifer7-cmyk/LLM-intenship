from src.llm.client import LLMClient

def score_relevance(user_message, response):
    judge_prompt = f"""
Rate relevance and completeness from 0 to 1.

User message: {user_message}
Assistant response: {response}

Return only a number.
"""
    result = LLMClient(model="gpt-4.1-mini").chat(
        [{"role": "user", "content": judge_prompt}]
    )

    try:
        return float(result.choices[0].message["content"])
    except:
        return 0.0

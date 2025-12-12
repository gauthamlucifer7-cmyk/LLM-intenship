import time
from .client import LLMClient

def generate_llm_answer(conversation, context_chunks, model="gpt-4.1"):
    client = LLMClient(model=model)

    prompt = (
        "Use only the following context when answering.\n\n"
        f"Context:\n{context_chunks}\n\n"
        "Conversation:\n"
        f"{conversation}\n"
    )

    start = time.time()
    completion = client.chat([{"role": "user", "content": prompt}])
    latency_ms = (time.time() - start) * 1000

    return {
        "response": completion.choices[0].message["content"],
        "latency_ms": latency_ms,
        "tokens_prompt": completion.usage["prompt_tokens"],
        "tokens_completion": completion.usage["completion_tokens"]
    }

def aggregate_metrics(relevance, accuracy, latency, cost, tokens):
    return {
        "relevance": relevance,
        "factual_accuracy": accuracy,
        "latency_ms": latency,
        "cost_usd": cost,
        "tokens": tokens
    }
  

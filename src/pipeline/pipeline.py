from .loader import load_json
from src.llm.generator import generate_llm_answer
from src.evaluators.relevance import score_relevance
from src.evaluators.factual_accuracy import score_factual_accuracy
from src.evaluators.latency_cost import compute_cost
from src.evaluators.aggregator import aggregate_metrics
from .utils import save_json, append_csv
import os

def run_evaluation(conversation_path, context_path):
    conversation = load_json(conversation_path)
    context = load_json(context_path)

    user_message = conversation[-1]["content"]
    context_chunks = [c["text"] for c in context["chunks"]]

    llm_output = generate_llm_answer(conversation, context_chunks)

    relevance = score_relevance(user_message, llm_output["response"])
    accuracy = score_factual_accuracy(llm_output["response"], context_chunks)

    cost = compute_cost(
        llm_output["tokens_prompt"],
        llm_output["tokens_completion"]
    )

    metrics = aggregate_metrics(
        relevance, accuracy, llm_output["latency_ms"], cost,
        {
            "prompt": llm_output["tokens_prompt"],
            "completion": llm_output["tokens_completion"]
        }
    )

    results = {
        "response": llm_output["response"],
        "metrics": metrics
    }

    save_json(results, "results/latest.json")

    append_csv(
        {
            "relevance": relevance,
            "accuracy": accuracy,
            "latency": llm_output["latency_ms"],
            "cost": cost
        }
    )

    with open("results/latest.md", "w") as f:
        f.write(f"# LLM Evaluation\n\n")
        f.write(f"**Relevance:** {relevance}\n")
        f.write(f"**Accuracy:** {accuracy}\n")
        f.write(f"**Latency:** {llm_output['latency_ms']} ms\n")
        f.write(f"**Cost:** ${cost}\n")

    return results

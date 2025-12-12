from src.pipeline.pipeline import run_evaluation

def test_pipeline_runs():
    res = run_evaluation("data/sample_conversation.json",
                         "data/sample_context_vectors.json")
    assert "response" in res
    assert "metrics" in res

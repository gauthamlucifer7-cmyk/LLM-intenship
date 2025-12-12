from src.evaluators.factual_accuracy import score_factual_accuracy

def test_accuracy_range():
    score = score_factual_accuracy("Test", ["test context"])
    assert 0 <= score <= 1

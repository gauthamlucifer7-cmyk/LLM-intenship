from src.evaluators.relevance import score_relevance

def test_relevance_basic():
    assert 0 <= score_relevance("Hello", "Hi!") <= 1

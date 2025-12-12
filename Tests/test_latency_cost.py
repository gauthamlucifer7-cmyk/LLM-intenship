from src.evaluators.latency_cost import compute_cost

def test_cost_calculation():
    cost = compute_cost(100, 100, model="gpt-4.1")
    assert cost > 0

import math

from conftest import make_queue, run_sim


def test_end_to_end_metrics_are_sane(queue):
    coord = run_sim(queue)
    stats = coord.get_global_stats()
    assert coord.request_serviced == len(queue) > 0
    assert 0 < stats.TTFT < stats.T99_latency
    assert stats.TPOT > 0
    assert stats.T50_latency <= stats.T90_latency <= stats.T99_latency
    assert stats.total_token_throughput > stats.output_throughput > 0


def test_total_throughput_ignores_reused_kv():
    """past_context (reused prefix KV) far larger than input_len must not drive throughput negative."""
    plain = make_queue()
    with_kv = make_queue()
    for req in with_kv:
        req.past_context = 20 * req.input_len
    stats = run_sim(with_kv).get_global_stats()
    assert stats.total_token_throughput > stats.output_throughput > 0
    # Prefilled tokens per request are the same with or without reused KV.
    base = run_sim(plain).get_global_stats()
    prompt_tokens_per_req = lambda s: (s.total_token_throughput - s.output_throughput) / s.rps
    assert math.isclose(prompt_tokens_per_req(stats), prompt_tokens_per_req(base))

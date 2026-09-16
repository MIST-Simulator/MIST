from concurrent.futures import ThreadPoolExecutor

from mist import BurstyDistribution, LengthVariables, NormalDistribution, PoissonDistribution, UniformDistribution
from conftest import make_queue, run_sim


def _signature(queue):
    return [(r.metrics.arrival_time, r.input_len, r.output_len) for r in queue]


def test_same_seed_same_simulation(tmp_path, monkeypatch):
    # Each run gets a cold latency cache: the on-disk cache matches by tolerance,
    # so a warm cache can legitimately return a slightly different nearby entry.
    monkeypatch.setenv("MIST_CACHE_DIR", str(tmp_path / "a"))
    a = run_sim(make_queue(seed=3)).get_global_stats()
    monkeypatch.setenv("MIST_CACHE_DIR", str(tmp_path / "b"))
    b = run_sim(make_queue(seed=3)).get_global_stats()
    assert a.to_list() == b.to_list()


def test_different_seed_different_queue():
    assert _signature(make_queue(seed=1)) != _signature(make_queue(seed=2))


def test_every_distribution_is_seeded():
    kw = dict(rps=5, sim_time=5000, input_vars=LengthVariables(512, 128), output_vars=LengthVariables(64, 16))
    for cls in (UniformDistribution, PoissonDistribution, NormalDistribution, BurstyDistribution):
        assert _signature(cls(rand_seed=11, **kw).request_queue) == _signature(cls(rand_seed=11, **kw).request_queue), cls


def test_concurrent_generation_is_reproducible():
    # Used to race on the global RNG when sweeps built queues in a thread pool.
    serial = [_signature(make_queue(seed=s)) for s in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        parallel = list(pool.map(lambda s: _signature(make_queue(seed=s)), range(8)))
    assert parallel == serial

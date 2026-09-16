"""Regression: LLMEngine runtime caches must not be shared across instances."""
from conftest import run_sim


def test_engines_do_not_share_runtime_cache(queue):
    fast = run_sim(queue, device="H100_GPU").get_global_stats()
    slow = run_sim(queue, device="A100_40GB_GPU").get_global_stats()
    assert fast.TPOT != slow.TPOT
    assert fast.TTFT != slow.TTFT


def test_engine_caches_are_distinct_objects():
    from mist import PlatformConfig, SchedulerConfig, BatchingMethod
    from mist.Engine import LLMEngine
    from conftest import MODEL
    engines = [
        LLMEngine(model=MODEL, scheduler_config=SchedulerConfig(batching_method=BatchingMethod.CHUNKED),
                  platform=PlatformConfig(device="H100_GPU", tensor_parallel_size=1, model=MODEL))
        for _ in range(2)
    ]
    assert engines[0]._decode_only_cache is not engines[1]._decode_only_cache
    assert engines[0]._mixed_batch_cache is not engines[1]._mixed_batch_cache
    assert engines[0].engine_types is not engines[1].engine_types

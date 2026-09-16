"""KV-cache retrieval: requests first fetch a cached prefix from a memory tier.

Each request carries a long reused prefix (past_context) that a
KVRetrievalEngine loads from a DRAM/SSD hierarchy before prefill and decode
run on the GPU engine.
"""
from mist import (BatchingMethod, LengthVariables, MemoryCacheConfig, PlatformConfig, SchedulerConfig,
                  SingleCacheConfig, UniformDistribution)
from mist.Coordinator import MISTCoordinator
from mist.Engine import EngineType, KVRetrievalEngine, LLMEngine, MISTEngine
from mist.Request import RequestStage

MODEL = "meta-llama/meta-llama-3.1-8b"

requests = UniformDistribution(
    rps=2, sim_time=5_000,
    input_vars=LengthVariables(512, 0), output_vars=LengthVariables(32, 0),
    stages=[RequestStage.CACHE_RETRIEVAL, RequestStage.PREFILL, RequestStage.DECODE],
    rand_seed=0,
).request_queue
for req in requests:
    req.past_context = 16_000  # tokens of reusable prefix KV stored off-GPU

gpu = LLMEngine(
    model=MODEL,
    engine_types=[EngineType.PREFILL, EngineType.DECODE],
    scheduler_config=SchedulerConfig(batching_method=BatchingMethod.CHUNKED, chunk_size=512),
    platform=PlatformConfig(device="H100_GPU", tensor_parallel_size=1, model=MODEL),
)
# SingleCacheConfig(type, memory_size GB, bandwidth GB/s, retrieval_latency ms, hit_rate)
memory = KVRetrievalEngine(
    model=MODEL,
    engine_types=[EngineType.CACHE_RETRIEVAL],
    platform=MemoryCacheConfig([
        SingleCacheConfig("DRAM", 1500, 256, 0.020, 0.4),
        SingleCacheConfig("SSD", 6000, 32, 0.100, 0.8),
    ], MODEL),
)
host = MISTEngine(model=MODEL, sim_duration=10_000_000, engine_types=[EngineType.HOST])

coordinator = MISTCoordinator(requests, logging_file=None)
coordinator.add_engine(gpu, [EngineType.PREFILL, EngineType.DECODE])
coordinator.add_engine(memory, [EngineType.CACHE_RETRIEVAL])
coordinator.add_engine(host, [EngineType.HOST])
coordinator.run_sim()

s = coordinator.get_global_stats()
retrieval = [r.stage_metrics["Cache Retrieval"].engine_exit_time - r.stage_metrics["Cache Retrieval"].engine_entry_time
             for r in coordinator.completed_requests]
print(f"served {coordinator.request_serviced}/{coordinator.request_accepted} requests")
print(f"mean KV retrieval: {sum(retrieval) / len(retrieval):.1f} ms  TTFT={s.TTFT:.1f} ms  TPOT={s.TPOT:.1f} ms")

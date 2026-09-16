"""Aggregated serving: one model replica runs both prefill and decode.

Compares chunked-prefill against continuous batching for Llama-3.1-8B on an
H100 under a Poisson workload.
"""
from copy import deepcopy

from mist import BatchingMethod, LengthVariables, PlatformConfig, PoissonDistribution, SchedulerConfig
from mist.Coordinator import MISTCoordinator
from mist.Engine import EngineType, LLMEngine

MODEL = "meta-llama/meta-llama-3.1-8b"

# All times in MIST are milliseconds.
requests = PoissonDistribution(
    rps=8, sim_time=10_000,
    input_vars=LengthVariables(1024, 256), output_vars=LengthVariables(128, 32),
    rand_seed=0,
).request_queue

for batching in (BatchingMethod.CHUNKED, BatchingMethod.CONTINUOUS):
    coordinator = MISTCoordinator(deepcopy(requests), logging_file=None)
    types = [EngineType.PREFILL, EngineType.DECODE]
    coordinator.add_engine(
        LLMEngine(
            model=MODEL,
            engine_types=types,
            scheduler_config=SchedulerConfig(batching_method=batching, chunk_size=512),
            platform=PlatformConfig(device="H100_GPU", tensor_parallel_size=1, model=MODEL, bits="bf16"),
        ),
        types,
    )
    coordinator.run_sim()
    s = coordinator.get_global_stats()
    print(f"{batching.name:<10} served {coordinator.request_serviced}/{coordinator.request_accepted}  "
          f"TTFT={s.TTFT:7.1f} ms  TPOT={s.TPOT:5.1f} ms  p99={s.T99_latency:8.1f} ms  "
          f"output={s.output_throughput:7.1f} tok/s")

from copy import deepcopy

import pytest

from mist import BatchingMethod, LengthVariables, PlatformConfig, PoissonDistribution, SchedulerConfig
from mist.Coordinator import MISTCoordinator
from mist.Engine import EngineType, LLMEngine

MODEL = "meta-llama/meta-llama-3.1-8b"


def make_queue(seed=0, rps=4, sim_time=3000):
    return PoissonDistribution(
        rps=rps, sim_time=sim_time,
        input_vars=LengthVariables(512, 128), output_vars=LengthVariables(64, 16),
        rand_seed=seed,
    ).request_queue


def run_sim(queue, device="H100_GPU", batching=BatchingMethod.CHUNKED, **sched_kwargs):
    coord = MISTCoordinator(deepcopy(queue), logging_file=None)
    types = [EngineType.PREFILL, EngineType.DECODE]
    engine = LLMEngine(
        model=MODEL, engine_types=types,
        scheduler_config=SchedulerConfig(batching_method=batching, chunk_size=512, **sched_kwargs),
        platform=PlatformConfig(device=device, tensor_parallel_size=1, model=MODEL),
    )
    coord.add_engine(engine, types)
    coord.run_sim()
    return coord


@pytest.fixture
def queue():
    return make_queue()

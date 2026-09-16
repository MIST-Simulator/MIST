from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
from mist.Scheduler.scheduler import Scheduler, SchedulerConfig, BatchingMethod
from mist.Request import Request, DataMetrics, RequestMetrics, RequestStage
import random
import heapq
import time
import enum

from mist.Input_requests.Request_inputs import RequestDistributions, UniformDistribution, PoissonDistribution, NormalDistribution

from dataclasses import dataclass, field
from mist.Platforms.platforms import PlatformConfig, PlatformType


class EngineType(enum.Enum):
    """Status of a sequence."""
    HOST = enum.auto()
    CACHE_RETRIEVAL = enum.auto()
    PREPROCESS = enum.auto()
    PREFILL = enum.auto()
    MIXED = enum.auto()
    DECODE = enum.auto()
    POSTPROCESS = enum.auto()
    RAG = enum.auto()

def stage_to_engine_mapping(stage: RequestStage):
    if stage == RequestStage.INPUT_QUERY or stage == RequestStage.OUTPUT_ANSWER:
        return EngineType.HOST
    if stage == RequestStage.RAG:
        return EngineType.RAG
    elif stage == RequestStage.PREPROCESS:
        return EngineType.PREPROCESS
    elif stage == RequestStage.PREFILL:
        return EngineType.PREFILL
    elif stage == RequestStage.DECODE:
        return EngineType.DECODE
    elif stage == RequestStage.POSTPROCESS:
        return EngineType.POSTPROCESS
    elif stage == RequestStage.CACHE_RETRIEVAL:
        return EngineType.CACHE_RETRIEVAL
    else:
        raise ValueError(f"No engine type found for the stage:{stage}")

@dataclass
class EngineMetrics:
    """Metrics associated with a Engine.
    Note all times are msecs

    Attributes:
        TTFT: Average time to first token generation time.
        TPOT: Average time per output token time.
        rps: Requests served per second.
        T50_latency: 50th percentile of request completion latency
        T90_latency: 90th percentile of request completion latency
        T95_latency: 95th percentile of request completion latency
        T99_latency: 99th percentile of request completion latency
        interactivity: Average interactivity (Tokens/s/user)
        output_throughput: Average output throughput (Tokens/s)
        total_token_throughput: Average total throughput (Tokens/s)
    """
    TTFT: float = None
    TPOT: float = None
    rps: float = None
    T50_latency: float = None
    T90_latency: float = None
    T95_latency: float = None
    T99_latency: float = None
    interactivity: float = None
    output_throughput: float = None
    total_token_throughput: float = None

    def to_list(self) -> List[float]:
        return [
            self.TTFT,
            self.TPOT,
            self.rps,
            self.T50_latency,
            self.T90_latency,
            self.T95_latency,
            self.T99_latency,
            self.interactivity,
            self.output_throughput,
            self.total_token_throughput,
        ]
    def column_names(self) -> List[str]:
        return [
            "TTFT",
            "TPOT",
            "rps",
            "T50_latency",
            "T90_latency",
            "T95_latency",
            "T99_latency",
            "interactivity",
            "output_throughput",
            "total_token_throughput"
        ]




class MISTEngine:

    def __init__(
        self,
        model: str = None,
        scheduler: Scheduler = None,
        sim_duration: float = 1000,
        engine_id:str=None,
        platform: PlatformConfig = None,
        engine_types: List[EngineType] = [EngineType.PREFILL, EngineType.DECODE],
        ) -> None:

        self.model = model
        self.scheduler = scheduler
        self.request_queue = []

        self.current_time = 0
        self.sim_duration = sim_duration    # time of simulation in msecs
        # Create the scheduler.
        self.platform = platform
        self.engine_id = engine_id
        self.engine_types = engine_types
        self.logger = None
        self.batched_engine = False     # If the engine is batched or not

        self.energy_consumed = 0

    def add_request(
        self,
        request: Request,
        arr_time: float = 0,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            input_len: The input Length of the input prompt the LLM model.
            output_len: Expected length of the output prompt for the given request.
            beam_size: number of parallel beams.

        Details:
            - Set arrival_time to the current time
            - Set request_id

        """
        ## Req id are used as sorting key for the finished requests
        # if arr_time < self.current_time and self.batched_engine == True:
            # raise ValueError(f"Request {request.request_id}-Stage {request.current_stage} had to arrive earlier to Engine:{self.engine_id}, arr_time: {arr_time}, current time: {self.current_time}")
        # else:
        # if self.engine_id == 1:
        # print(f"Add Request {request.request_id} to the Engine:{self.engine_id} and to be scheduled at {arr_time}")
        if stage_to_engine_mapping(request.current_stage) in self.engine_types:
            self.logger.log_event(request.request_id, self.engine_id, "Engine"+str(request.current_stage), time=arr_time,type= "B")
            request.update_engine_entry_time(self.engine_id, arr_time)
            heapq.heappush(self.request_queue, (arr_time, request))
        else:
            raise ValueError(f"Request {request.request_id}-Stage {request.current_stage} is not for this engine:{self.engine_id} with type:{self.engine_types}")

    def engine_idle(self) -> bool:
        if self.request_queue:
            return False
        else:
            return True

    def step(self, current_time) -> None:
        """Performs one iteration and returns newly generated results.

            Overview of the step function.

        Details:
            - Step 1: Get a list of requests that have arrived by the current time.
            - Step 2: Calls the cost model to get the runtime for each request.
            - Step 3: Processes the output. This mainly includes:
                - Append the relevant outputs of the requests.
                - Update the exit time of the request.
        """
        self.current_time = current_time
        current_req_queue = []
        while self.request_queue:
            ## Push all requests that have arrived by the current time to the scheduler
            if self.request_queue[0][0] <= self.current_time:
                req_engine_entry_time, req = heapq.heappop(self.request_queue)
                current_req_queue.append(req)
            ## If no requests have arrived by the current time.
            # break and continue to schedule
            elif current_req_queue == []:
                return self.request_queue[0][0], []
            else:
                break

        machine_runtimes = self.get_machine_runtimes(current_req_queue)
        for req, runtime in zip(current_req_queue, machine_runtimes):
            req_exit_time = req.get_engine_entry_time() + runtime
            req.update_engine_exit_time(req_exit_time)
            self.logger.log_event(req.request_id, self.engine_id, "Engine"+str(req.current_stage), time=req_exit_time,type= "E")

        self.current_time += min(machine_runtimes)

        return self.current_time, current_req_queue

    def get_machine_runtimes(self, current_req_queue):
        ## We assume that by default the machine start executing the request as soon as it arrives.
        runtimes = []
        for req in current_req_queue:
            runtimes.append(10)
        return runtimes

    # TODO: Add the logic to get the engine stats
    def get_engine_stats(self) -> EngineMetrics:
        sum_TTFT = 0
        sum_TPOT = 0
        sum_queuing_delay = 0
        sum_latency = 0
        e2e_latencies = []          # Keeps track of all the end to end latencies
        # queue_latencies = []        # Keeps track of all the queue latencies

        for i, request in enumerate(self.scheduler.finished):
            sum_TTFT += (request.data[0].finished_time - request.metrics.arrival_time)
            sum_queuing_delay += request.metrics.duration_in_queue
            # queue_latencies.append(request.metrics.duration_in_queue)
            if request.gen_tokens > 1:
                req_TPOT = 0
                for j in range(1, len(request.data)):
                    req_TPOT += (request.data[j].finished_time - request.data[j].scheduled_time)
                sum_TPOT += req_TPOT / (request.gen_tokens - 1)
            latency = request.data[-1].finished_time - request.metrics.arrival_time
            sum_latency += latency
            e2e_latencies.append(latency)

        num_requests = len(self.scheduler.finished)
        avg_TTFT = sum_TTFT / num_requests
        avg_TPOT = sum_TPOT / num_requests
        avg_queuing_delay = sum_queuing_delay / num_requests
        avg_latency = sum_latency / num_requests
        rps = 1000 / avg_latency

        e2e_latencies.sort()
        T50_latency = e2e_latencies[int(0.5 * num_requests)]
        T90_latency = e2e_latencies[int(0.9 * num_requests)]
        T95_latency = e2e_latencies[int(0.95 * num_requests)]
        T99_latency = e2e_latencies[int(0.99 * num_requests)]

        return EngineMetrics(
            TTFT=avg_TTFT,
            TPOT=avg_TPOT,
            queue_delay=avg_queuing_delay,
            rps=rps,
            Latency=avg_latency,
            T50_latency=T50_latency,
            T90_latency=T90_latency,
            T95_latency=T95_latency,
            T99_latency=T99_latency,
        )

    def can_handle_task(self, task: EngineType) -> bool:
        """Check if the engine can handle a specific task."""
        return task in self.engine_types

class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0

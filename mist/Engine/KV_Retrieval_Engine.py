from .MIST_Engine import MISTEngine, EngineMetrics, EngineType
from mist.Scheduler.scheduler import Scheduler, SchedulerConfig, BatchingMethod
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
from mist.Platforms import MemoryCacheConfig
from mist.Request import Request, DataMetrics, RequestMetrics, RequestStage
import logging
logger = logging.getLogger('mist')
import heapq
import time
import math
import time
import random

class KVRetrievalEngine(MISTEngine):

    def __init__(
        self,
        model: str,
        engine_id:str=None,
        platform: MemoryCacheConfig = None,
        engine_types: List[EngineType] = [EngineType.CACHE_RETRIEVAL],
        ) -> None:

        super().__init__(
            model=model,
            engine_id=engine_id,
            platform=platform,
            engine_types=engine_types,
        )

    def get_machine_runtimes(self, requests: List[Request]) -> List[float]:
        """
            Get the runtime for each request in the current cycle.

            Inputs:
            requests : List of all the requests in the current cycle.

            Outputs:
            runtimes: List of runtimes for each request.
        """
        req = requests[0]
        # print(f"Getting Machine Runtimes for req:{req.request_id}, {req.input_len}, {req.output_len}, {req.past_context}")
        retrieval_time = self.platform.get_KV_cache_time(requests)
        return retrieval_time
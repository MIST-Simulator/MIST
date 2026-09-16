from mist.Request import Request, DataMetrics, RequestMetrics
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
from GenZ import ModelConfig, get_configs

class SingleCacheConfig:
    """Represents a single memory cache configuration.
    """
    def __init__(self, type,
                memory_size,
                memory_bandwidth,
                retrieval_latency,
                hit_rate):
        """Initializes the memory cache configuration.

        Args:
            type (str): Type of memory
            memory_size (float): Memory Size in GB
            memory_bandwidth (float): Memory Bandwidth in GB/s
            retrieval_latency (float): Retrieval Latency in ms
            hit_rate (float): Hit rate of cache memory

        """
        self.type = type
        self.memory_size = memory_size
        self.memory_bandwidth = memory_bandwidth    ## Memory in GB/s
        self.retrieval_latency = retrieval_latency  ## Memory in ms
        self.hit_rate = hit_rate

        assert self.hit_rate >= 0 and self.hit_rate <= 1, "Hit rate should be between 0 and 1"

    def get_retrieval_time(self, size: int) -> float:
        """
            Calculate the retrieval time for a given size of data.

            Inputs:
            size : Size of the data in MB.

            Outputs:
            time: Time to retrieve the data in milliseconds.
        """
        return self.retrieval_latency + size / self.memory_bandwidth


class MemoryCacheConfig:
    """Represents a multi-level memory cache configuration.

    Args:
        PlatformConfig (_type_): _description_
    """

    def __init__(self,
                cache_hierarchy: List[SingleCacheConfig],
                model = None):
        self.cache_hierarchy = cache_hierarchy
        self.model = model
        assert len(self.cache_hierarchy) > 0, "At least one cache level should be present"

    def update_model(self, model):
        self.model = model

    def name(self):
        return f"[{'-'.join([cache.type for cache in self.cache_hierarchy])}]"

    def get_KV_cache_time(self, requests: List[Request]) -> float:
        """
            API call to the speculative cost model for getting the KV cache time.

            Inputs:
            requests : List of all the requests in the current cycle.

            Outputs:
            time: Time of complete the batch of the requests.
        """
        model_config = get_configs(self.model)
        KV_retrieval_time = []
        for req in requests:
            KV_size = req.get_past_context() * model_config.get_kv_size()/2**20   # Convert to MB
            time = 0
            miss_prob = 1
            for cache in self.cache_hierarchy:
                time += miss_prob * cache.hit_rate * cache.get_retrieval_time(KV_size)  # Contribution from this level
                miss_prob *= (1 - cache.hit_rate)  # Update miss probability
            KV_retrieval_time.append(time)

        return KV_retrieval_time
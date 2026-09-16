#TODO: Refactor everything below to support csv input for prefill and decodes instead of GenZ
import enum
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
from mist.Request import Request, DataMetrics, RequestMetrics
from mist.Platforms.performance_model import PerformanceModel, SplitwisePerformanceModel
from mist.Platforms.platforms import PlatformConfig, PlatformType

import os
import logging

logger = logging.getLogger(__name__)


class TracePlatformConfig(PlatformConfig):
    """Configuration for distributed Platform.

    Args:
        device: Either an exisisting system config or making a new custom system
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
    """
    
    def __init__(self,
                device: str = "h100-80gb",
                tensor_parallel_size: int = 1,
                pipeline_parallel_size: int = 1,
                engine_type: Optional[PlatformType] = PlatformType.MIXED_MACHINE,
                model: Optional[str] = None,
                # Give the relative path to the csv file
                data_path: Optional[str] = "data/perf_model.csv",
                bits: str = 'bf16',
                ) -> None:
        super().__init__(device=device, tensor_parallel_size=tensor_parallel_size, 
                         pipeline_parallel_size=pipeline_parallel_size, engine_type=engine_type, model=model, bits=bits)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, data_path)
        logger.debug(f"current working directory: {data_path}") 
        # Initialize the performance  
        self.perf_model = SplitwisePerformanceModel(data_path)


    def get_chunked_time(self, prefills: List[Request], decodes: List[Request]) -> float:
        """

        """
        #TODO: Replace this with API call to the cost models where diff KV size are present
        if self.engine_type == PlatformType.PREFILL_MACHINE or self.engine_type == PlatformType.DECODE_MACHINE:
            raise ValueError("Machine should be MIXED to process chunked requests.")
        total_tokens = 0
        batch_size = 0
        for d in decodes:
            total_tokens += 1
            batch_size += 1
        for p in prefills:
            prefill_chunk_size = p.get_current_chunk_size()
            total_tokens += prefill_chunk_size
            batch_size += 1

        chunk_time = None
        cache_key = (self.model, self.device, self.tensor_parallel_size, total_tokens)
        predictors_key = (self.model, self.device, self.tensor_parallel_size)
        # print("Total Tokens: ", total_tokens, self.model, prefill_kv_caches, decode_kv_caches)
    
        if len(prefills) == batch_size:
            chunk_time = self.perf_model.prompt_time_cache.get(cache_key)
            if chunk_time is None:
                chunk_time = float(self.perf_model.prompt_time_predictors[predictors_key](total_tokens))
                self.perf_model.prompt_time_predictors[cache_key] = float(chunk_time)
        elif len(decodes) == batch_size:
            chunk_time = self.perf_model.token_time_cache.get(cache_key)
            if chunk_time is None:
                chunk_time = float(self.perf_model.token_time_predictors[predictors_key](total_tokens))
                self.perf_model.token_time_cache[cache_key] = float(chunk_time)
        else:
            chunk_time = self.perf_model.prompt_time_cache.get(cache_key)
            if chunk_time is None:
                chunk_time = float(self.perf_model.prompt_time_predictors[predictors_key](total_tokens))
                self.perf_model.prompt_time_cache[cache_key] = float(chunk_time)
            chunk_time *= 1.1

        return chunk_time
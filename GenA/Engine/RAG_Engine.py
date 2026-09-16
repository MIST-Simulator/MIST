from .GenA_Engine import GenAEngine, EngineMetrics, EngineType
from GenA.Scheduler.scheduler import Scheduler, SchedulerConfig, BatchingMethod
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
from GenA.Platforms.platforms import PlatformConfig, PlatformType
from GenZ import get_configs
import logging
logger = logging.getLogger('GenA')
import heapq
import time
import math
import time
import random
from dataclasses import dataclass
import numpy as np
class Retrieval_Algorithm:
    """
    Represents the Retrieval Model details for an IVF-PQ based RAG Retrieval Engine.

    Parameters:
        D (int): Dimensionality of the query vectors.
        n_centroids (int): Number of centroids used for coarse quantization.
        n_probe (int): Number of centroids probed for candidate selection.
        M (int): Number of sub-quantizers for PQ.
        N_per_cluster (int): Number of candidate vectors per selected centroid.
        k (int): Number of top candidates to re-rank.
    """
    def __init__(self, D, n_centroids, n_probe, M, N_per_cluster, k):
        self.D = D
        self.n_centroids = n_centroids
        self.n_probe = n_probe
        self.M = M
        self.code_size = 8
        self.N_per_cluster = N_per_cluster
        self.k = k
        self.n_vectors = self.n_centroids * self.N_per_cluster

    def __str__(self):
        return f"Retrieval_Algorithm(D={self.D}, n_centroids={self.n_centroids}, n_probe={self.n_probe}, M={self.M}, N_per_cluster={self.N_per_cluster}, k={self.k})"
    def __repr__(self):
        return f"Retrieval_Algorithm(D={self.D}, n_centroids={self.n_centroids}, n_probe={self.n_probe}, M={self.M}, N_per_cluster={self.N_per_cluster}, k={self.k})"

    def update_D(self, D):
        self.D = D

    def estimate_db_memory_size(self):
        """
        Estimates the memory size of the retrieval engine in bytes.
        """
        ## HNSW Memory Size
        ## 1.1 * (4 * D + 8 * m) * num_vectors * Num_Graphs bytes
        # m is the algorithm parameter that controls the number of connections each node will have in a layer
        # For use cases where the data size is 1 billion vectors of 128 dimensions each and m is set to the default value of 16, the estimated amount of memory required would be:
        # 1.1 * (4 * 128 + 8 * 16) * 1 billion * 2 graphs = 1.408 TB

        ## IVF Memory Size
        ## 1.1 * (((4 * D) * n_vectors) + (4 * n_centroids * D)) bytes
        ## 1.1 * (((4 * 128) * 1 billion) + (4 * 4096 * 128)) bytes = 1.126 TB

        ## IVF-PQ Memory Size
        # PQ will encode each vector into roughly m * code_size bits plus some overhead for each vector.
        # 1.1 * ((((code_size/8) * m + overhead_per_vector) * num_vectors) + (4 * n_centroids * dimension) + (2^code_size * 4 * dimension) bytes
        # Using 1 billion vectors, dimension = 128, m = 8, code_size = 8, and n_centroids = 4096.
        # 1.1 * ((((8 / 8) * 8 + 24) * 1 billion) + (4 * 4096 * 128) + (2^8 * 4 * 128)) * 2 = 70 GB.

        # For IVF, we can tune two parameters:
        # n_centroids – Controls the granularity of the partitioning. The recommended value for this parameter is a function of the number of vectors in the index. One thing to keep in mind is that there are Faiss indexes that map to Lucene segments. There are several Lucene segments per shard and several shards per OpenSearch index. For our estimates, we assumed that there would be 100 segments per shard and 24 shards, so about 420,000 vectors per Faiss index. With this value, we estimated a good value to be 4096 and kept this constant for the experiments.
        # nprobes – Controls the number of n_centroids buckets we search. Higher values generally lead to improved recalls at the expense of increased search latencies.

        # For PQ, we can tune two parameters:
        # m – Controls the number of partitions to break the vector into. The larger this value is, the better the encoding will approximate the original, at the expense of raising memory consumption.
        # code_size – Controls the number of bits to encode a sub-vector with. The larger this value is, the better the encoding approximates the original, at the expense of raising memory consumption.
        # The max value is 8, so we kept it constant at 8 as default for all Engines.


        mem_centroids = self.n_centroids * self.D * 4
        mem_PQ_codes = self.n_vectors * ((self.code_size/8) * self.M + 24)
        mem_PQ_table = 2**self.code_size * self.D * 4

        return 1.1 * (mem_centroids + mem_PQ_codes + mem_PQ_table) / (2**30)  # Convert to GB

    def compute_IVF_PQ_query_retrieval_time(self, platform, batch_size=1):
        """
        Computes the overall compute and memory time for a query using the retrieval engine parameters
        and the provided hardware profile.

        The computation is broken into three steps:

        1. Coarse Quantization (IVF Search)
            - Compute Time: 2 * D * n_centroids / hw.flops
            - Memory Operations:
            * Read query vector: engine.D * 4 bytes (FP32)
            * Read centroids: engine.n_centroids * engine.D * 4 bytes (FP32)
            * Write top n_probe results: engine.n_probe * 8 bytes
                (assuming each result comprises an index (4 bytes) and a distance (4 bytes))
            Total Memory Time:
                T_memory_1 = (engine.D*4 + engine.n_centroids*engine.D*4 + engine.n_probe*8) / hw.memory_bw

        2. PQ Distance Computation
            - Compute Time: engine.M * engine.N_per_cluster / hw.flops
            - Memory Operations:
            * Read PQ codes: engine.N_per_cluster * engine.M bytes (assuming 1 byte per code)
            * Read PQ lookup table: 256 * engine.D * 4 bytes (since each sub-quantizer covers D/M dims, and total across M equals D)
            * Write top-k distances: engine.k * 4 bytes
            Total Memory Time:
                T_memory_2 = (engine.N_per_cluster*engine.M + 256*engine.D*4 + engine.k*4) / hw.memory_bw

        3. Re-ranking (Exact Distance Computation)
            - Compute Time: 2 * engine.D * engine.k / hw.flops
            - Memory Operations:
                * Read full-resolution vectors: engine.k * engine.D * 4 bytes (FP32)
                * Write sorted results: engine.k * 4 bytes
            Total Memory Time:
                T_memory_3 = (engine.k*engine.D*4 + engine.k*4) / hw.memory_bw

        Returns:
            total_compute_time (float): Total compute time (seconds)
            total_memory_time (float): Total memory time (seconds)
        """
        parallelize_compute_memory = False

        D = self.D
        n_centroids = self.n_centroids
        n_probe = self.n_probe
        M = self.M
        N_per_cluster = self.N_per_cluster
        k = self.k

        # Unpack hardware parameters
        P = platform.get_system_config()['Flops'] * (10**12)         # Total FLOPs per second
        BW = platform.get_system_config()['Memory_BW'] * (2**30)      # Memory bandwidth in bytes per second

        # --- Per-query computations ---
        # Step 1: Coarse Quantization (IVF Search)
        T_compute_1_per_query = (2 * D * n_centroids) / P
        mem_query = D * 4                    # Query vector in bytes
        mem_centroids = n_centroids * D * 4    # Centroids in bytes (could be loaded once)
        mem_top_centroids = n_probe * 8        # Top n_probe results (assume 8 bytes each)
        # Here we assume centroids are shared among queries if already in memory; otherwise, multiply by batch_size.
        T_memory_1_per_query = (mem_query + mem_top_centroids) / BW
        # Constant cost for centroids (loaded once per batch)
        T_memory_1_constant = mem_centroids / BW

        # Step 2: PQ Distance Computation
        T_compute_2_per_query = (M * N_per_cluster * n_probe) / P
        mem_PQ_codes = N_per_cluster * M             # PQ codes in bytes (1 byte each)
        mem_PQ_table = 256 * D * 4                     # PQ lookup table in bytes (loaded once per batch)
        mem_top_k = k * 4                              # Top-k distances (assume 4 bytes each)
        T_memory_2_per_query = (mem_PQ_codes + mem_top_k) / BW
        T_memory_2_constant = mem_PQ_table / BW


        # Step 3: Re-ranking (Exact Distance Computation)
        T_compute_3_per_query = (2 * D * k) / P
        mem_full_vectors = k * D * 4   # Full-resolution vectors in bytes
        mem_sorted_results = k * 4      # Sorted results in bytes
        T_memory_3_per_query = (mem_full_vectors + mem_sorted_results) / BW

        # --- Total times for the batch ---
        total_compute_time = batch_size * (T_compute_1_per_query + T_compute_2_per_query + T_compute_3_per_query)
        total_memory_time = (T_memory_1_constant + T_memory_2_constant +
                            batch_size * (T_memory_1_per_query + T_memory_2_per_query + T_memory_3_per_query))

        if parallelize_compute_memory:
            ## We assume maximum fusion in 3 steps of the retrieval engine, hence max of compute time of memory time
            return max(total_compute_time, total_memory_time) * 1000  # Convert to ms
        else:
            # print(f'Compute times : {batch_size * T_compute_1_per_query* 1000} + {batch_size * T_compute_2_per_query* 1000} + {batch_size * T_compute_3_per_query* 1000}')
            # print(f'Memory times : Constant:{T_memory_1_constant* 1000} + {T_memory_2_constant* 1000}, B: {batch_size * T_memory_1_per_query* 1000} + {batch_size * T_memory_2_per_query* 1000} + {batch_size * T_memory_3_per_query* 1000}')
            return (total_compute_time + total_memory_time) * 1000  # Convert to ms


class RAGEngine(GenAEngine):

    def __init__(
        self,
        engine_id:str=None,
        platform: PlatformConfig = None,
        separate_retrieval_platform: PlatformConfig = None,
        RetrievalAlgo: Retrieval_Algorithm = None,
        EmbeddingModel: str = 'e5_base',
        engine_types: List[EngineType] = [EngineType.RAG],
        doc_size: int = 512,
        max_pending_prompt_tokens: int = 8192,
        ) -> None:

        super().__init__(
            model=EmbeddingModel,
            engine_id=engine_id,
            platform=platform,
            engine_types=engine_types,
        )
        self.EmbeddingModel = EmbeddingModel
        assert self.EmbeddingModel in ['e5_large', 'e5_base', 'e5_small', 'e5_mistral_7b'], 'Invalid Embedding Model:{self.EmbeddingModel}'
        self.RetrievalAlgo = RetrievalAlgo
        self.RetrievalAlgo.update_D(get_configs(self.EmbeddingModel).hidden_size)
        self.max_pending_prompt_tokens = max_pending_prompt_tokens
        self.doc_size = doc_size
        if separate_retrieval_platform is not None:
            self.separate_retrieval_platform = separate_retrieval_platform
            self.transfer_BW = 10
        else:
            self.separate_retrieval_platform = platform
            self.transfer_BW = np.inf


    def step(self, req_step_start_time) -> None:
        """Performs one iteration and returns newly generated results.

            Overview of the step function.

        Details:
            - Step 1: Schedules the batches to be executed in the next
            iteration
            - Step 2: Calls the cost model to get the runtime
            - Step 3: Processes the model output. This mainly includes:
                - Append the relevant outputs of the requests.
                - Frees the finished sequence groups.
        """
        # if self.current_time > req_step_start_time:
            # print(f"Engine {self.engine_id} is trying to step back in time. Current time: {self.current_time}, New time: {req_step_start_time}")
        if self.current_time <= req_step_start_time:
            self.current_time = req_step_start_time
        current_batch = []
        current_batch_tokens = 0
        while self.request_queue:
            ## Push all requests that have arrived by the current time to the scheduler
            if self.request_queue[0][0] <= self.current_time and current_batch_tokens < self.max_pending_prompt_tokens:
                req_engine_entry_time, req = heapq.heappop(self.request_queue)
                current_batch.append(req)
                current_batch_tokens += req.input_len
                req.update_scheduled_time(self.current_time)
            ## If no requests have arrived by the current time, break and continue to schedule
            elif current_batch:
                break
            ## Next request to arrive in the future, so update the current time
            else:
                return self.request_queue[0][0], []


        embedding_time = self.platform.get_embedding_time(current_batch)
        ## TODO: If this becomes a requirement, make a separate engine for retrieval
        embedding_transfer_time = sum([req.input_len for req in current_batch]) * self.RetrievalAlgo.D * 4 / (self.transfer_BW * (2**30))
        retrieval_time = self.RetrievalAlgo.compute_IVF_PQ_query_retrieval_time(self.separate_retrieval_platform, len(current_batch))

        for req in current_batch:
            assert req.current_stage.value == "RAG", f"Request {req.request_id} is not in RAG stage, but in {req.current_stage} stage"
            req.stage_metrics["RAG"].rag_embedding_start_time = self.current_time
            req.stage_metrics["RAG"].rag_embedding_end_time = self.current_time + embedding_time
            req.stage_metrics["RAG"].rag_retrieval_start_time = self.current_time + embedding_time + embedding_transfer_time
            req.stage_metrics["RAG"].rag_retrieval_end_time = self.current_time + embedding_time + embedding_transfer_time +  retrieval_time
            self.logger.log_event(req.request_id, self.engine_id, str(req.current_stage)+"Embed", time=self.current_time,type= "B")
            self.logger.log_event(req.request_id, self.engine_id, str(req.current_stage)+"Embed", time=self.current_time + embedding_time ,type= "E")
            self.logger.log_event(req.request_id, self.engine_id, str(req.current_stage)+"Retrieve", time=self.current_time + embedding_time + embedding_transfer_time  ,type= "B")
            self.logger.log_event(req.request_id, self.engine_id, str(req.current_stage)+"Retrieve", time=self.current_time + embedding_time + embedding_transfer_time + retrieval_time ,type= "E")

        # print(f"Engine {self.engine_id}: Did {len(current_batch)} reqs - Embedding Time: {embedding_time}, Retrieval Time: {retrieval_time}")
        self.current_time += embedding_time + retrieval_time + embedding_transfer_time

        finished_reqs = current_batch

        for req in finished_reqs:
            req.update_rag_context(self.RetrievalAlgo.k * self.doc_size)
            req.update_engine_exit_time(self.current_time)
            self.logger.log_event(req.request_id, self.engine_id, "Engine"+str(req.current_stage), time=self.current_time,type= "E")

        return self.current_time, finished_reqs


Retrieval_AlgorithmExamples = [
    ## **1 M-32M Vectors (1M)**
    Retrieval_Algorithm(D=128, n_centroids=8192, n_probe=10, M=8, N_per_cluster=1000, k=5),
        # - **Faster:** Fewer centroids and a lower probe count reduce search time.
        # - **Lower Recall:** Since only 10 clusters are searched, some relevant results might be missed.
    Retrieval_Algorithm(D=128, n_centroids=32768, n_probe=20, M=16, N_per_cluster=2000, k=10),
        # - **Moderate Speed & Accuracy:** A good balance, with more centroids and a reasonable probe count.
    Retrieval_Algorithm(D=128, n_centroids=65536, n_probe=50, M=32, N_per_cluster=5000, k=20),
        # - **Higher Recall:** More centroids, a higher probe count, and a larger candidate pool improve accuracy.
        # - **Slower:** Needs more memory and compute for fine-grained search.
    ## **1 Billion Vectors (1B)**
    Retrieval_Algorithm(D=128, n_centroids=65536, n_probe=20, M=8, N_per_cluster=1000, k=5),
        # - **Fast retrieval:** Works well for latency-sensitive applications.
        # - **Lower Recall:** Misses some relevant candidates.
    Retrieval_Algorithm(D=128, n_centroids=131072, n_probe=40, M=16, N_per_cluster=2500, k=10),
        # - **Good tradeoff:** Improves recall while keeping search efficient.
    Retrieval_Algorithm(D=128, n_centroids=262144, n_probe=100, M=32, N_per_cluster=5000, k=20),
        # - **High recall but expensive:** Requires more compute and memory.
    ## **10 Billion Vectors (10B)**
    Retrieval_Algorithm(D=128, n_centroids=524288, n_probe=40, M=8, N_per_cluster=1000, k=5),
        # - **Optimized for speed:** Limited probing for quick lookups.
    Retrieval_Algorithm(D=128, n_centroids=1048576, n_probe=80, M=16, N_per_cluster=2500, k=10),
        # - **Stronger recall:** A larger number of centroids and moderate probing.
    Retrieval_Algorithm(D=128, n_centroids=2097152, n_probe=200, M=32, N_per_cluster=5000, k=20),
        # - **Best recall:** Maximizes accuracy but increases compute cost.
    ## **100 Billion Vectors (100B)**
    Retrieval_Algorithm(D=128, n_centroids=1048576, n_probe=50, M=8, N_per_cluster=1000, k=5),
        # - **Fastest but coarse:** Sacrifices recall for efficiency.
    Retrieval_Algorithm(D=128, n_centroids=2097152, n_probe=100, M=16, N_per_cluster=2500, k=10),
        # - **Handles large-scale search efficiently.**
    Retrieval_Algorithm(D=128, n_centroids=4194304, n_probe=400, M=32, N_per_cluster=5000, k=20),
        # - **Best accuracy:** Maximizes recall with a high number of centroids and candidates.
        # - **Massive compute cost:** Requires advanced hardware.
]

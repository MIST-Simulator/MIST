from logging import raiseExceptions
import enum
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
from mist.Request import Request, DataMetrics, RequestMetrics
import os
import csv
import itertools
import time
import threading
from GenZ import chunked_moddeling, ModelConfig, System
from GenZ.power import get_energy
import pandas as pd
import ast
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PlatformType(enum.Enum):
    PREFILL_MACHINE = enum.auto()
    DECODE_MACHINE = enum.auto()
    MIXED_MACHINE = enum.auto()

class PlatformConfig:
    """Configuration for distributed Platform.

    Args:
        device: Either an existing system config or making a new custom system
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        bits: Weight/activation precision passed to GenZ ('bf16', 'fp8', 'int8', ...).
    """

    def __init__(self,
                device: str = "h100_sxm",
                tensor_parallel_size: int = 1,
                pipeline_parallel_size: int = 1,
                sys_eff: Optional[float] = 0.75,
                engine_type: Optional[PlatformType] = PlatformType.MIXED_MACHINE,
                chunk_size: Optional[int] = 512,
                model: Optional[str] = None,
                beam_merge: Optional[bool] = False,
                decode_step_size: int = 256,
                mixed_kv_step_size: int = 32000,
                bits: str = 'bf16',
                ) -> None:
        self.device = device
        self.bits = bits
        self.system = None
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.engine_type = engine_type
        self.sys_eff = sys_eff
        self.model = model
        self.beam_merge = beam_merge
        self.all_runs = []
        self.chunk_size = chunk_size
        self.decode_step_size = decode_step_size
        self.mixed_kv_step_size = mixed_kv_step_size

        self.create_log_file()
        # Power estimation in kW
        self.power = 1000
        self.power_breakdown = {
                                    'Static': 30,
                                    'Compute': 40,
                                    'Memory': 20,
                                    'Network': 10
                                }


    def update_llm(self, model: str) -> None:
        if self.model != model:
            self.model = model
            self.create_log_file()

    def get_max_kv_tokens(self) -> int:
        model_df, model_summary = chunked_moddeling(
            model=self.model,
            prefill_kv_sizes = [],                 # [(prefill_past_kv, num_prefill)],
            decode_kv_sizes =  [1],            # [decode_past_kv]*num_decodes,
            system_name=self.get_system_config(),
            bits=self.bits,
            system_eff=self.sys_eff,
            tensor_parallel=self.tensor_parallel_size,
            pipeline_parallel=self.pipeline_parallel_size,
            model_profilling=True
        ) 
        model_size = model_summary[f'Total Weights (MB)'].values[0] # Per Chip
        kv_cache = model_summary[f'KV Cache (MB)'].values[0]        # Per Chip
        # per_kv_token_size = model_config.get_kv_size() # Number of Elements
        HBM_size_per_chip = self.get_system_memory() / 2**20        ## in MB
        max_kv_tokens = (HBM_size_per_chip - model_size) / kv_cache
        return int(max_kv_tokens*0.95)

    def get_log_prefix(self) -> str:
        return ""

    def create_log_file(self) -> None:
        if isinstance(self.model, str):
            model_name = self.model.lower()
        elif isinstance(self.model, ModelConfig):
            model_name = self.model.model.lower()
        else:
            raise ValueError("Model should be a string or ModelConfig instance.")
        record_filename = f"{self.get_log_prefix()}record_{str(model_name).replace('/', '_')}_{self.device}_TP{self.tensor_parallel_size}_PP{self.pipeline_parallel_size}_{self.bits}.db"
        record_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Platform_traces_logs"))
        os.makedirs(record_dir, exist_ok=True)
        self.record_path = os.path.join(record_dir, record_filename)

        csv_filename = f"{self.get_log_prefix()}record_{str(model_name).replace('/', '_')}_{self.device}_TP{self.tensor_parallel_size}_PP{self.pipeline_parallel_size}_{self.bits}.csv"
        self.csv_path = os.path.join(record_dir, csv_filename)
        self.decode_cache_path = os.path.join(record_dir, csv_filename.replace('.csv', '_decode_cache.csv'))
        self.mixed_cache_path  = os.path.join(record_dir, csv_filename.replace('.csv', '_mixed_cache.csv'))

        import time
        import random
        import fcntl
        if not hasattr(self, '_write_lock'):
            self._write_lock = threading.Lock()

        lock_path = self.csv_path + ".init.lock"
        with open(lock_path, 'w') as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                if not os.path.exists(self.csv_path):
                    with open(self.csv_path, 'w', newline='') as f:
                        f.write("decode_only,prefill_only,total_tokens,total_batches,total_context,prefill_tokens,prefill_context,decode_context,latency,used_energy\n")
                if not os.path.exists(self.decode_cache_path):
                    with open(self.decode_cache_path, 'w', newline='') as f:
                        f.write("n_dec,sum_kv,latency,energy\n")
                if not os.path.exists(self.mixed_cache_path):
                    with open(self.mixed_cache_path, 'w', newline='') as f:
                        f.write("chunk_size,total_kv,latency,energy\n")
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        self._load_caches()

    def _load_caches(self) -> None:
        """Populate in-memory nested dicts from the CSV so all subsequent reads are lock-free."""
        self._decode_only_cache: Dict[int, Dict[int, tuple]] = {}
        self._mixed_batch_cache: Dict[int, Dict[int, tuple]] = {}
        try:
            if os.path.exists(self.decode_cache_path):
                df = pd.read_csv(self.decode_cache_path)
                for n_dec, sum_kv, lat, nrg in zip(df['n_dec'], df['sum_kv'], df['latency'], df['energy']):
                    self._decode_only_cache.setdefault(int(n_dec), {})[int(sum_kv)] = (lat, nrg)
            if os.path.exists(self.mixed_cache_path):
                df = pd.read_csv(self.mixed_cache_path)
                for c_size, tot_kv, lat, nrg in zip(df['chunk_size'], df['total_kv'], df['latency'], df['energy']):
                    self._mixed_batch_cache.setdefault(int(c_size), {})[int(tot_kv)] = (lat, nrg)
        except Exception:
            pass

    @staticmethod
    def _find_closest_kv(inner_cache: Optional[Dict[int, tuple]], current_kv: int, tolerance: int) -> Optional[tuple]:
        """Return the cached (latency, energy) where 0 <= current_kv - cached_kv <= tolerance,
        preferring the largest cached_kv (closest match). Returns None on miss."""
        if not inner_cache:
            return None
        best_key = None
        for cached_kv in inner_cache:
            diff = current_kv - cached_kv
            if 0 <= diff <= tolerance:
                if best_key is None or cached_kv > best_key:
                    best_key = cached_kv
        return inner_cache[best_key] if best_key is not None else None

    def _persist_cache_entry(self, table: str, key1: int, key2: int, latency: float, energy: float) -> None:
        import fcntl
        path = self.decode_cache_path if table == 'decode_only_cache' else self.mixed_cache_path
        with self._write_lock:
            try:
                with open(path, 'a', newline='') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    writer = csv.writer(f)
                    writer.writerow([key1, key2, float(latency), float(energy)])
                    f.flush()
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    def log_chunked_output(self, decode_only, prefill_only, total_tokens, total_batches, total_context, prefill_tokens, prefill_context, decode_context, latency, used_energy):
        if not hasattr(self, 'csv_path'):
            self.create_log_file()
        import fcntl
        with self._write_lock:
            try:
                with open(self.csv_path, 'a', newline='') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    writer = csv.writer(f)
                    writer.writerow([
                        bool(decode_only), bool(prefill_only), int(total_tokens), int(total_batches), float(total_context), 
                        int(prefill_tokens), float(prefill_context), float(decode_context), float(latency), float(used_energy)
                    ])
                    f.flush()
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    def search_log_db(self, total_tokens, total_batches, target_context, tolerance=0.05):
        """Read from CSV using pandas. File reads map cleanly."""
        if not hasattr(self, 'csv_path') or not os.path.exists(self.csv_path):
            return 0, 0
            
        try:
            df = pd.read_csv(self.csv_path)
            # Filter rows eagerly to reduce iteration size
            df = df[(df['total_tokens'] == int(total_tokens)) & (df['total_batches'] == int(total_batches))]
            
            matching_latencies = []
            matching_energies = []
            
            for dc, lat, nrg in zip(df['decode_context'], df['latency'], df['used_energy']):
                if abs(dc - target_context) <= tolerance * target_context:
                    matching_latencies.append(lat)
                    matching_energies.append(nrg)
                    if len(matching_latencies) >= 5:
                        break
            
            if matching_latencies:
                import numpy as np
                return np.mean(matching_latencies), np.mean(matching_energies)
        except Exception:
            pass
            
        return 0, 0

    def get_chunked_time(self, prefills: List[Request], decodes: List[Request]) -> float:
        """
            API call to the speculative cost model for getting the chunked prefill/decode time.

            Inputs:
            prefills : List of all the prefill requests in the current cycle.
            decodes  : List of all the decode requests in the current cycle.

            Outputs:
            time: Time of complete the batch with chunked prefill and decode.

            Caching strategy (two-level):
            1. In-memory nested dicts (_decode_only_cache / _mixed_batch_cache) are checked
               first — reads are a plain dict lookup with no locks.
            2. On miss, chunked_moddeling is called and the result is written to both the
               in-memory dict and the SQLite DB (persists across runs).

            Decode-only key : (n_dec, sum_kv)   — tolerance = decode_step_size * n_dec
            Mixed/prefill   : (chunk_size, total_kv) — tolerance = mixed_kv_step_size
        """
        if self.engine_type == PlatformType.PREFILL_MACHINE or self.engine_type == PlatformType.DECODE_MACHINE:
            raise ValueError("Machine should be MIXED to process chunked requests.")

        decode_only  = len(prefills) == 0 and len(decodes) > 0

        if decode_only:
            n_dec  = len(decodes)
            sum_kv = sum(d.get_current_kv_length() for d in decodes)
            cached = self._find_closest_kv(
                self._decode_only_cache.get(n_dec), sum_kv,
                self.decode_step_size * n_dec
            )
            if cached is not None:
                return cached[0], cached[1]

            # Cache miss — build kv list and compute
            decode_kv_caches = []
            for d in decodes:
                if self.beam_merge:
                    req_decode_tokens  = d.get_generated_tokens_length()
                    req_prefill_tokens = d.get_current_kv_length() - req_decode_tokens
                    decode_kv_caches.append((1, d.beam_size, req_prefill_tokens, req_decode_tokens))
                else:
                    decode_kv_caches.append((d.beam_size, d.get_current_kv_length()))

            try:
                chunked_output = chunked_moddeling(
                    model=self.model,
                    prefill_kv_sizes=[],
                    decode_kv_sizes=decode_kv_caches,
                    system_name=self.get_system_config(),
                    bits=self.bits,
                    system_eff=self.sys_eff,
                    tensor_parallel=self.tensor_parallel_size,
                    pipeline_parallel=self.pipeline_parallel_size
                )
                used_energy = get_energy(chunked_output['model_df'], self.power, self.power_breakdown)
            except Exception:
                raise ValueError(f"Error in chunked_moddeling: prefill=[], decode={decode_kv_caches}")

            latency = chunked_output['Latency']
            logger.debug("Computed decode-only: n_dec=%s, sum_kv=%s, latency=%.6f", n_dec, sum_kv, latency)
            self._decode_only_cache.setdefault(n_dec, {})[sum_kv] = (latency, used_energy)
            self._persist_cache_entry('decode_only_cache', n_dec, sum_kv, latency, used_energy)
            return latency, used_energy

        else:
            # Mixed or prefill-only batch
            chunk_size = (sum(p.get_current_chunk_size() for p in prefills)
                          + sum(d.beam_size for d in decodes))
            total_kv   = (sum(p.get_current_kv_length() for p in prefills)
                          + sum(d.get_current_kv_length() for d in decodes))
            cached = self._find_closest_kv(
                self._mixed_batch_cache.get(chunk_size), total_kv,
                self.mixed_kv_step_size
            )
            if cached is not None:
                return cached[0], cached[1]

            # Cache miss — build kv lists and compute
            decode_kv_caches  = []
            prefill_kv_caches = []
            for d in decodes:
                if self.beam_merge:
                    req_decode_tokens  = d.get_generated_tokens_length()
                    req_prefill_tokens = d.get_current_kv_length() - req_decode_tokens
                    decode_kv_caches.append((1, d.beam_size, req_prefill_tokens, req_decode_tokens))
                else:
                    decode_kv_caches.append((d.beam_size, d.get_current_kv_length()))
            for p in prefills:
                prefill_kv_caches.append((p.get_current_kv_length(), p.get_current_chunk_size()))

            try:
                chunked_output = chunked_moddeling(
                    model=self.model,
                    prefill_kv_sizes=prefill_kv_caches,
                    decode_kv_sizes=decode_kv_caches,
                    system_name=self.get_system_config(),
                    bits=self.bits,
                    system_eff=self.sys_eff,
                    tensor_parallel=self.tensor_parallel_size,
                    pipeline_parallel=self.pipeline_parallel_size
                )
                used_energy = get_energy(chunked_output['model_df'], self.power, self.power_breakdown)
            except Exception:
                raise ValueError(f"Error in chunked_moddeling: prefill={prefill_kv_caches}, decode={decode_kv_caches}")

            latency = chunked_output['Latency']
            logger.debug("Computed mixed batch: chunk_size=%s, total_kv=%s, latency=%.6f", chunk_size, total_kv, latency)
            self._mixed_batch_cache.setdefault(chunk_size, {})[total_kv] = (latency, used_energy)
            self._persist_cache_entry('mixed_batch_cache', chunk_size, total_kv, latency, used_energy)
            return latency, used_energy

    def get_embedding_time(self, requests: List[Request]) -> float:
        """
            API call to the speculative cost model for getting the embedding time.

            Inputs:
            requests : List of all the requests in the current cycle.
            Outputs:
            time: Time of complete the batch.
        """
        total_tokens = 0
        prefill_kv_caches = []

        ## Entire prefill is embedding in 1 shot
        for p in requests:
            prefill_chunk_size = p.input_len
            total_tokens += prefill_chunk_size
            prefill_kv_caches.append((0,prefill_chunk_size))   ## Corresponds to KV cache

        # print("Total Tokens: ", total_tokens, self.model, prefill_kv_caches, decode_kv_caches)

        chunked_output = chunked_moddeling(model = self.model,
                                    prefill_kv_sizes = prefill_kv_caches,
                                    decode_kv_sizes = [],
                                    system_name = self.get_system_config(), bits = self.bits,
                                    system_eff=self.sys_eff,
                                    tensor_parallel = self.tensor_parallel_size,
                                    pipeline_parallel= self.pipeline_parallel_size
        )

        return chunked_output['Latency']

    def get_system_memory(self):
        if self.system:
            return self.system.off_chip_mem_size
        else:
            return self.get_system_config()['Memory_size'] * 2**30
    
    def get_system_config(self):
        if self.device in ["a100_sxm", "cerebras_cs3", "groq_lpx", "l40s", "mi355x",
                            "b200_sxm", "gb200", "h100_sxm", "mi300x", "tpu_v6e",
                            "b60", "gb300", "h200_sxm", "mi350x", "tpu_v7", "etched"]:
            self.system = System(system_name=self.device, compute_engine = "profiled-ops", bits = self.bits, collective_strategy="profiled-ops")
            return  self.system
        if self.device == "H100_GPU" or self.device == "Real_H100_GPU" or self.device == "H100":
            return {'Flops': 989, 'Memory_size': 80, 'Memory_BW': 3400, 'ICN': 450 , 'real_values':True}
        if self.device == "L40S" or self.device == "Real_L40S" or self.device == "L40S_GPU":
           return {'Flops': 733, 'Memory_size': 48, 'Memory_BW': 864, 'ICN': 32 , 'real_values':True} 
        elif self.device == "B100_GPU":
            return {'Flops': 3500, 'Memory_size': 192, 'Memory_BW': 8000, 'ICN': 900 , 'real_values':True}
        elif self.device == "B200_GPU":
            return {'Flops': 4500, 'Memory_size': 192, 'Memory_BW': 8000, 'ICN': 900 , 'real_values':True}
        elif self.device == "GH200_GPU":
            return {'Flops': 1979, 'Memory_size': 144, 'Memory_BW': 4900, 'ICN': 450 , 'real_values':True}
        elif self.device == "TPUv5e":
            return {'Flops': 197, 'Memory_size': 16, 'Memory_BW': 820, 'ICN': 50 , 'real_values':True}
        elif self.device == "A100_40GB_GPU" or self.device == "A100":
            return {'Flops': 312, 'Memory_size': 40, 'Memory_BW': 1600, 'ICN': 150 , 'real_values':True}
        elif self.device == "A100_80GB_GPU":
            return {'Flops': 312, 'Memory_size': 80, 'Memory_BW': 2039, 'ICN': 150 , 'real_values':True}
        elif self.device == "MI300X":
            return {'Flops': 1307, 'Memory_size': 192, 'Memory_BW': 5300, 'ICN': 400 , 'real_values':True}
        elif self.device == "Gaudi3":
            return {'Flops': 1600, 'Memory_size': 144, 'Memory_BW': 3675, 'ICN': 300 , 'real_values':True}
        elif self.device == "Sapphire_rapids":
            # Intel Xeon Platinum 8490H (Sapphire Rapids)
            # TFLOPs (SP): ~6.27 TFLOPs (AVX-512, 56 cores @ 3.5 GHz, 32 FLOPs/cycle/core)
            # Memory BW: 307.2 GB/s (8-channel DDR5-4800)
            # Memory Size: Up to 4TB DDR5 per socket.
            return {'Flops': 6.27, 'Memory_size': 4000, 'Memory_BW': 308, 'ICN': 32 , 'real_values':True}
        elif self.device == "Genoa":
            # AMD EPYC 9654 (Zen 4 "Genoa")
            # TFLOPs (SP): ~11.37 TFLOPs (AVX-512, 96 cores @ 3.7 GHz, 32 FLOPs/cycle/core)
            # Memory BW: 460.8 GB/s (12-channel DDR5-4800)
            # Memory Size: Up to 6TB DDR5 per socket.
            return {'Flops': 11.37, 'Memory_size': 6000, 'Memory_BW': 460.8, 'ICN': 32 , 'real_values':True}
        elif self.device == "Grace":
            # NVIDIA Grace Superchip Specifications
            # FP64 (Double Precision): ~3.55 TFLOPS per die, 7.1 TFLOPS for the entire Superchip.
            # FP32 (Single Precision): Likely ~14.2 TFLOPS (assuming 2x FP64 performance).
            # Cores: 144 ARM Neoverse V2 cores (72 cores per die, 2 dies per Superchip).
            # Threads: 144 threads (1 thread per core).
            # Memory Type: LPDDR5X with ECC (Error-Correcting Code).
            # Memory Capacity: Up to 960 GB.
            # Memory Bandwidth: Up to 768 TB/s (terabyte per second).
            # https://resources.nvidia.com/en-us-grace-cpu/data-center-datasheet?ncid=no-ncid
            return {'Flops': 14.2, 'Memory_size': 960, 'Memory_BW': 768, 'ICN': 32 , 'real_values':True}
        elif isinstance(self.device, dict):
            return self.device

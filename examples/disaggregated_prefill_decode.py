"""Disaggregated (Splitwise-style) serving: separate prefill and decode engines.

Replays a bundled production-style trace through 2 prefill + 2 decode
Llama-2-70B replicas (H100 x TP8), including KV-cache transfer between them.
"""
from pathlib import Path

import mist
from mist import PlatformConfig, TraceIngestion
from mist.Coordinator import MISTCoordinatorDisagg

MODEL = "meta-llama/Llama-2-70B"
TRACE = Path(mist.__file__).parent / "Input_requests" / "Example_Traces" / "rr_code_10.csv"

requests = TraceIngestion(ingestion_trace_file=str(TRACE)).request_queue

coordinator = MISTCoordinatorDisagg(
    request_queue_distr=requests,
    model=MODEL,
    platform=PlatformConfig(device="H100_GPU", tensor_parallel_size=8, model=MODEL),
    num_llm_engines=4,
    num_prefill_engines=2,
    num_decode_engines=2,
    logging_file=None,
)
coordinator.initial_engine_connection_df()
coordinator.run_sim()

s = coordinator.get_global_stats()
kv = [r.metrics.kv_transfer_time for r in coordinator.completed_requests if r.metrics.kv_transfer_time is not None]
print(f"served {coordinator.request_serviced}/{coordinator.request_accepted} requests")
print(f"TTFT={s.TTFT:.1f} ms  TPOT={s.TPOT:.1f} ms  p99={s.T99_latency:.1f} ms")
print(f"mean KV transfer time: {sum(kv) / len(kv):.2f} ms")

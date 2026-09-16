from pathlib import Path

import pandas as pd

import mist
from mist import TraceDistributions, TraceIngestion

TRACES = Path(mist.__file__).parent / "Input_requests"


def test_trace_distribution_arrivals_are_milliseconds(tmp_path):
    trace = tmp_path / "trace.csv"
    pd.DataFrame({
        "TIMESTAMP": ["2023-11-16 18:17:03.000", "2023-11-16 18:17:03.250", "2023-11-16 18:17:05.000"],
        "ContextTokens": [100, 200, 300],
        "GeneratedTokens": [10, 20, 30],
    }).to_csv(trace, index=False)
    queue = TraceDistributions(trace_file=str(trace), n=10).request_queue
    assert [r.metrics.arrival_time for r in queue] == [0, 250, 2000]
    assert [r.input_len for r in queue] == [100, 200, 300]


def test_trace_distribution_n_limits_requests():
    queue = TraceDistributions(trace_file=str(TRACES / "Example Traces/AzureLLMInferenceTrace_code.csv"), n=5).request_queue
    assert len(queue) == 5
    assert queue[1].metrics.arrival_time == 52  # 18:17:03.97996 -> 18:17:04.03196


def test_trace_ingestion_arrivals_are_milliseconds():
    queue = TraceIngestion(ingestion_trace_file=str(TRACES / "Example_Traces/rr_code_10.csv")).request_queue
    assert abs(queue[1].metrics.arrival_time - 125.00005357219274) < 1e-9

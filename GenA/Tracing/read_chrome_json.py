import json
from typing import List
from GenA.Request.request import Request

def read_trace_files(trace_files) -> List[List[Request]]:
    """Reads trace files and returns the trace data."""
    trace_data = []
    for trace_file in trace_files:
        with open(trace_file, "r") as f:
            trace_data.extend(json.load(f)["traceEvents"])

    for i, event in enumerate(trace_data):
        if event["ph"] == "B":
            trace_data[i]["ph"] = "E"
        elif event["ph"] == "E":
            trace_data[i]["ph"] = "B"


    return trace_data
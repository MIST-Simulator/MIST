import json
import time
import os

from mist.Request.request import RequestStage

class ChromeTracingLogger:
    def __init__(self, filename="trace.json"):
        self.filename = filename
        self.request_wise_events = []
        self.engine_wise_events = []

    def _log_event(self, req_id, engine, stage, time, type):
        """Logs a single event in Chrome Tracing format."""
        event = {
            "name": str(stage),
            "cat": str(engine),
            "ph": type,  # Begin or end event
            "ts": time,  # Convert to seconds
            "pid": 1,  # Use engine name as process ID
            "tid": req_id
        }
        self.request_wise_events.append(event)
        newevent = event.copy()
        newevent["name"] = f"{req_id} - {stage}"
        newevent["pid"] = engine  # Use engine name as process ID
        self.engine_wise_events.append(newevent)

    def log_event(self, req_id, engine, stage, time, type="B"):
        """Logs an event in Chrome Tracing format.

        Args:
            req_id (int): Unique request ID.
            engine (str): Engine name (e.g., 'GPU1', 'GPU2').
            stage (str): Stage name (e.g., 'Prefill', 'Decode-0').
            time (float): Start time in seconds.
            endtime (float): End time in seconds.
            type (str, optional): The type of event ('B' for begin, 'E' for end). Defaults to "B".
        """
        # If this event is a sub-event, log start and end separately
        if type == "B":
            self._log_event(req_id, engine, stage, time, "B")
        elif type == "E":
            self._log_event(req_id, engine, stage, time, "E")
        else:
            raise ValueError(f"Invalid event type: {type}")

    def _save_trace(self, filename, events):
        """Saves events to a JSON file in Chrome Tracing format."""
        trace_data = {
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "metadata": {
                "trace-config": "systemTraceEvents",
                "systemTraceEvents": "",
            }
        }
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(trace_data, f, indent=2)

    def save(self):
        """Saves logged events to a JSON file in correct Chrome Tracing format."""
        request_filename = self.filename.replace(".json", "_request.json")
        engine_filename = self.filename.replace(".json", "_engine.json")
        self._save_trace(request_filename, self.request_wise_events)
        self._save_trace(engine_filename, self.engine_wise_events)
        print(f"Saved trace to {self.filename}. Open in chrome://tracing")


# Example usage
if __name__ == "__main__":
    logger = ChromeTracingLogger("simulation_trace.json")

    # Simulating different requests with sub-events
    logger.log_event(1, "E1", "Prefill", time=0, type="B")
    logger.log_event(req_id=1, engine="E1", Stage="Prefill", time=1, type="E")

    logger.log_event(1, "E1", "Queue", time=0, type="B")
    logger.log_event(1, "E1", "Queue", time=0.3, type="E")
    logger.log_event(1, "E1", "Execution", time=0.3, type="B")
    logger.log_event(1, "E1", "Execution", time=1, type="E")

    # Add a parent event for decode
    logger.log_event(1, "E2", "Decode", time=1, type="B")
    logger.log_event(1, "E2", "Decode", time=2, type="E")

    for i in range(10):
        logger.log_event(1, "E2", f"Decode-{i}", time=1 + i * 0.1,type="B")
        logger.log_event(1, "E2", f"Decode-{i}", time=1.1 + i * 0.1, type="E")

    logger.log_event(1, "E3", "Postprocess", time=2, type="B")
    logger.log_event(1, "E3", "Postprocess", time=3, type="E")

    logger.log_event(2, "E1", "Prefill", time=1, type="B")
    logger.log_event(2, "E1", "Prefill", time=1.5, type="E")

    # Another nested decode sequence
    logger.log_event(2, "E2", "Decode", time=1.5, type="B")
    logger.log_event(2, "E2", "Decode", time=3, type="E")

    for i in range(15):
        logger.log_event(2, "E2", f"Decode-{i}", time=1.5 + i * 0.1, type="B")
        logger.log_event(2, "E2", f"Decode-{i}", time=1.6 + i * 0.1, type="E")

    logger.log_event(2, "E3", "Postprocess", time=3, type="B")
    logger.log_event(2, "E3", "Postprocess", time=3.5,type= "E")

    # Save and view in Chrome Tracing
    logger.save()

import enum
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple,
                    Union, Set, Deque)
from collections import deque
from dataclasses import dataclass, field

# from .base_stage import Stage
from GenZ import ModelConfig, get_configs
import math

class RequestStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED_SERVICED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status in [
            RequestStatus.FINISHED_SERVICED,
            RequestStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> Union[str, None]:
        if status == RequestStatus.FINISHED_SERVICED:
            finish_reason = "serviced"
        elif status == RequestStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class RequestStage(enum.Enum):
    INPUT_QUERY = "Input Query"
    CACHE_RETRIEVAL = "Cache Retrieval"
    RAG = "RAG"
    PREPROCESS = "Preprocess"
    PREFILL = "Prefill"
    DECODE = "Decode"
    POSTPROCESS = "Postprocess"
    OUTPUT_ANSWER = "Output Answer"

class RequestType(enum.Enum):
    STANDARD = enum.auto()
    RAG = enum.auto()
    CUSTOM = enum.auto()

@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_token_time: The time when the first token was generated.
        duration_in_queue: The time duration the request spent in the queue.
        finished_time: The time when the request was finished.
        unscheduled_duration: The time duration after first token is completed but when decode is not running
    """
    arrival_time: float
    first_scheduled_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finished_time: Optional[float] = None
    # Hanjiang: This is used for disaggregated
    kv_transfer_time:Optional[float] = None
    duration_in_queue: Optional[float] = None
    unscheduled_duration: Optional[float] = None

@dataclass
class DataMetrics:
    """Metrics associated with a data beat (Prefill/Decode).

    Attributes:
        token_num : Which token is the metric associated to.
            0 -> Prefill
            1 onwards -> Decode
        scheduled_time: The time when the beat was  scheduled.
        finished_time: The time when the beat was finished.
    """
    token_num: Optional[int]
    scheduled_time: Optional[float]
    finished_time: Optional[float] = None

@dataclass
class StageMetrics:
    """Metrics associated with a stage of a request.

    Attributes:
        engine_entry_time: The time when the stage was sent to an engine.
        scheduled_time: The time when the stage was scheduled.
        finished_time: The time when the stage was finished.
        engine_exit_time: The time when the stage was finished by the engine.
    """
    engine_id: Optional[int] = None
    engine_entry_time: Optional[float] = None
    scheduled_time: Optional[float] = None
    finished_time: Optional[float] = None
    engine_exit_time: Optional[float] = None

@dataclass
class RagStageMetrics(StageMetrics):
    """Metrics associated with a RAG stage.

    Attributes:
        rag_embedding_start_time: The time when the Embedding was sent to an engine.
        rag_embedding_end_time: The time when the Embedding was finished.
        rag_retrieval_start_time: The time when the Retrieval was sent to an engine.
        rag_retrieval_end_time: The time when the Retrieval was finished.
    """
    rag_embedding_start_time: Optional[float] = None
    rag_embedding_end_time: Optional[float] = None
    rag_retrieval_start_time: Optional[float] = None
    rag_retrieval_end_time: Optional[float] = None

class Request:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        input_len: Length of the Input Prompt.
        output_len: Expected Len of the output.
        arrival_time: The arrival time of the request.
    """

    def __init__(
        self,
        request_id: str = -1,
        input_len: int = 100,
        output_len: int = 100,
        arrival_time: float = 0,
        beam_size: Optional[int] = 1,
        # stages: List[RequestStage] = [RequestStage.PREFILL, RequestStage.DECODE, RequestStage.OUTPUT_ANSWER],
        stages: List[RequestStage] = [RequestStage.PREFILL, RequestStage.DECODE],
        past_context: Optional[int] = 0,
        engine_preference: Optional[int] = None,
    ) -> None:
        ## Set at init time
        self.request_id = request_id
        self.input_len = input_len
        self.beam_size = beam_size
        self.output_len = output_len

        self.stages = list(stages) # List of stages
        self.metrics = RequestMetrics(arrival_time=arrival_time)
        self.current_stage = self.stages[0]
        self.current_stage_index = 0
        self.stage_metrics = {}
        for i, stage in enumerate(self.stages):
            stage_key = f"{stage.value}"
            if stage == RequestStage.RAG:
                self.stage_metrics[stage_key] = RagStageMetrics()
            else:
                self.stage_metrics[stage_key] = StageMetrics()

        # Adds for tracking the engine assigned for the recorded stages
        # For Splitwise and DistServe, it has cases that require this pre-assigned
        self.engine_assigned_stage = {stage: None for stage in stages}

        ## Represents the KV cache stored in remote memory or was stored due to APC(Automatic Prefix Caching).
        self.past_context = past_context
        # For Memory Caches, this is keep track of which engine has the pas KV cache
        self.engine_preference = engine_preference

        ## RAG related variables
        self.RAGContextLength = 0

        ## Update at each engine step
        self.data = []
        self.gen_tokens = 0
        self.remaining_prefill_tokens = input_len
        self.current_scheduled_prefill = input_len
        ## Scheduler Flags
        self.status = RequestStatus.WAITING

    def add_past_context(self, context: int) -> None:
        self.past_context = context

    def get_past_context(self) -> int:
        return self.past_context

    def get_next_stage(self) -> RequestStage:
        return self.stages[self.stages.index(self.current_stage)+1]

    def get_current_stage(self) -> RequestStage:
        return self.current_stage

    def request_finished(self) -> bool:
        return self.current_stage == self.stages[-1]

    def update_engine_entry_time(self, engine_id, time: float) -> None:
        self.stage_metrics[self.current_stage.value].engine_id = engine_id
        self.stage_metrics[self.current_stage.value].engine_entry_time = time

    def get_engine_entry_time(self) -> float:
        return self.stage_metrics[self.current_stage.value].engine_entry_time

    def update_scheduled_time(self, time: float) -> None:
        self.stage_metrics[self.current_stage.value].scheduled_time = time

    def update_finished_time(self, time: float) -> None:
        self.stage_metrics[self.current_stage.value].finished_time = time

    def update_engine_exit_time(self, time: float) -> None:
        self.stage_metrics[self.current_stage.value].engine_exit_time = time

    def get_engine_exit_time(self) -> float:
        return self.stage_metrics[self.current_stage.value].engine_exit_time

    def update_request_finished_time(self, time: float) -> None:
        self.metrics.finished_time = time

    def __rps__(self) -> str:
        a = f'Req Id:{self.request_id}, Input Len:{self.input_len}, Output Len:{self.output_len}, Beams:{self.beam_size}\n'
        b = f'Current Status:{self.status}'
        return a+b

    def __str__(self):
        a = f'Req Id:{self.request_id}, Input Len:{self.input_len}, Output Len:{self.output_len}, Beams:{self.beam_size}\n'
        b = f'Current Status:{self.status}'
        return a+b


    def __lt__(self, other):  # Less than operator
        return self.request_id < other.request_id

    def __le__(self, other):  # Less than or equal to operator
        return self.request_id <= other.request_id

    def __eq__(self, other):  # Equal to operator
        return self.request_id == other.request_id

    def __gt__(self, other):  # Greater than operator
        return self.request_id > other.request_id

    def __ge__(self, other):  # Greater than or equal to operator
        return self.request_id >= other.request_id

    ## Data movement at coordinator level
    def get_data_movement_size(self, model) -> int:
        model_config = get_configs(model)
        next_stage = self.get_next_stage()
        if self.current_stage == RequestStage.PREFILL and next_stage == RequestStage.DECODE:
            # KV cache transfer
            return  self.get_current_kv_length() * model_config.get_kv_size()
        elif self.current_stage == RequestStage.DECODE and next_stage == RequestStage.POSTPROCESS:
            return self.gen_tokens * model_config.hidden_size
        elif self.current_stage == RequestStage.DECODE and next_stage == RequestStage.OUTPUT_ANSWER:
            return self.input_len * math.log(model_config.vocab_size,2)
        elif self.current_stage == RequestStage.PREPROCESS and next_stage == RequestStage.PREFILL:
            return self.input_len * model_config.hidden_size
        elif self.current_stage == RequestStage.INPUT_QUERY:
            return self.input_len * math.log(model_config.vocab_size,2)
        elif self.current_stage == RequestStage.CACHE_RETRIEVAL:
            return self.get_past_context() * model_config.get_kv_size()
        elif self.current_stage == RequestStage.RAG:
            return self.get_rag_context() * model_config.hidden_size
        else:
            raise ValueError(f"Data movement not defined for {self.current_stage} to {next_stage}")

    ### Prefill and Decode related functions
    def get_current_kv_length(self) -> int:
        if self.current_stage == RequestStage.DECODE:
            return self.past_context + self.input_len + self.gen_tokens*self.beam_size
        elif self.current_stage == RequestStage.PREFILL:
            return self.past_context + self.input_len - self.remaining_prefill_tokens - self.current_scheduled_prefill
        else:
            return 0

    def get_generated_tokens_length(self) -> int:
        return self.gen_tokens*self.beam_size

    def get_current_req_storage(self, model) -> int:
        model_config = get_configs(model)
        if self.current_stage == RequestStage.DECODE:
            return  self.get_current_kv_length() * model_config.get_kv_size()
        elif self.current_stage == RequestStage.PREFILL:
            return self.input_len * model_config.hidden_size
        else:
            return 0

    def add_token(self, data_beat)-> None:
        self.data.append(data_beat)
        self.gen_tokens += 1

    def get_num_new_tokens(self,
                            enable_chunking: bool= False,
                            chunk_size:int = None) -> int:
        """Get the number of new tokens to be computed.

        Returns:
            The new number of tokens to be computed. I.e., BeamSize for decode, or
            the prompt size for prefill.
        """
        if self.token_generation_finished():
            return 0
        elif self.current_stage == RequestStage.DECODE:
            return self.beam_size
        elif self.current_stage == RequestStage.PREFILL:
            if enable_chunking:
                assert chunk_size, "When chunking is enabled, chunk size should be an int."
                return min(chunk_size, self.remaining_prefill_tokens)
            else:
                return self.input_len
        else:
            print(f"When in {self.current_stage} don't schedule the request.")
            return 0

    def current_scheduled(self, tokens_scheduled: int) -> None:
        """Update the number of tokens scheduled for the current request."""
        if self.current_stage == RequestStage.PREFILL:
            self.remaining_prefill_tokens -= tokens_scheduled
            self.current_scheduled_prefill = tokens_scheduled

    def get_current_chunk_size(self) -> int:
        return self.current_scheduled_prefill

    def token_generation_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def is_prefill(self) -> bool:
        return (self.current_stage == RequestStage.PREFILL) and (self.remaining_prefill_tokens > 0)

    ### RAG related functions
    def update_rag_context(self, ContextLength:int) -> None:
        self.RAGContextLength = ContextLength
        self.input_len += ContextLength
        self.remaining_prefill_tokens += ContextLength
        self.current_scheduled_prefill += ContextLength

    def get_rag_context(self) -> int:
        return self.RAGContextLength

    ## KV Retrieval related functions
    def get_engine_preference(self) -> int:
        return self.engine_preference
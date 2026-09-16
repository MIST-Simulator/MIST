## Adopted from vLLM
import enum
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple,
                    Union, Set, Deque)
from collections import deque
from dataclasses import dataclass, field

from GenA.Scheduler.policy import Policy, PolicyFactory
from GenA.Request import RequestStatus, RequestStage, RequestMetrics, Request, DataMetrics
import time
import warnings


class BatchingMethod(enum.Enum):
    STATIC = enum.auto()
    CONTINUOUS = enum.auto()
    MIXED = enum.auto()
    CHUNKED = enum.auto()
    DISAGGREGATED = enum.auto()


class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_batch_size: Maximum number of sequences to be processed in a single
            iteration.
        max_total_tokens_per_sample: Maximum length of a sequence (including prompt
            and generated text).
        delay_factor: Apply a delay (of delay factor multiplied by previous
            prompt latency) before scheduling next prompt.
        batching_method: Apply the batching method for the scheduler according to the policy.
            Choices are [STATIC, CONTINUOUS, MIXED, CHUNKED, DISAGGREGATED]
    """

    def __init__(self,
                max_num_batched_tokens: Optional[int]=128000,
                max_batch_size: Optional[int]=128,
                max_total_tokens_per_sample: Optional[int]=128000,
                max_batched_kv_size: Optional[int]=1000*1000,
                delay_factor: Optional[float] = 0.0,
                batching_method: Optional[BatchingMethod] = BatchingMethod.CONTINUOUS,
                chunk_size: Optional[int] = 512) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # If max_total_tokens_per_sample is too short, use 2048 as the default value
            # for higher throughput.
            self.max_num_batched_tokens = max(max_total_tokens_per_sample, 128*1024)

        self.max_batch_size = max_batch_size
        self.max_batched_kv_size = max_batched_kv_size
        self.max_total_tokens_per_sample = max_total_tokens_per_sample
        self.delay_factor = delay_factor


        if isinstance(batching_method, int):
            self.batching_method = BatchingMethod(batching_method)
        elif isinstance(batching_method, BatchingMethod):
            self.batching_method = batching_method
        else:
            raise ValueError('Invalid Batching Method')

        self.chunk_size = chunk_size
        if self.batching_method == BatchingMethod.CHUNKED:
            if self.chunk_size is None:
                raise ValueError(
                    "chunk_size must be provided when using the CHUNKED batching method.")
            self.max_num_batched_tokens = self.chunk_size

        ## Verify the configuration
        if self.max_num_batched_tokens < self.max_batch_size:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_batch_size "
                f"({self.max_batch_size}).")

        if self.batching_method not in BatchingMethod:
            raise ValueError(
                f"batching_method must be one of {list(BatchingMethod)}.")

@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    """
    kv_cache_budget: int = 1000000
    token_budget: int = 200000
    max_batch_size: int = 128
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_kv_cache: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0
    _num_batched_kv_cache: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int, new_kv_cache: int) -> bool:
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_batch_size
                and self.num_batched_kv_cache + new_kv_cache <= self.kv_cache_budget)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            return

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def add_kv_cache(self, req_id: str, kv_cache: int):
        if req_id in self._request_ids_kv_cache:
            return

        self._request_ids_kv_cache.add(req_id)
        self._num_batched_kv_cache += kv_cache

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_batched_kv_cache(self):
        return self._num_batched_kv_cache

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs

@dataclass
class SchedulerState:
    """Metrics to keep track of the scheduler's stastics with time.

    Attributes:
        time : Engine time at which the state is recorded
        waiting_len: The length of waiting queue
        running_len: The length of running queue
        finished_len: The length of finished queue
        prefills_sched: Number of prefills scheduled
        decodes_sched: Number of decodes scheduled
    """
    time: Optional[int] = 0
    waiting_len: Optional[int] = 0
    running_len: Optional[int] = 0
    finished_len: Optional[int] = 0
    prefills_sched: Optional[int] = 0
    decodes_sched: Optional[int] = 0

class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.scheduler_config = scheduler_config

        ## TODO: Make queues private and write api call on which functions can push/pop from queue
        # Requests in the WAITING state.
        # Contain new to request to be sent to prefill.
        self.waiting: Deque[Request] = deque()
        # Requests in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[Request] = deque()

        # Requests in finished state.
        # Contains requests that have completed execution
        self.finished: Deque[Request] = deque()

        self.state_tracker: List[SchedulerState] = []

        # Time at previous scheduling step
        self.prev_time = 0.0
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        # Whether the previous prompt was delayed
        self.logger = None

    def track_scheduler_state(self, time: int, prefills: List[Request], decodes: List[Request]) -> None:

        current_state = SchedulerState(
                                time= time,
                                waiting_len=len(self.waiting),
                                running_len=len(self.running),
                                finished_len=len(self.finished),
                                prefills_sched=len(prefills),
                                decodes_sched=len(decodes)
                                )

        self.state_tracker.append(current_state)

    def add_request(self, req: Request) -> None:
        # Add sequence groups to the waiting queue.
        if req.current_stage == RequestStage.DECODE:
            self.running.append(req)
        else:
            self.waiting.append(req)

    def end_request(self, req: Request) -> None:
        # Add sequence groups to the waiting queue.
        if len(self.finished) == 0:
            self.finished.append(req)
        elif self.finished[0].request_id < req.request_id :
            self.finished.append(req)
        else:
            self.finished.appendleft(req)

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_prompt_limit(self) -> int:
        prompt_limit = min(self.scheduler_config.max_total_tokens_per_sample,
                            self.scheduler_config.max_num_batched_tokens)
        return prompt_limit

    def _schedule_from_queue(
            self,
            current_time: float,
            queue: deque,
            budget: SchedulingBudget,
            policy: Optional[Policy] = None,
            enable_chunking: bool = False
            )-> Tuple[deque, List[Request]] :
        """
        It schedules requests as long as it fits `budget` from the scheduling config.
        The input arguments `budget` is updated based on request served.

        Args:
            queue: The queue that contains requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.

        Returns:
            A tuple of remaining queue after scheduling and
            List of Request to run.
        """

        request_queue: List[Request] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        # Queue is input as waiting queue
        queue = deque([s for s in queue])

        ## If we need to allow delay, then based on schedulerConfig
        # while self._passed_delay(current_time) and queue:

        ## If there is a particular policy in effect, sort the queue using that.
        if policy:
            queue = policy.sort_by_priority(current_time, queue)
        while queue and budget.remaining_token_budget() > 0 and budget.num_curr_seqs < self.scheduler_config.max_batch_size and budget.num_batched_kv_cache < self.scheduler_config.max_batched_kv_size:
            ## Get first
            req = queue[0]

            num_new_tokens =  self._get_num_new_tokens(req,budget, enable_chunking)
            prompt_limit = self._get_prompt_limit()
            if num_new_tokens > prompt_limit:
                warnings.warn(
                    f"For Req ID: {req.request_id}, Input prompt ({req.remaining_prefill_tokens} tokens) is too long "
                    f"and exceeds limit of {prompt_limit}", UserWarning)
                req.status = RequestStatus.FINISHED_IGNORED
                self.end_request(req)
                # self.finished.append(req)
                queue.popleft()
                continue

            if num_new_tokens > 0:
                request_queue.append(req)
                req.current_scheduled(num_new_tokens)
                queue.popleft()
                budget.add_num_batched_tokens(req.request_id, num_new_tokens)
                budget.add_kv_cache(req.request_id, req.get_current_kv_length())
                budget.add_num_seqs(req.request_id, 1)
                # print("Scheduling: ", req.request_id, "Num Tokens: ", num_new_tokens, "Num Seqs: ", 1, "KV Cache: ", req.get_current_kv_length(), "Total Tokens: ", budget.num_batched_tokens, "Total Seqs: ", budget.num_curr_seqs, "Total KV Cache: ", budget.num_batched_kv_cache)
            else:   ## TODO:This doesn't accommodate the next request even if it can be scheduled.
                break

            if ((budget.remaining_token_budget() <= 0) or
            (budget.num_curr_seqs >= self.scheduler_config.max_batch_size) or
            (budget.num_batched_kv_cache >= self.scheduler_config.max_batched_kv_size)):
                # print("Budget exhausted: ", budget.remaining_token_budget(), budget.num_curr_seqs, budget.num_batched_kv_cache)
                break

        return queue, request_queue

    def schedule(self,current_time) -> Tuple[List[Request], List[Request], List[Request]]:
        prefills, decodes = [], []
        remaining_waiting = self.waiting
        # Include running requests to the budget.

        # Hanjiang: Should we move this to the init
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_batch_size=self.scheduler_config.max_batch_size,
            kv_cache_budget=self.scheduler_config.max_batched_kv_size,
        )

        if ((self.scheduler_config.batching_method == BatchingMethod.STATIC and len(self.running) == 0) or
            (self.scheduler_config.batching_method == BatchingMethod.CONTINUOUS) or
            (self.scheduler_config.batching_method == BatchingMethod.MIXED) or
            (self.scheduler_config.batching_method == BatchingMethod.DISAGGREGATED)) :
            ## With static batching run Prefill if no decodes present
            ## With continuous batching if new prefill has come, then schedule it and add to decode queue.
            remaining_waiting, prefills = self._schedule_from_queue(current_time, self.waiting, budget)

        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")

        ## Decode is a list of running decodes
        remaining_running, decodes = (self.running, [])
        # Don't schedule decodes if prefills are scheduled.
        if ((self.scheduler_config.batching_method == BatchingMethod.STATIC and len(prefills) == 0) or
            ((self.scheduler_config.batching_method == BatchingMethod.CONTINUOUS) and len(prefills) == 0) or
            (self.scheduler_config.batching_method == BatchingMethod.MIXED) or
            (self.scheduler_config.batching_method == BatchingMethod.CHUNKED) or
            (self.scheduler_config.batching_method == BatchingMethod.DISAGGREGATED)):
            # if len(self.running) > 1:
                # print(f"running decode size: {len(self.running)}")
            # In continuous batching, we can schedule decodes along with entire prefills
            remaining_running, decodes = self._schedule_from_queue(
                current_time,
                self.running,
                budget,
                fcfs_policy)

        if self.scheduler_config.batching_method == BatchingMethod.CHUNKED:
            # print("Chunked Budget: ", budget.remaining_token_budget())
            remaining_waiting, prefills = self._schedule_from_queue(current_time, self.waiting, budget, fcfs_policy, enable_chunking=True)

        if len(prefills) > 0:
            for p in prefills:
                if p.metrics.first_scheduled_time is None:
                    p.metrics.first_scheduled_time = current_time
                    p.metrics.duration_in_queue = current_time - p.metrics.arrival_time

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_batch_size, f"Current Seqs: {budget.num_curr_seqs}, Max Seqs: {self.scheduler_config.max_batch_size}"

        # Update waiting requests.
        self.waiting = remaining_waiting
        # Update requesting in running but that are not yet scheduled
        self.running = remaining_running

        self.track_scheduler_state(current_time, prefills, decodes)


        return prefills, decodes


    def process_scheduled(self,
                        prefill: List[Request],
                        decodes: List[Request],
                        step_start_time: float,
                        step_end_time: float
                    ) -> List[Tuple[Request, int]]:
        """
        Process the queues based on the requests that were processed.
        Args:
            prefill (List[Request]): List of processed prefill requests
            decodes (List[Request]): List of processed decode requests
        Output:
            finished_reqs : List of requests that have finished and corresponding destination type.
        """
        finished_req = []
        for req in prefill + decodes:
            if req.is_prefill() == False:
                dataBeat = DataMetrics(token_num=req.gen_tokens,
                                    scheduled_time=step_start_time,
                                    finished_time=step_end_time)
                req.data.append(dataBeat)
                req.gen_tokens += 1

        running_queue = self.running

        for d in decodes:
            if d.gen_tokens >= d.output_len:
                d.status = RequestStatus.FINISHED_SERVICED
            if d.token_generation_finished() :
                self.end_request(d)
                finished_req.append(d)
            else:
                running_queue.append(d)

        ## All the requests in prefill list are assumed to be done in the next
        ## cycle and thus goes to decode queue.

        for p in prefill:
            # print("Prefill: ", p.request_id, p.remaining_prefill_tokens)
            if p.remaining_prefill_tokens > 0:
                p.current_stage = RequestStage.PREFILL
                p.status = RequestStatus.RUNNING
                self.waiting.append(p)
            elif self.scheduler_config.batching_method == BatchingMethod.DISAGGREGATED:
                self.end_request(p)
                p.current_scheduled(0)
                finished_req.append(p)
            else:
                p.current_stage = RequestStage.DECODE
                p.status = RequestStatus.RUNNING
                running_queue.append(p)

        self.running = running_queue

        return finished_req


    def _get_num_new_tokens(self, req: Request,
                            budget: SchedulingBudget,
                            enable_chunking: bool) -> int:
        """Get the next new tokens to compute for a given sequence group
            that's in a given `status`.

        Returns 0 if the new token cannot be computed due to token budget.
        """

        num_new_tokens = req.get_num_new_tokens(enable_chunking=enable_chunking, chunk_size=self.scheduler_config.chunk_size)
        # assert num_new_tokens > 0, f"request: {req.request_id}, request stage: {req.stage}"
        # print("Num New Tokens: ", num_new_tokens, "Remaining Token Budget: ", budget.remaining_token_budget())
        if budget.remaining_token_budget() > 0:
            if num_new_tokens <=  budget.remaining_token_budget():
                return num_new_tokens
            elif enable_chunking:
                return budget.remaining_token_budget()
            else:
                return 0
        else:
            return 0

    def has_unfinished_reqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0

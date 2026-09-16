from .MIST_Engine import MISTEngine, EngineMetrics, EngineType
from mist.Scheduler.scheduler import Scheduler, SchedulerConfig, BatchingMethod
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
from mist.Platforms.platforms import PlatformConfig, PlatformType
from mist.Request import Request, DataMetrics, RequestMetrics, RequestStage
import logging
logger = logging.getLogger('mist')
import heapq
import time
from mist.Coordinator.global_router import LoadTypes

class LLMEngine(MISTEngine):

    def __init__(
        self,
        model: str,
        scheduler_config: SchedulerConfig,
        engine_id:str=None,
        platform: PlatformConfig = None,
        engine_types: List[EngineType] = [EngineType.PREFILL, EngineType.DECODE],
        max_pending_prompt_tokens: int = 8192,
        step_size: int = 256,    # Number of tokens to process without recalculating time
        engine_kv_step_size: int = 32000,  # Number of KV tokens to process without recalculating time
        decode_only_cache: Optional[Dict[int, Dict[int, tuple]]] = None,
        mixed_batch_cache: Optional[Dict[int, Dict[int, tuple]]] = None,
        ) -> None:
        
        
        platform.update_llm(model)

        max_kv_tokens = platform.get_max_kv_tokens()
        scheduler_config.max_batched_kv_size = max_kv_tokens
        if scheduler_config.max_num_batched_tokens > max_kv_tokens:
            print(f"Warning: max_num_batched_tokens {scheduler_config.max_num_batched_tokens} is greater than max_kv_tokens {max_kv_tokens}. Setting max_num_batched_tokens to max_kv_tokens.")
            scheduler_config.max_num_batched_tokens = max_kv_tokens

            
        super().__init__(
            model=model,
            scheduler=Scheduler(scheduler_config),
            engine_id=engine_id,
            platform=platform,
            engine_types=engine_types,
        )


        ## TODO: Values added for splitwise
        self.max_pending_prompt_tokens = max_pending_prompt_tokens
        self.scheduler.logger = self.logger
        self.batched_engine = True
        self.current_load = 0
        self.current_scheduled = {RequestStage.PREFILL: [], RequestStage.DECODE: []}

        dummy_req = Request(input_len=self.scheduler.scheduler_config.chunk_size, output_len=0, request_id="dummy",)
        dummy_req.current_scheduled(dummy_req.get_num_new_tokens(self.scheduler.scheduler_config.batching_method == BatchingMethod.CHUNKED, self.scheduler.scheduler_config.chunk_size))
        self.prefill_step_time, _ = self.platform.get_chunked_time([dummy_req], [])

        dummy_req = Request(input_len=self.scheduler.scheduler_config.chunk_size, output_len=0, request_id="dummy", stages=[RequestStage.DECODE])
        self.per_decode_step, _ = self.platform.get_chunked_time([], [dummy_req])
        self.current_storage = 0

        ## Memory Load tracker
        self.memory_load = []

        ## TEMP
        self.scheduling_time = 0
        self.platform_time = 0

        ## For faster execution — nested dictionary-based lookup caches
        # _decode_only_cache[num_decodes][sum_decode_kv] = (runtime, energy)
        # Pass a dict explicitly to share a cache between engines on the same platform.
        self._decode_only_cache: Dict[int, Dict[int, tuple]] = {} if decode_only_cache is None else decode_only_cache
        # _mixed_batch_cache[chunk_size][total_kv] = (runtime, energy)
        self._mixed_batch_cache: Dict[int, Dict[int, tuple]] = {} if mixed_batch_cache is None else mixed_batch_cache
        self.engine_step_size = step_size
        self.engine_kv_step_size = engine_kv_step_size
    @staticmethod
    def _find_closest_kv(inner_cache: Optional[Dict[int, tuple]], current_kv: int, tolerance: int) -> Optional[tuple]:
        """Find a cached entry where 0 <= current_kv - cached_kv <= tolerance.
        Returns the (runtime, energy) tuple of the closest match, or None."""
        if not inner_cache:
            return None
        best_key = None
        for cached_kv in inner_cache:
            diff = current_kv - cached_kv
            if 0 <= diff <= tolerance:
                if best_key is None or cached_kv > best_key:
                    best_key = cached_kv
        return inner_cache[best_key] if best_key is not None else None

    def engine_idle(self) -> bool:
        if self.request_queue or self.scheduler.has_unfinished_reqs():
            return False
        else:
            return True

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
        if self.current_time > req_step_start_time:
            raise ValueError(f"Engine {self.engine_id} is ahead of the current time. Engine time: {self.current_time}, Current time: {req_step_start_time}")
        # if self.engine_id == 1:
        #     print(f"Engine {self.engine_id}. Current time: {self.current_time}, New time: {req_step_start_time}")
        if self.current_time <= req_step_start_time:
            self.current_time = req_step_start_time

        # Current load and storage count
        self.current_load = 0
        self.current_scheduled = {RequestStage.PREFILL: [], RequestStage.DECODE: []}
        self.current_storage = 0
        while self.request_queue:
            ## Push all requests that have arrived by the current time to the scheduler
            if self.request_queue[0][0] <= self.current_time:
                req_engine_entry_time, req = heapq.heappop(self.request_queue)
                self.scheduler.add_request(req)
            ## If no requests have arrived by the current time, break and continue to schedule
            elif self.scheduler.has_unfinished_reqs():
                break
            ## Next request to arrive in the future, so update the current time
            else:
                return self.request_queue[0][0], []

        sch_time = self.current_time
        # if self.can_handle_task(EngineType.PREFILL) or self.can_handle_task(EngineType.DECODE):
        start_time = time.time()
        prefills, decodes = self.scheduler.schedule(self.current_time)
        end_time = time.time()
        self.scheduling_time += (end_time - start_time)
        # Check the load of the current schduled requests
        # Used to update the engine load and engine storage
        dup_reqs = set()
        for req in prefills + decodes:
            dup_reqs.add(req.request_id)
            self._update_token_load_storage(req)
            if req.get_current_stage() == RequestStage.PREFILL:
                self.current_load += req.input_len
                self.current_scheduled[RequestStage.PREFILL].append(req)
            elif req.get_current_stage() == RequestStage.DECODE:
                self.current_scheduled[RequestStage.DECODE].append(req)
                self.current_load += 1
            else:
                self.current_load += 0
            self.logger.log_event(req.request_id, self.engine_id, str(req.current_stage)+str(req.gen_tokens), time=self.current_time,type= "B")

        for req_list in (self.scheduler.waiting, self.scheduler.running):
            for req in req_list:
                if req.request_id not in dup_reqs:
                    self._update_token_load_storage(req)
                else:
                    print(f"unfinished dup reqs: {req.request_id}")


        start_time = time.time()
        if len(prefills) == 0 and len(decodes) > 0:
            # Decode-only batch: nested cache[num_decodes][sum_decode_kv]
            n_dec = len(decodes)
            sum_kv = sum(d.get_current_kv_length() for d in decodes)
            cached = self._find_closest_kv(self._decode_only_cache.get(n_dec), sum_kv, self.engine_step_size * n_dec)
            if cached is not None:
                machine_runtime, energy = cached
            else:
                machine_runtime, energy = self.platform.get_chunked_time(prefills, decodes)
                self._decode_only_cache.setdefault(n_dec, {})[sum_kv] = (machine_runtime, energy)
                # print(f"Calculating time for engine {self.engine_id=} {self.current_time=}, #Prefills=0, #Decodes={n_dec}, {machine_runtime=}, {sum_kv=}")
        else:
            # Mixed or prefill-only batch: nested cache[chunk_size][total_kv]
            chunk_size = sum(p.get_current_chunk_size() for p in prefills) + sum(d.beam_size for d in decodes)
            total_kv = sum(p.get_current_kv_length() for p in prefills) + sum(d.get_current_kv_length() for d in decodes)
            cached = self._find_closest_kv(self._mixed_batch_cache.get(chunk_size), total_kv, self.engine_kv_step_size)
            if cached is not None:
                machine_runtime, energy = cached
            else:
                machine_runtime, energy = self.platform.get_chunked_time(prefills, decodes)
                self._mixed_batch_cache.setdefault(chunk_size, {})[total_kv] = (machine_runtime, energy)
                # print(f"Calculating time for engine {self.engine_id=} {self.current_time=}, #Prefills={len(prefills)}, #Decodes={len(decodes)}, {machine_runtime=}, {chunk_size=}, {total_kv=}")
        end_time = time.time()
        self.platform_time += (end_time - start_time)
        self.current_time += machine_runtime
        # self.memory_load.append((machine_runtime, sum([req.get_current_kv_length() + (req.get_current_chunk_size() if req.current_stage == RequestStage.PREFILL else req.beam_size) for req in prefills + decodes]) + sum([req.get_current_kv_length() for req in self.scheduler.running])))
        # if self.engine_id == 1:
            # print(f"Engine {self.engine_id} machine time: {machine_runtime}, {[(req.get_current_kv_length(), (req.get_current_chunk_size() if req.current_stage == RequestStage.PREFILL else req.beam_size)) for req in prefills + decodes]}")
        self.energy_consumed += energy
        for req in prefills + decodes:
            self.logger.log_event(req.request_id, self.engine_id, str(req.current_stage)+str(req.gen_tokens), time=self.current_time,type= "E")

        finished_reqs = self.scheduler.process_scheduled(prefills, decodes, sch_time, self.current_time)

        for req in finished_reqs:
            req.update_engine_exit_time(self.current_time)
            self.logger.log_event(req.request_id, self.engine_id, "Engine"+str(req.current_stage), time=self.current_time,type= "E")
        # if self.engine_id == 1:
        #     print(f"Engine end time: {self.current_time}")
        return self.current_time, finished_reqs

    # TODO: For splitwise, Return the load of current engine
    # Below is for splitwise
    # def tokens_storage(self, time):
    #     """ 
    #         This counts how many total tokens is currently stored on the machine 
    #         This is normally expressed as kv cache storage
    #     """
    #     if time < self.current_time:
    #         tokens_storage = self.current_storage
    #     else:
    #         tokens_storage = 0
    #     for (_,req) in self.request_queue:
    #         if (req.get_current_stage() == RequestStage.DECODE):
    #             tokens_storage += 1
    #         elif (req.get_current_stage() == RequestStage.PREFILL):
    #             tokens += req.input_len
    #         else:
    #             # No need for other stage
    #             tokens += 0
    #     for req in self.scheduler.waiting:
    #         if (req.get_current_stage() == RequestStage.DECODE):
                
    #         elif (req.get_current_stage() == RequestStage.PREFILL):
    #             tokens += req.input_len
    #         else:
    #             # No need for other stage
    #             tokens += 0
    #     for req in self.scheduler.running:
    #         if (req.get_current_stage() == RequestStage.DECODE):
    #             tokens += 1
    #         elif (req.get_current_stage() == RequestStage.PREFILL):
    #             tokens += req.input_len
    #         else:
    #             # No need for other stage
    #             tokens += 0
    #     return tokens

    def _update_token_load_storage(self,req:Request):
        if req.get_current_stage() == RequestStage.PREFILL:
            # print(f"request id {req.request_id}")
            self.current_load += req.input_len
            self.current_storage += req.get_current_req_storage(self.model)
        elif req.get_current_stage() == RequestStage.DECODE:
            self.current_load += 1
            self.current_storage += req.get_current_req_storage(self.model)
            # print(f"request id {req.request_id}")


    def tokens_load(self, time):
        """ 
            No need to differentiate Decode and Prefill (because the assignment of Engines) 
            Includes both tokens in the pending queue and also the tokens that already scheduled
        """
        if time < self.current_time:
            tokens = self.current_load
        else:
            tokens = 0
        for (_,req) in self.request_queue:
            if (req.get_current_stage() == RequestStage.DECODE):
                tokens += 1
            elif (req.get_current_stage() == RequestStage.PREFILL):
                tokens += req.input_len
            else:
                # No need for other stage
                tokens += 0
        return tokens
    def engine_storage(self, time):
        """ 
            Similar to tokens_load(), this is used to track the memory storage used 
            for the tokens laod
        """
        if time < self.current_time:
            storage = self.current_storage
        else:
            storage = 0
        for (_,req) in self.request_queue:
            if (req.get_current_stage() == RequestStage.DECODE):
                tokens += 1
            elif (req.get_current_stage() == RequestStage.PREFILL):
                tokens += req.input_len
            else:
                # No need for other stage
                tokens += 0
        return tokens

    def get_load(self, time, type):
        tokens = 0
        if type == LoadTypes.PREFILL_TOKENS:
            for req in self.scheduler.waiting:
                if (req.get_current_stage() == RequestStage.PREFILL):
                    tokens += req.remaining_prefill_tokens
                    # print(f"Req:{req.request_id} in waiting: {req.remaining_prefill_tokens}")
            if time < self.current_time:
                for req in  self.current_scheduled[RequestStage.PREFILL]:
                    tokens += req.current_scheduled_prefill
                    # print(f"Req:{req.request_id} in current_scheduled : {req.current_scheduled_prefill}")
            for (_,req) in self.request_queue:
                if (req.get_current_stage() == RequestStage.PREFILL):
                    tokens += req.input_len
                    # print(f"Req:{req.request_id} in request_queue")
        elif type == LoadTypes.DECODE_TOKENS:
            for req in self.scheduler.running:
                if (req.get_current_stage() == RequestStage.DECODE):
                    tokens += 1
                    # print(f"Req:{req.request_id} in running")
            if time < self.current_time:
                for req in self.current_scheduled[RequestStage.DECODE]:
                    tokens += 1
                    # print(f"Req:{req.request_id} in current_scheduled")
            for (_,req) in self.request_queue:
                if (req.get_current_stage() == RequestStage.DECODE):
                    tokens += 1
                    # print(f"Req:{req.request_id} in request_queue")
        elif type == LoadTypes.TOKEN_PROCESSING:
            tokens = self.get_load(time, LoadTypes.PREFILL_TOKENS) + self.get_load(time, LoadTypes.DECODE_TOKENS)
        elif type == LoadTypes.KV_CACHE_SIZE:
            for req in self.scheduler.running:
                tokens += req.get_current_kv_length()
            for req in  self.scheduler.waiting:
                tokens += req.get_current_kv_length()
            for req in self.current_scheduled[RequestStage.PREFILL]:
                tokens += req.get_current_kv_length()
            for req in self.current_scheduled[RequestStage.DECODE]:
                tokens += req.get_current_kv_length()
            for (_,req) in self.request_queue:
                tokens += req.get_current_kv_length()
        elif type == LoadTypes.ORACLE_DECODE:
            tokens = self.get_load(time, LoadTypes.PREFILL_TOKENS) / self.scheduler.scheduler_config.chunk_size
            for req in self.scheduler.waiting:
                tokens += req.output_len
            for req in self.scheduler.running:
                tokens += req.output_len
            if time < self.current_time:
                for req in  self.current_scheduled[RequestStage.DECODE]:
                    tokens += req.output_len
                    # print(f"Req:{req.request_id} in current_scheduled : {req.current_scheduled_prefill}")
            for (_,req) in self.request_queue:
                tokens += req.output_len
        elif type == LoadTypes.SIM_TIME:
            time = 0
            prefills, decodes = [], []
            for req in  self.scheduler.waiting:
                prefills.append(req)
            for req in self.scheduler.running:
                decodes.append(req)
            for req in self.request_queue:
                if req[1].get_current_stage() == RequestStage.PREFILL:
                    prefills.append(req[1])
                elif req[1].get_current_stage() == RequestStage.DECODE:
                    decodes.append(req[1])
            prefill_steps = self.get_load(time, LoadTypes.PREFILL_TOKENS) / self.scheduler.scheduler_config.chunk_size
            max_decode_steps = max([0] + [(req.output_len - req.gen_tokens) for req in decodes] + [req.output_len for req in prefills])
            if prefill_steps and prefills:
                time += prefill_steps * self.prefill_step_time
                # print(f"Prefill time: {prefill_steps} * {self.prefill_step_time} = {time}")

            time += max(0, max_decode_steps - prefill_steps) * self.per_decode_step
            # print(f"Num Curr Decodes :{len(decodes)}, Decode steps: {max_decode_steps} , per step: {self.per_decode_step}")

            tokens = time
        # print(f"Engine {self.engine_id} load of type {type} at time {time}  = {tokens}")
        return tokens

    def check_is_overload(self, time):
        # if self.tokens_load(time) > self.max_pending_prompt_tokens:
        #     return True
        # else:
        return False
    
    def check_is_overload_with_prefill(self, time, request):
        """ Check the loading status of the engine """
#        print(f"request: {request.request_id} in engine: {self.engine_id} has load: {self.tokens_load(time)+ request.input_len}")

        if (self.tokens_load(time)+ request.input_len) > self.max_pending_prompt_tokens:
            
            return True
        else:
            return False

    def check_req_stage_existence(self, req_stage:RequestStage):
        """
            Check if the specified request type is existing.
            Used for Splitwise to move engines type.
            For e.g. If req_stage is Prefill, it checks if the request list has Prefill requests.
        """
        for (_,req) in self.request_queue:
            if (req.get_current_stage() == req_stage):
                return True
        return False


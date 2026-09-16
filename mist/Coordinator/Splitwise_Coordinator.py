
from .MIST_Coordinator import MISTCoordinator, CoordRouterType
from mist.Engine import EngineType, MISTEngine, LLMEngine
from mist.Scheduler import SchedulerConfig, BatchingMethod
from mist.Input_requests import UniformDistribution, TraceDistributions, PoissonDistribution, LengthVariables
from mist.Platforms import PlatformConfig
from mist.Request import Request,RequestStage
from typing import List, Optional
from mist.Global_Network import get_network_bw_between_engines, get_network_spec
import pandas as pd
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

def engine_to_stage_mapping(type: EngineType):
    if type == EngineType.RAG:
        return RequestStage.RAG
    elif type == EngineType.PREPROCESS:
        return RequestStage.PREPROCESS
    elif type == EngineType.PREFILL:
        return RequestStage.PREFILL
    elif type == EngineType.DECODE:
        return RequestStage.DECODE
    elif type == EngineType.POSTPROCESS:
        return RequestStage.POSTPROCESS
    else:
        raise ValueError(f"No stage  found for the engine:{type}")

class MISTCoordinatorDisagg(MISTCoordinator):

    def __init__(
        self,
        request_queue_distr = None,
        model = 'meta-llama/meta-llama-3.1-70b',
        num_llm_engines=3,
        num_prefill_engines=1,
        num_decode_engines=1,
        num_mixed_engines=0,
        platform = None,
        decode_platform = None,
        cluster_schedule=CoordRouterType.JOIN_SHORTEST_QUEUE,
        convert_to_mixed_engine=True,

        logging_file: Optional[str] = 'trace.json',
        max_sim_time: Optional[float] = np.inf,
        max_pending_prompt_tokens: int = 8192,
        ) -> None:

        # Defaults are built per instance: a request queue or platform in the signature
        # would be created at import time and shared (and mutated) by every coordinator.
        if request_queue_distr is None:
            request_queue_distr = UniformDistribution(
                rps=30, sim_time=1000,
                input_vars=LengthVariables(4096,3), output_vars=LengthVariables(4096,3),
            ).request_queue
        if platform is None:
            platform = PlatformConfig(device='H100_GPU', tensor_parallel_size=4,
                                      model='meta-llama/meta-llama-3.1-70b', pipeline_parallel_size=1)

        assert num_prefill_engines != 0 and num_decode_engines != 0, "Must have initial prefill and decode engines"
        assert num_decode_engines + num_prefill_engines + num_mixed_engines == num_llm_engines, "Incorrect engines assignment"
        self.num_prefill_engines=num_prefill_engines
        self.num_decode_engines=num_decode_engines
        self.num_mixed_engines=num_mixed_engines

        self.platform = platform
        self.decode_platform = decode_platform if decode_platform is not None else platform
        self.model = model
        # Initialize the parent class
        super().__init__(
            starting_request_queue=request_queue_distr,
            logging_file=logging_file,
            max_sim_time=max_sim_time
            )

        # PARAMS for the cluster (system) configs
        self.num_llm_engines=num_llm_engines
        self.cluster_schedule=cluster_schedule

        # TODO: For Splitwise and DistServe, they should have this
        self.fixed_prefill_to_decode = True
        self.convert_to_mixed_engine = convert_to_mixed_engine

        # PARAMS for simulation logs
        self.req_cntr = 0
        self.sim_time = 1000
        self.current_time = 0

        # This is a queue used to track the movement and will be moved back to original state
        # value: (type moved from, type moved to, engine_id)
        self.engine_movement_tracker = deque()

        self.prefill_cntr=0
        self.decode_cntr=0
        self.mixed_cntr=0
        # Assign the engines to the
        self.engine_matcher[EngineType.PREFILL] = []
        self.engine_matcher[EngineType.DECODE] = []
        # Need a mixed pool for the purpose of easy management
        self.engine_matcher[EngineType.MIXED] = []

        logger.debug(f"model used by platform: {self.platform.model}")

        pre_engine_decode_only_cache = {}
        pre_engine_mixed_batch_cache = {}
        for i in range(self.num_prefill_engines):
            prefill_engine_to_add = LLMEngine(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED, max_batched_kv_size=900000),
            platform=self.platform,
            engine_id="PREFILL"+str(i),
            engine_types = [EngineType.PREFILL, EngineType.DECODE],
            max_pending_prompt_tokens=max_pending_prompt_tokens,
            decode_only_cache = pre_engine_decode_only_cache,
            mixed_batch_cache = pre_engine_mixed_batch_cache,
            )
            self.add_engine(prefill_engine_to_add, [EngineType.PREFILL])
        
        dec_engine_decode_only_cache = {}
        dec_engine_mixed_batch_cache = {}
        for i in range(self.num_decode_engines):
            decode_engine_to_add = LLMEngine(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.MIXED),
            platform=self.platform,
            engine_id="DECODE"+str(i),
            engine_types = [EngineType.DECODE],
            max_pending_prompt_tokens=max_pending_prompt_tokens,
            decode_only_cache = dec_engine_decode_only_cache,
            mixed_batch_cache = dec_engine_mixed_batch_cache,
            
            )
            self.add_engine(decode_engine_to_add, [EngineType.DECODE])

        # Mixed engines are able to run both Prefill and Decode in chunked prefill manner
        for i in range(self.num_mixed_engines):
            mixed_engine_to_add = LLMEngine(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.CHUNKED, max_batched_kv_size=900000),
            platform=self.platform,
            engine_id="MIXED"+str(i),
            engine_types = [EngineType.PREFILL, EngineType.DECODE],
            max_pending_prompt_tokens=max_pending_prompt_tokens,
            decode_only_cache = pre_engine_decode_only_cache,
            mixed_batch_cache = pre_engine_mixed_batch_cache,
            )
            self.add_engine(mixed_engine_to_add, [EngineType.MIXED])

        # # Add An Engine for the HOST (Determine characteristics of host later)
        # self.add_engine(MISTEngine(
        #     model = self.model,
        #     sim_duration = 10000000,
        #     engine_types = [EngineType.HOST],
        #     ), [EngineType.HOST])

        # Print out layout and assignment of engines
        logger.info(" ".join([
            f"Prefill Engines: {self.engine_matcher[EngineType.PREFILL]},",
            f"Decode Engines: {self.engine_matcher[EngineType.DECODE]},",
            f"Mixed Engines: {self.engine_matcher[EngineType.MIXED]}"
        ]))

    def _check_the_engine_load(self, engine_id):
        assert engine_id < self.num_llm_engines, f"Invalid Engine id"
        engine_load_num = len(self.engines[engine_id].request_queue)

    def _reassign_engine_type(self, engine_id, old_type:EngineType, new_type:EngineType):
        """
            TODO: This is used by Splitwise-like system to reassign engine to another type
        """
        assert engine_id < self.num_llm_engines, f"Invalid Engine id"
        assert len(self.engine_matcher[old_type]) > 0, f"cannot remove more engines from type -- {old_type}"
        self.engine_matcher[old_type].remove(engine_id)
        self.engine_matcher[new_type].append(engine_id)
        # print(f"reassigning engine: {engine_id} from {old_type} to {new_type}")
        if old_type == EngineType.MIXED:
            # Move back to the original engine requires remove the capability of running the opposite types
            self.engines[engine_id].engine_types.remove(self._get_opposite_engine_type(new_type))
            self.engines[engine_id].scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED)
        else:
        # Capable to run both prefill and decode now
            self.engines[engine_id].engine_types.append(self._get_opposite_engine_type(old_type))
            self.engines[engine_id].scheduler_config = SchedulerConfig(batching_method=BatchingMethod.MIXED)

    def _get_least_loaded_engines(self, engine_types:List[EngineType], request:Request):
        """
            Return the least loaded engines for a list of engine types inside the coordinator
        """
        engine_to_select = -1
        # print(f"Engine Type:{engine_type} number of engines: {num_engines}")
        engine_loads = {}
        for eng_type in engine_types:
            num_engines = len(self.engine_matcher[eng_type])
            for type_id in range(num_engines):
                engine_id = self.engine_matcher[eng_type][type_id]
                engine_loads[(engine_id, eng_type)] = self.engines[engine_id].tokens_load(request.metrics.arrival_time)

        ## If all the engines have the same load, just return in RR manner for the original type (prefill or decode)
        if len(set(engine_loads.values())) == 1:
            return super()._determine_dst_engine(engine_types[0])
        ## Else find the least loaded engine
        elif len(engine_loads) > 0:
            (engine_to_select, eng_type) = min(engine_loads, key=engine_loads.get)

        return engine_to_select

    def _get_least_loaded_engines_prefill(self, engine_types:List[EngineType], request:Request):
        """
            Return the least loaded engines for a list of engine types inside the coordinator
        """
        engine_to_select = -1
        # print(f"Engine Type:{engine_type} number of engines: {num_engines}")
        engine_loads = {}
        for eng_type in engine_types:
            num_engines = len(self.engine_matcher[eng_type])
            for type_id in range(num_engines):
                engine_id = self.engine_matcher[eng_type][type_id]
                engine_loads[(engine_id,eng_type)] = self.engines[engine_id].tokens_load(request.metrics.arrival_time) + request.input_len

        ## If all the engines have the same load, just return in RR manner for the original type (prefill or decode)
        if len(set(engine_loads.values())) == 1 and len(engine_loads) > 0:
            engine_to_select = super()._determine_dst_engine(engine_types[0])
        elif len(engine_loads) > 0:
            (engine_to_select, eng_type) = min(engine_loads, key=engine_loads.get)
        return engine_to_select

    def _get_least_loaded_engine(self, engine_type:EngineType, request:Request):
        """
            Return the least loaded engine for a single engine type inside the coordinator
        """
        engine_to_select = -1
        num_engines = len(self.engine_matcher[engine_type])
        # print(f"Engine Type:{engine_type} number of engines: {num_engines}")
        engine_loads = {}
        for type_id in range(num_engines):
            engine_id = self.engine_matcher[engine_type][type_id]
            engine_loads[engine_id] = self.engines[engine_id].tokens_load(request.metrics.arrival_time)
        # print("Engine Loads: ", engine_loads)
        ## If all the engines have the same load, just return in RR manner
        if len(set(engine_loads.values())) == 1 or sum(1 for load in engine_loads.values() if load == 0) > 0:
            return super()._determine_dst_engine(engine_type)
        ## Else find the least loaded engine
        elif len(engine_loads) > 0:
            engine_to_select = min(engine_loads, key=engine_loads.get)
        return engine_to_select

    # TODO: This can be removed, because add input length of request does not change anything
    def _get_least_loaded_engine_prefill(self, engine_type:EngineType, request:Request):
        """
            Return the least loaded engine for a single engine type inside the coordinator
        """
        engine_to_select = -1
        num_engines = len(self.engine_matcher[engine_type])
        # print(f"Engine Type:{engine_type} number of engines: {num_engines}")
        engine_loads = {}
        for type_id in range(num_engines):
            engine_id = self.engine_matcher[engine_type][type_id]
            engine_loads[engine_id] = self.engines[engine_id].tokens_load(self.global_time) + request.input_len

        ## If all the engines have the same load, just return in RR manner
        if len(set(engine_loads.values())) == 1:
            return super()._determine_dst_engine(engine_type)
        ## Else find the least loaded engine
        elif len(engine_loads) > 0:
            engine_to_select = min(engine_loads, key=engine_loads.get)
        return engine_to_select

    def _get_opposite_engine_type(self, engine_type:EngineType):
        assert engine_type in [EngineType.PREFILL, EngineType.DECODE], "it has to be either PREFILL or DECODE"
        if engine_type == EngineType.PREFILL:
            return EngineType.DECODE
        else:
            return EngineType.PREFILL

    def _get_mixed_engine_overloaded(self, init_engine_type:EngineType, request:Request):
        # Go over all mixed engines to check if there is a good loaded engine
        engine_to_select = self._get_least_loaded_engine(EngineType.MIXED, request)
        # print(f"number of mixed engine {len(self.engine_matcher[EngineType.MIXED])}")

        if engine_to_select == -1 or self.engines[engine_to_select].check_is_overload(request.metrics.arrival_time):
            if self.convert_to_mixed_engine:
                # If mixed engines are overloaded
                type_to_find = self._get_opposite_engine_type(init_engine_type)
                engine_to_select = self._get_least_loaded_engine(type_to_find, request)
                # print(f"No Mixed found, converting {type_to_find} to mixed engine. Engine to select: {engine_to_select}")
                if len(self.engine_matcher[type_to_find]) < 1 or self.engines[engine_to_select].check_is_overload(request.metrics.arrival_time):
                    # All the engines are overloaded, just don't move, find the
                    # least loaded engine in the original pool
                    engine_to_select = self._get_least_loaded_engine(init_engine_type, request)
                else:
                    self._reassign_engine_type(engine_to_select, type_to_find, EngineType.MIXED)

                    # print(f"Convert {type_to_find} to {EngineType.MIXED} at time: {self.global_time}")
                    # Track the movement that just took place
                    self.engine_movement_tracker.append((type_to_find, EngineType.MIXED, engine_to_select))
            else:
                # Not convert to mixed engine, Find the engine that has the least load in both initiated engine pool or Mixed engine pool
                engine_to_select = self._get_least_loaded_engines([init_engine_type, EngineType.MIXED], request)
                assert engine_to_select != -1, f"must select an engine because no movement of engines between pools"
        
        return engine_to_select

    def _get_mixed_engine_overloaded_prefill(self, init_engine_type:EngineType, request:Request=None):
        # Go over all mixed engines to check if there is a good loaded engine
        engine_to_select = self._get_least_loaded_engine_prefill(EngineType.MIXED, request)
        # print(f"number of mixed engine {len(self.engine_matcher[EngineType.MIXED])}")

        if engine_to_select == -1 or self.engines[engine_to_select].check_is_overload_with_prefill(request.metrics.arrival_time, request):
            if self.convert_to_mixed_engine:
                # If mixed engines are overloaded
                type_to_find = self._get_opposite_engine_type(init_engine_type)
                engine_to_select = self._get_least_loaded_engine_prefill(type_to_find, request)
                # print(f"No Mixed found, converting {type_to_find} to mixed engine. Engine to select: {engine_to_select}")
                if len(self.engine_matcher[type_to_find]) < 1 or self.engines[engine_to_select].check_is_overload_with_prefill(request.metrics.arrival_time, request):
                    # All the engines are overloaded, just don't move, find the
                    # least loaded engine in the original pool
                    engine_to_select = self._get_least_loaded_engines_prefill([init_engine_type, EngineType.MIXED], request)
                else:
                    self._reassign_engine_type(engine_to_select, type_to_find, EngineType.MIXED)

                    # print(f"Convert {type_to_find} to {EngineType.MIXED} at time: {self.global_time}")
                    # Track the movement that just took place
                    self.engine_movement_tracker.append((type_to_find, EngineType.MIXED, engine_to_select))
            else:
                # Not convert to mixed engine, Find the engine that has the least load in both initiated engine pool or Mixed engine pool
                engine_to_select = self._get_least_loaded_engines([init_engine_type, EngineType.MIXED], request)
                assert engine_to_select != -1, f"must select an engine because no movement of engines between pools"
        
        return engine_to_select

    def _determine_prefill_engine(self, request:Request):
        assert self.num_prefill_engines > 0, "Cannot run with 0 prefill engines"
        if self.cluster_schedule == CoordRouterType.ROUND_ROBIN:
            #TODO: Use RoundRobin in base class, may need to do the engine movement like JSQ later.
            return super()._determine_dst_engine(EngineType.PREFILL)
        elif self.cluster_schedule == CoordRouterType.JOIN_SHORTEST_QUEUE:
            engine_to_select = self._get_least_loaded_engine(EngineType.PREFILL, request)
            # print(f'Req:{request.request_id} engine to select:', engine_to_select)
            if engine_to_select != -1 and self.engines[engine_to_select].check_is_overload_with_prefill(request.metrics.arrival_time, request):
                # If the engine is overloaded, we need to use engines in the mixed pool
                engine_to_select = self._get_mixed_engine_overloaded_prefill(EngineType.PREFILL,request)

            if engine_to_select == -1:
                # If the engine to select is still -1, go to find Mixed Engine
                engine_to_select = self._get_least_loaded_engine(EngineType.MIXED, request)

            if self.fixed_prefill_to_decode:
                # Determine the decode engine in here
                request.engine_assigned_stage[RequestStage.DECODE] = self._determine_decode_engine(request)
                ## Keeping track of the request that is enqueued to the decode engine
            return engine_to_select

        else:
            raise ValueError(f"Coordinator Scheduler Type Not Supported:{self.cluster_schedule}")

    # Use Join Shortest Queue below, because RoundRobin is used in the base class
    def _determine_decode_engine(self, request:Request):
        assert self.num_prefill_engines > 0, "Cannot run with 0 prefill engines"
        if self.cluster_schedule == CoordRouterType.ROUND_ROBIN:
            #TODO: Do a similar thing like Prefill
            return super()._determine_dst_engine(EngineType.PREFILL)
        elif self.cluster_schedule == CoordRouterType.JOIN_SHORTEST_QUEUE:
            engine_to_select = self._get_least_loaded_engine(EngineType.DECODE, request)
            # print(f"engine id: {self.engines[engine_to_select].engine_id}, type: {self.engines[engine_to_select].engine_types}")
            if engine_to_select != -1 and self.engines[engine_to_select].check_is_overload(request.metrics.arrival_time):
                # If the engine is overloaded, we need to use engines in the mixed pool
                engine_to_select = self._get_mixed_engine_overloaded(EngineType.DECODE, request)

            if engine_to_select == -1:
                # If the engine to select is still -1, go to find
                engine_to_select = self._get_least_loaded_engine(EngineType.MIXED, request)
            return engine_to_select
        else:
            raise ValueError(f"Coordinator Scheduler Type Not Supported:{self.cluster_schedule}")

    # def _determine_mixed_engine(self, request:Request):
    #     mixed_engine_id = self.mixed_cntr % self.num_prefill_engines
    #     self.mixed_cntr += 1
    #     engine_id = self.engine_matcher[EngineType.MIXED][mixed_engine_id]
    #     return engine_id

    # TODO: Splitwise has optimized movement protocal that does per-layer transfer
    def _get_move_request_time(self, request, src_engine, dst_engine) -> float:
        # Get the cost of moving the request from src to dst engine
        return super()._get_move_request_time(request, src_engine, dst_engine)
    def initial_engine_connection_df(self):
        network_spec = get_network_spec("H100_nvl8")
        intra_domain_sz = self.platform.tensor_parallel_size * self.platform.pipeline_parallel_size
        global_size = self.num_llm_engines
        self.engine_connection_bw_latency_df = get_network_bw_between_engines(global_size, network_spec, intra_domain_sz)
        ## Each node has 1 CPU host
        for i in range(self.num_total_engines):
            for j in range(self.num_llm_engines, self.num_total_engines):
                bw = network_spec.intra_node_bw * network_spec.intra_node_eff
                latency = network_spec.intra_node_latency
                # Append bidirectional connections
                cur_data = [
                    {"src": i, "dst": j, "latency(msec)": latency, "BW(GB/s)": bw},
                    {"src": j, "dst": i, "latency(msec)": latency, "BW(GB/s)": bw},
                ]
                df = pd.DataFrame(cur_data)
                self.engine_connection_bw_latency_df = pd.concat([self.engine_connection_bw_latency_df, df])

    def _determine_dst_engine(self, engine_type, request):
        """
            Overloaded function to determine the destination engine
        """
        # TODO: Temporarily put the determination of the engine movement here and also need to see
        # if the determination is too frequent
        pop_count = len(self.engine_movement_tracker)
        while self.engine_movement_tracker and pop_count > 0:
            orig_type, cur_type, engine_id = self.engine_movement_tracker.popleft()  # Pop from front
            pop_count -= 1
            eng_type = self._get_opposite_engine_type(orig_type)
            req_stage = engine_to_stage_mapping(eng_type)
            if not self.engines[engine_id].check_req_stage_existence(req_stage):
                self._reassign_engine_type(engine_id, cur_type, orig_type)
            else:
                self.engine_movement_tracker.append((orig_type, cur_type, engine_id))

        # TODO: May combine PEFILL into the same function (the callee functions are similar)
        if engine_type == EngineType.PREFILL:
            prefill_engine = self._determine_prefill_engine(request)
            # print(f"determine dest for req: {request.request_id}, with engine: {engine_type} assigned to {prefill_engine}")
            return prefill_engine
        elif engine_type == EngineType.DECODE:
            if self.fixed_prefill_to_decode:
                # If the decode engine is determined while prefill
                assert request.engine_assigned_stage[RequestStage.DECODE] != None, "The decode should be assigned"
                decode_engine = request.engine_assigned_stage[RequestStage.DECODE]
            else:
                decode_engine = self._determine_decode_engine(request)
            # print(f"determine dest for req: {request.request_id}, with engine: {engine_type} assigned to {decode_engine}")
            return decode_engine
        else:
            return super()._determine_dst_engine(engine_type)

from .MIST_Coordinator import MISTCoordinator, CoordRouterType
from mist.Engine import EngineType, MISTEngine, LLMEngine
from mist.Scheduler import SchedulerConfig, BatchingMethod
from mist.Input_requests import UniformDistribution, TraceDistributions, PoissonDistribution, LengthVariables
from mist.Platforms import PlatformConfig
from mist.Request import Request,RequestStage
from typing import List
from mist.Global_Network import get_network_bw_between_engines, get_network_spec
import pandas as pd
from collections import deque
import math

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

class MISTCoordinatorDisagg_DistServe(MISTCoordinator):

    def __init__(
        self,
        request_queue_distr = UniformDistribution(
                rps=30,
                sim_time=1000,
                input_vars=LengthVariables(4096,3),
                output_vars=LengthVariables(4096,3),
        ).request_queue,
        model = 'meta-llama/meta-llama-3.1-70b',
        num_llm_engines=2,
        num_prefill_engines=1,
        num_decode_engines=1,
        platform = PlatformConfig(device='H100_GPU', tensor_parallel_size=4,
                                model='meta-llama/meta-llama-3.1-70b', pipeline_parallel_size=1),
        cluster_schedule=CoordRouterType.JOIN_SHORTEST_QUEUE
        ) -> None:
        
        assert num_decode_engines + num_prefill_engines == num_llm_engines, "Incorrect engines assignment"
        self.num_prefill_engines=num_prefill_engines
        self.num_decode_engines=num_decode_engines
        self.num_llm_engines=num_llm_engines

        self.platform = platform
        self.model = model
        # Initialize the parent class
        super().__init__(request_queue_distr)

        # PARAMS for the cluster (system) configs
        self.cluster_schedule=cluster_schedule
        # PARAMS for simulation logs
        self.req_cntr = 0
        self.sim_time = 1000
        self.current_time = 0

       

        # This is a queue used to track the movement and will be moved back to original state
        # value: (type moved from, type moved to, engine_id)
        self.engine_movement_tracker = deque()
        
        # Used to help identify the group of decodes to be assigned to
        self.prefill_to_decode_matcher = dict()

        self.prefill_cntr=0
        self.decode_cntr=0

         # TODO: specify the affinity in here to choose either add_engine/place_engine, use post_init for sugar
        # Assign the engines to the 
        self.engine_matcher[EngineType.PREFILL] = []
        self.engine_matcher[EngineType.DECODE] = []
        for i in range(self.num_prefill_engines):
            prefill_engine_to_add = LLMEngine(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED),
            platform=self.platform,
            engine_id="PREFILL"+str(i),
            engine_types = [EngineType.PREFILL],
            )
            self.add_engine(prefill_engine_to_add, [EngineType.PREFILL])
        for i in range(self.num_decode_engines):
            decode_engine_to_add = LLMEngine(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED),
            platform=self.platform,
            engine_id="DECODE"+str(i),
            engine_types = [EngineType.DECODE],
            )
            self.add_engine(decode_engine_to_add, [EngineType.DECODE])

        # # Add An Engine for the HOST (Determine characteristics of host later)
        # self.add_engine(MISTEngine(
        #     model = self.model,
        #     sim_duration = 10000000,
        #     engine_types = [EngineType.HOST],
        #     ), [EngineType.HOST])

        # Print out layout and assignment of engines
        print(" ".join([
            f"Decode Engines: {len(self.engine_matcher[EngineType.PREFILL])},",
            f"Prefill Engines: {len(self.engine_matcher[EngineType.DECODE])},"
        ]))

    def _check_the_engine_load(self, engine_id):
        assert engine_id < self.num_llm_engines, f"Invalid Engine id"
        engine_load_num = len(self.engines[engine_id].request_queue)

    def _get_least_loaded_engine(self, engine_list:List, request:Request):
        engine_to_select = -1
        # print(f"Engine Type:{engine_type} number of engines: {num_engines}")
        engine_loads = {}
        # for type_id in range(num_engines):
        #     engine_id = self.engine_matcher[engine_type][type_id]
        #     engine_loads[engine_id] = self.engines[engine_id].tokens_load(self.global_time)
        for engine_id in engine_list:
            print(f"At time: {request.metrics.arrival_time}, engine: {engine_id} has load: {self.engines[engine_id].tokens_load(request.metrics.arrival_time)}")
            engine_loads[engine_id] = self.engines[engine_id].tokens_load(request.metrics.arrival_time)

        ## If all the engines have the same load, just return in RR manner
        if len(set(engine_loads.values())) == 1:
            engine_to_select= engine_list.pop(0)
            engine_list.append(engine_to_select)
        ## Else find the least loaded engine
        elif len(engine_loads) > 0:
            engine_to_select = min(engine_loads, key=engine_loads.get)
        
        print(f"At time: {request.metrics.arrival_time}, {engine_to_select} is selected")
        return engine_to_select
    
    # def _get_least_loaded_engine_prefill(self, engine_type:EngineType):
    #     engine_to_select = -1
    #     num_engines = len(self.engine_matcher[engine_type])
    #     # print(f"Engine Type:{engine_type} number of engines: {num_engines}")
    #     engine_loads = {}
    #     for type_id in range(num_engines):
    #         engine_id = self.engine_matcher[engine_type][type_id]
    #         engine_loads[engine_id] = self.engines[engine_id].tokens_load(self.global_time)

    #     ## If all the engines have the same load, just return in RR manner
    #     if len(set(engine_loads.values())) == 1:
    #         return super()._determine_dst_engine(engine_type, None)
    #     ## Else find the least loaded engine
    #     elif len(engine_loads) > 0:
    #         engine_to_select = min(engine_loads, key=engine_loads.get)
    #     return engine_to_select


    def _determine_prefill_engine(self, request:Request):
        assert self.num_prefill_engines > 0, "Cannot run with 0 prefill engines"
        if self.cluster_schedule == CoordRouterType.ROUND_ROBIN:
            #TODO: Use RoundRobin in base class, may need to do the engine movement like JSQ later.
            return super()._determine_dst_engine(EngineType.PREFILL, request)
        elif self.cluster_schedule == CoordRouterType.JOIN_SHORTEST_QUEUE:
            print(f"request: {request.request_id} find engines: {self.engine_matcher[EngineType.PREFILL]}")
            engine_to_select = self._get_least_loaded_engine(self.engine_matcher[EngineType.PREFILL], request)
            request.engine_assigned_stage[RequestStage.DECODE] = self._determine_decode_engine(request, engine_to_select)
            return engine_to_select
        else:
            raise ValueError(f"Coordinator Scheduler Type Not Supported:{self.cluster_schedule}")

    # Use Join Shortest Queue below, because RoundRobin is used in the base class
    def _determine_decode_engine(self, request:Request, prefill_id=-1):
        assert self.num_prefill_engines > 0, "Cannot run with 0 prefill engines"
        if self.cluster_schedule == CoordRouterType.ROUND_ROBIN:
            #TODO: Do a similar thing like Prefill
            return super()._determine_dst_engine(EngineType.PREFILL, request)
        elif self.cluster_schedule == CoordRouterType.JOIN_SHORTEST_QUEUE:
            if request.engine_assigned_stage[RequestStage.DECODE] == None:
                engine_to_select = self._get_least_loaded_engine(self.prefill_to_decode_matcher[prefill_id], request)
            else:
                engine_to_select = request.engine_assigned_stage[RequestStage.DECODE]
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
        data_size = request.get_data_movement_size(self.engines[dst_engine].model)/2**20   # Convert to MB
        if src_engine == dst_engine:
            return 0
        else:
            latency, BW = self.compute_bw_latency(self.engine_connection_bw_latency_df, src_engine, dst_engine)
            return latency + data_size/BW

    def intial_engine_connection_df(self):
        network_spec = get_network_spec("H100_nvl8")
        intra_domain_sz = self.platform.tensor_parallel_size * self.platform.pipeline_parallel_size
        global_size = self.num_llm_engines
        self.engine_connection_bw_latency_df = get_network_bw_between_engines(global_size, network_spec, intra_domain_sz)
        ## Each node has 1 CPU host
        for i in range(self.num_llm_engines):
            j = self.num_llm_engines
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

        # print(f"determine dest for req: {request.request_id}, with engine: {engine_type}")
        if engine_type == EngineType.PREFILL:
            return self._determine_prefill_engine(request)
        elif engine_type == EngineType.DECODE:
            return self._determine_decode_engine(request)
        else:
            # Determine the engine destination for other stages
            return super()._determine_dst_engine(engine_type,request)

    def affinity_dermination_low(self, df, num_prefills, num_decodes):
        """
            This function is used to help determine the affinity given the number of prefill/decode engines

            High Affinity means that it determines the parallelism for both prefill and decode and replicate them
            It means that all the interconnects have the same bandwidth
        """
        selected_eng = set()

        # Map 
        low_affinity_engine = []
        for i in range (self.num_llm_engines):
            if len(low_affinity_engine) == num_prefills:
                break
            row = df[(df['src'] == i) & (df['dst'] < self.num_llm_engines)& (df['dst'] != df['src'])]
            bw_columns = [col for col in df.columns if col.startswith("BW")]
            dst_max_bw = row.loc[row[bw_columns].idxmax()]['dst'].values[0]
            if not dst_max_bw in selected_eng:        
                selected_eng.add(dst_max_bw)
                selected_eng.add(i)
                low_affinity_engine.append((i, dst_max_bw))

        # TODO: Need to reorganize the engine layouts if this is the case -- 0 is prefill, 1 is decode
        # Move this to the initialization
        for (p, d) in low_affinity_engine:
            prefill_engine_to_add = LLMEngine(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED),
            platform=self.platform,
            engine_types = [EngineType.PREFILL],
            )
            self.place_engine(prefill_engine_to_add, [EngineType.PREFILL], p)
            decode_engine_to_add = LLMEngine(
            model = self.model,
            scheduler_config = SchedulerConfig(batching_method=BatchingMethod.DISAGGREGATED),
            platform=self.platform,
            engine_types = [EngineType.DECODE],
            )
            self.place_engine(decode_engine_to_add, [EngineType.DECODE], d)

    def affinity_dermination_high(self, num_prefills, num_decodes):
        """
            This function is used to help determine the affinity given the number of prefill/decode engines
            TODO: This is not planned to support for now

            High Affinity means that it determines the parallelism for both prefill and decode and replicate them
            It means that all the interconnects have the same bandwidth
        """ 
        assert num_prefills<=self.num_prefill_engines, f"incorrect prefill engines number"
        assert num_decodes<=self.num_decode_engines, f"incorrect decode engines number"
        num_prefill_reps = math.ceil(self.num_prefill_engines // num_prefills)
        num_decode_reps = math.ceil(self.num_decode_engines // num_decodes)
        assert num_decode_reps == num_decode_reps, f"The number of prefill-decode replications should match"
        print(f"number of reps: {num_prefill_reps}")
        for i in range(num_prefill_reps):
            strt_p = i*num_prefills
            end_p = min(strt_p + num_prefills, self.num_prefill_engines)
            
            strt_d = i*num_decodes + self.num_prefill_engines
            end_d = min(strt_d + num_decodes, self.num_llm_engines)
            for j_p in range(strt_p, end_p):
                self.prefill_to_decode_matcher[j_p] = []
                for j_d in range(strt_d, end_d):
                    self.prefill_to_decode_matcher[j_p].append(j_d)


    def affinity_engine_mapping(self, affinity_groups):
        """
            This function is used to assign the instance engines based on the connection
            
            The input includes the affinity groups of prefill and decode
            format: [([0,1], [4]), ([2,3], [5])] --> the 1st index is prefill and 2nd index is decode
            Then, this function replicate the prefill and decode pairs to limit the 
            accessibility of each prefill-to-decode assignment
        """
        for (p,d) in affinity_groups: 
            for engine_id in p:
                self.prefill_to_decode_matcher[engine_id] = d

        print(self.prefill_to_decode_matcher)     
        

            



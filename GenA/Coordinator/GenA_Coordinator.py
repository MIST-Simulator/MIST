from GenA.Tracing import ChromeTracingLogger
from GenA.Engine import GenAEngine,EngineType, stage_to_engine_mapping,EngineMetrics
from GenA.Platforms.platforms import PlatformConfig, PlatformType
from GenA.Scheduler.scheduler import Scheduler, SchedulerConfig, BatchingMethod
from GenA.Request import Request, DataMetrics, RequestMetrics, RequestStatus, RequestStage
from GenA.Input_requests.Request_inputs import RequestDistributions, UniformDistribution, PoissonDistribution, NormalDistribution
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
import heapq
from uuid import uuid4
import pandas as pd
import numpy as np
import enum

from GenA.Coordinator.global_router import CoordRouterType, LoadTypes, req_is_heavy


class EventType(enum.Enum):
    """Status of a sequence."""
    REQUEST_ARRIVAL = enum.auto()
    ENGINE_RUN_STEP = enum.auto()

    def __lt__(self, other):
        return self.value < other.value

class GenACoordinator:
    def __init__(
            self,
            starting_request_queue: List[Request] = [],
            logging_file: Optional[str] = 'trace.json',
            max_sim_time: Optional[float] = np.inf,
            network_file: Optional[str | pd.DataFrame] = None,
            router_type: Optional[CoordRouterType] = CoordRouterType.ROUND_ROBIN,
        ):

        self.router_type = router_type

        ## Engine related parameters
        self.num_total_engines = 0
        self.GenA_engines= []
        self.engine_matcher = dict()
        self.engine_active = []
        self.engine_next_step = dict()
        self.engine_connection_bw_latency_df = pd.DataFrame()
        self.network_file = network_file

        ## Request related parameters
        self.starting_request_queue = starting_request_queue
        self.request_accepted = len(starting_request_queue)
        self.event_queue = []
        self.request_serviced = 0
        self.completed_requests = []

        ## Simulation related parameters
        self.engine_sim_queue = []
        self.global_time = 0
        self.max_sim_time = max_sim_time

        ## Logger
        self.logger = ChromeTracingLogger(logging_file)

    def add_engine(self, engine:GenAEngine, engine_types:List[EngineType]):
        """
            This function is designed to add an engine to the GenA system
        """
        engine_id = self.num_total_engines
        engine.engine_id = engine_id
        self.GenA_engines.append(engine)
        engine.logger = self.logger
        for engine_type in engine_types:
            if engine_type not in self.engine_matcher:
                self.engine_matcher[engine_type] = []
            self.engine_matcher[engine_type].append(engine_id)
        self.num_total_engines += 1
        self.engine_active.append(False)
        self.engine_next_step[engine_id] = 0
        # heapq.heappush(self.engine_sim_queue, (0, engine_id))

    def add_request(self, request:Request, time:float):
        """
            This function is designed to add a request to the GenA system
        """
        request.request_id = self.request_accepted
        self.request_accepted += 1
        heapq.heappush(self.request_queue, (time, request, self._get_current_engine(request)))

    def _get_current_engine(self, request: Request):
        """
            This function is designed to get the current engine based on the request
        """
        current_stage = request.current_stage
        engine_type_required = stage_to_engine_mapping(current_stage)
        engine_id = self._determine_dst_engine(engine_type_required, request)
        return engine_id

    def _get_next_engine(self, request: Request):
        """
            This function is designed to get the next engine based on the request
        """
        next_stage = request.get_next_stage()
        engine_type_required = stage_to_engine_mapping(next_stage)
        engine_id = self._determine_dst_engine(engine_type_required, request)
        # print(f"Request {request.request_id} current stage {request.current_stage} next stage {next_stage}, engine type {engine_type_required}, engine {engine_id}")
        return engine_id

    def run_sim(self):
        """
            This function is designed to simulate the running of the GenA system
        """
        ## Initialize the DF for engine connection
        # self.initial_engine_connection_df("network_bw.csv")
        if isinstance(self.network_file, pd.DataFrame):
            self.engine_connection_bw_latency_df = self.network_file
        elif isinstance(self.network_file, str):
            self.initial_engine_connection_from_file()
        else:
            self.initial_engine_connection_df()

        next_print_time = 1000

        ## Engine initialization might change the first engine for each request
        ## Determine the first engine for each request
        for req_id in range(self.request_accepted):
            self.starting_request_queue[req_id].request_id = req_id
            heapq.heappush(self.event_queue, (self.starting_request_queue[req_id].metrics.arrival_time,
                                            EventType.REQUEST_ARRIVAL, (self.starting_request_queue[req_id], None)))

        queued_engine_events = set()
        while (self.event_queue and (self.request_serviced < self.request_accepted) and
            ((self.global_time < self.max_sim_time) or (self.event_queue and self.event_queue[0][0] < self.max_sim_time))):
            ## Add all the requests to appropriate engines
            # Pop out the event and determine the type
            event_push_time, event_type, event_items = heapq.heappop(self.event_queue)
            # print(f"Global Time {self.global_time}, Next Event Start Time: {event_push_time}, Event Type: {event_type}")

            if (event_type == EventType.REQUEST_ARRIVAL):
                (cur_req, next_engine_id) = event_items
                if next_engine_id == None:
                    ## First engine should be determined only once the req has arrived
                    next_engine_id = self._get_current_engine(cur_req)

                request_added_engine_time = self.GenA_engines[next_engine_id].current_time
                if request_added_engine_time < event_push_time:
                    ##
                    request_added_engine_time = event_push_time
                # push the run engine event into the queue
                req_push_event = (request_added_engine_time, next_engine_id)
                ## If the engine is active but will be active later, update the sim queue
                if (self.engine_next_step[next_engine_id] > request_added_engine_time) or self.GenA_engines[next_engine_id].engine_idle():
                    self.engine_next_step[next_engine_id] = request_added_engine_time
                    # if next_engine_id == 1:
                    #     print(f"Request added to request queue {next_engine_id} at {request_added_engine_time}")
                    heapq.heappush(self.event_queue, (request_added_engine_time, EventType.ENGINE_RUN_STEP, req_push_event))
                    queued_engine_events.add(req_push_event)

                # TODO: Here it uses the event push time but it might have delay between arrival time
                # and the actual added request time
                self.GenA_engines[next_engine_id].add_request(cur_req, event_push_time)

            elif (event_type == EventType.ENGINE_RUN_STEP):
                (cur_time,engine_id) = event_items
                if event_items in queued_engine_events:
                    queued_engine_events.remove(event_items)
                self.global_time = cur_time
                # print(f"Engine {engine_id} is idle =  {self.GenA_engines[engine_id].engine_idle()}")
                # if self.GenA_engines[engine_id].engine_idle() == False:
                if self.GenA_engines[engine_id].engine_idle() == False:
                    # if engine_id == 1:
                    #     print(f"Normal Engine {engine_id} step to start at {cur_time}")
                    engine_step_end_time, req_list = self.GenA_engines[engine_id].step(cur_time)
                    self.global_time = cur_time
                else:
                    continue
                # if self.request_queue:
                #     self.global_time = min(self.global_time, self.request_queue[0][0])
                if self.global_time >= next_print_time:
                    print(f"Global Time {self.global_time}")
                    next_print_time = (self.global_time // 1000 + 1) * 1000
                eng_push_event = (engine_step_end_time, engine_id)
                if self.GenA_engines[engine_id].engine_idle() == False:
                    # if engine_id == 1:
                    #     print(f"Normal Engine {engine_id} step end at {engine_step_end_time}")
                    heapq.heappush(self.event_queue, (engine_step_end_time, EventType.ENGINE_RUN_STEP, eng_push_event))
                    self.engine_next_step[engine_id] = engine_step_end_time

                # Handling the finished requests list and reinsert them into the event queue as needed
                if req_list:
                    for req in req_list:
                        request_engine_exit_time = req.get_engine_exit_time()
                        # print(f'Req {req.request_id} given out by engine {engine_id} at step end. {req.request_finished()}')
                        if req.request_finished():
                            self.request_serviced += 1
                            req.update_request_finished_time(request_engine_exit_time)
                            self.completed_requests.append(req)
                            # print(f'Req {req.request_id} serviced')
                        else:
                            next_engine_id = self._get_next_engine(req)
                            self.logger.log_event(req.request_id, 'Coordinator', str(engine_id)+"->"+str(next_engine_id), time=request_engine_exit_time,type= "B")
                            movement_time = self._get_move_request_time(req, engine_id, next_engine_id)
                            self.logger.log_event(req.request_id, 'Coordinator', str(engine_id)+"->"+str(next_engine_id), time=request_engine_exit_time+movement_time,type= "E")
                            # print(f'Req {req.request_id} pushed to {next_engine_id} at {request_engine_exit_time+movement_time}')
                            req.current_stage = req.get_next_stage()
                            # Handle requests that are needed to be reinserted to the engine
                            req_push_event = (req, next_engine_id)
                            heapq.heappush(self.event_queue, (request_engine_exit_time+movement_time, EventType.REQUEST_ARRIVAL, req_push_event))
            else:
                raise ValueError(f"No event type found for the engine:{event_type}")
        if self.logger.filename is not None:
            self.logger.save()
        ## Log the completion of the simulation
        self.global_stats = self.get_global_stats()

    # TODO: Add the logic to get the engine stats
    def get_global_stats(self) -> EngineMetrics:
        sum_TTFT = 0
        total_input = 0
        actual_output_lens = 0
        e2e_latencies = []          # Keeps track of all the end to end latencies
        TTFT_latencies = []          # Keeps track of all the TTFT latencies

        request_gen_times = []
        for engine in self.GenA_engines:
            if EngineType.PREFILL in engine.engine_types or EngineType.DECODE in engine.engine_types:
                total_input += sum([req.input_len - req.past_context - req.remaining_prefill_tokens  for req in engine.scheduler.running])
                actual_output_lens += sum([req.gen_tokens for req in engine.scheduler.running])
                request_gen_times.extend([req.data[j].finished_time - req.data[j].scheduled_time for req in engine.scheduler.running for j in range(len(req.data))] )
                TTFT_latencies.extend([req.data[0].finished_time - req.metrics.arrival_time for req in engine.scheduler.running if len(req.data) > 0])
        # request_gen_times = [item for sublist in request_gen_times for item in sublist]
        for i, request in enumerate(self.completed_requests):
            TTFT = (request.data[0].finished_time - request.metrics.arrival_time)
            sum_TTFT += TTFT
            total_input += request.input_len - request.past_context - request.remaining_prefill_tokens
            if request.gen_tokens > 1:
                actual_output_lens += request.gen_tokens
                for j in range(1, len(request.data)):
                    request_gen_times.append(request.data[j].finished_time - request.data[j].scheduled_time)
            latency = request.metrics.finished_time - request.metrics.arrival_time
            TTFT_latencies.append(TTFT)
            e2e_latencies.append(latency)

        num_requests = self.request_serviced
        if num_requests:
            avg_TTFT = sum(TTFT_latencies) / len(TTFT_latencies) if TTFT_latencies else 0
            avg_TPOT = sum(request_gen_times) / len(request_gen_times) if request_gen_times else 0
            interactivity = 1000/avg_TPOT if avg_TPOT > 0 else 0
            rps = self.request_serviced / (self.global_time/1000)

            e2e_latencies.sort()
            T50_latency = e2e_latencies[int(0.5 * num_requests)]
            T90_latency = e2e_latencies[int(0.9 * num_requests)]
            T95_latency = e2e_latencies[int(0.95 * num_requests)]
            T99_latency = e2e_latencies[int(0.99 * num_requests)]

            return EngineMetrics(
                TTFT=avg_TTFT,
                TPOT=avg_TPOT,
                rps=rps,
                T50_latency=T50_latency,
                T90_latency=T90_latency,
                T95_latency=T95_latency,
                T99_latency=T99_latency,
                interactivity=interactivity,
                output_throughput = actual_output_lens / (self.global_time/1000),
                total_token_throughput=(total_input + actual_output_lens) / (self.global_time/1000),
            )
        else:
            return EngineMetrics(
                TTFT=0,
                TPOT=0,
                rps=0,
                T50_latency=0,
                T90_latency=0,
                T95_latency=0,
                T99_latency=0,
                interactivity=0,
                output_throughput=0,
                total_token_throughput=0,
            )



    def get_heavy_light_split(self, engine_type, req, load_type):
        def circular_left_shift(lst, indexes):
            """Perform a circular left shift on the specified indexes of the list."""
            if not indexes:
                return lst
            first = lst[indexes[0]]
            for i in range(len(indexes) - 1):
                lst[indexes[i]] = lst[indexes[i + 1]]
            lst[indexes[-1]] = first
            return lst

        if req_is_heavy(req, load_type):
            # Circular left shift on even indexes
            engine_id = self.engine_matcher[engine_type][0]
            even_indexes = [i for i in range(len(self.engine_matcher[engine_type])) if i % 2 == 0]
            self.engine_matcher[engine_type] = circular_left_shift(self.engine_matcher[engine_type], even_indexes)
        else:
            # Circular left shift on odd indexes
            engine_id = self.engine_matcher[engine_type][1]
            odd_indexes = [i for i in range(len(self.engine_matcher[engine_type])) if i % 2 != 0]
            self.engine_matcher[engine_type] = circular_left_shift(self.engine_matcher[engine_type], odd_indexes)
        return engine_id

    def get_load_balanced_engine(self, engine_type, load_type):
        min_tokens = np.inf
        engine_id = -1
        for engine in self.engine_matcher[engine_type]:
            engine_load = self.GenA_engines[engine].get_load(self.global_time, load_type)
            if  engine_load < min_tokens:
                min_tokens = engine_load
                engine_id = engine
        return engine_id

    ## Overloadable functions
    def _determine_dst_engine(self, engine_type:EngineType, request=-1):
        """
            This function is designed to determine the destination engine based on the engine type.
            By default, it uses a round robin policy.
            This function can be overloaded to use different coordinator policies
        """
        if engine_type in self.engine_matcher:
            if self.router_type == CoordRouterType.ROUND_ROBIN:
                ## Round robin policy
                # Find the least recently used engine_id
                ## If the request has a preference, use that
                if isinstance(request, Request) and  request.get_engine_preference():
                    engine_ids_requested = request.get_engine_preference()
                    engine_id = next((engine for engine in self.engine_matcher[engine_type] if engine in engine_ids_requested), None)
                    if engine_id is not None:
                        self.engine_matcher[engine_type].remove(engine_id)
                    if engine_id is None:
                        raise ValueError(f"No matching engine found for the requested engine IDs: {engine_ids_requested}")
                ## Else use the round robin policy
                else:
                    engine_id = self.engine_matcher[engine_type].pop(0)
                # Rotate the list by 1
                self.engine_matcher[engine_type].append(engine_id)
            # We assume that the total number of output tokens is known beforehand for the request and we balance the sum of decode tokens on both the model instances.
            elif self.router_type == CoordRouterType.DECODE_BALANCER:
                ## Find the engine with the least number of decode tokens
                engine_id = self.get_load_balanced_engine( engine_type, LoadTypes.DECODE_TOKENS)
            # Dedicated half instance for servicing only the heavy-decode requests and other half services only the light-decode requests.
            elif self.router_type == CoordRouterType.DECODE_HEAVY_LIGHT:
                engine_id = self.get_heavy_light_split( engine_type, request, LoadTypes.DECODE_TOKENS)
            # Keep number of prefill tokens balanced across all the model instances
            elif self.router_type == CoordRouterType.PREFILL_BALANCER:
                ## Find the engine with the least number of prefill tokens
                engine_id = self.get_load_balanced_engine(engine_type, LoadTypes.PREFILL_TOKENS)
            # Dedicated instance for heavy prefills and the other for light prefills
            elif self.router_type == CoordRouterType.PREFILL_HEAVY_LIGHT:
                engine_id = self.get_heavy_light_split(engine_type, request, LoadTypes.PREFILL_TOKENS)
            # Keep load balanced across all the model instances
            elif self.router_type == CoordRouterType.CHUNKED_LOAD_BALANCER:
                engine_id = self.get_load_balanced_engine(engine_type, LoadTypes.TOKEN_PROCESSING)
            # Dedicated half machines for heavy loads and other half for light loads
            elif self.router_type == CoordRouterType.CHUNKED_HEAVY_LIGHT:
                engine_id = self.get_heavy_light_split(engine_type, request, LoadTypes.TOKEN_PROCESSING)
            elif self.router_type == CoordRouterType.ORACLE_BALANCER:
                engine_id = self.get_load_balanced_engine(engine_type, LoadTypes.ORACLE_DECODE)
            elif self.router_type == CoordRouterType.RUNTIME_LOAD_BALANCER:
                engine_id = self.get_load_balanced_engine(engine_type, LoadTypes.SIM_TIME)
            else:
                raise ValueError(f"Invalid router type {self.router}")
            # print(f"Request {request.request_id}, {request.input_len}/{request.output_len} for {engine_type}: Sent to engine {engine_id}")
            return engine_id
        else:
            raise ValueError(f"No engine type found for the stage:{engine_type}")

    def _update_most_recent_engine(self, engine_type, engine_id):
        """
            The input includes the engine type and the engine id that was most recently used.
        """
        assert engine_id != -1 and engine_id is not None, "Invalid engine id to update the most recent used"
        # Find the least recently use
        self.engine_matcher[engine_type].remove(engine_id)
        # Rotate the list by 1
        self.engine_matcher[engine_type].append(engine_id)

    def compute_bw_latency(self, df: pd.DataFrame, src, dst):
        # Filter the row corresponding to the given src and dst
        row = df[(df['src'] == src) & (df['dst'] == dst)]

        if row.empty:
            raise ValueError(f"No connection found from src:{src} to dst:{dst}")

        ## Assuming no congestion, we can use the minimum bandwidth and total latency
        # Extract bandwidth and latency columns
        bw_columns = [col for col in df.columns if col.startswith("BW")]
        latency_columns = [col for col in df.columns if col.startswith("latency")]

        # Compute minimum bandwidth and total latency
        min_bw = row[bw_columns].min(axis=1).values[0]
        total_latency = row[latency_columns].sum(axis=1).values[0]

        return total_latency, min_bw

    # TODO: Use this as a movement between engine
    def _get_move_request_time(self, request, src_engine, dst_engine) -> float:
        # Get the cost of moving the request from src to dst engine
        data_size = request.get_data_movement_size(self.GenA_engines[dst_engine].model)/2**20   # Convert to MB
        if src_engine == dst_engine:
            # Update the KV Cache Transfer Time
            if request.current_stage == RequestStage.PREFILL:
                request.metrics.kv_transfer_time = 0
            
            return 0
        else:
            latency, BW = self.compute_bw_latency(self.engine_connection_bw_latency_df, src_engine, dst_engine)
            # print(f'Req:{request.request_id}, Src: {src_engine}, Dst: {dst_engine}, Data Size: {data_size}, Latency: {latency}, BW: {BW}')
            if request.current_stage == RequestStage.PREFILL:
                request.metrics.kv_transfer_time = latency + data_size/BW
            return latency + data_size/BW

    ## Overloadable functions
    def initial_engine_connection_df(self):
        # Initialize the engine connection dataframe
        ## All BWs should be in GB/s and latencies in msec
        data = []
        for src_engine in range(self.num_total_engines):
            for dst_engine in range(self.num_total_engines):
                if src_engine == dst_engine:
                    data.append([src_engine, dst_engine, 0,  0])
                else:
                    data.append([src_engine, dst_engine, 5, 10])
        self.engine_connection_bw_latency_df = pd.DataFrame(data, columns=['src', 'dst', 'latency(msec)', 'BW(GB/s)'])

    def initial_engine_connection_from_file(self):
        file_name = self.network_file
        if isinstance(file_name, str) and file_name.endswith('.csv'):
            with open(file_name, 'r') as f:
                network_fields = f.readline().strip()  # First line has names (may have multiple fields for latency)
                column_names = network_fields.split(',')
            self.engine_connection_bw_latency_df = pd.read_csv(file_name, skiprows=1, names=column_names)
        else:
            raise ValueError(f"Invalid file name {file_name}")


    def _add_request_to_engine(self, request, engine_id):
        # The arrival time is based on priority time
        self.GenA_engines[engine_id].add_request(request, request.priority_time)
        None





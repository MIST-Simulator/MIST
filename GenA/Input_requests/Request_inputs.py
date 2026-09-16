import random
import numpy as np
from scipy.stats import poisson, norm
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
from GenA.Request.request import Request, RequestStage
from dataclasses import dataclass, field
import pandas as pd
from pandas import read_csv

@dataclass
class LengthVariables:
    length: int = None
    variance: int = None
    max_length: int = None

class RequestDistributions:
    def __init__(self,
                rand_seed : Optional[int] = 259,
                input_vars: Optional[LengthVariables] = None,
                output_vars: Optional[LengthVariables] = None,
                trace_file: str = None,
                num_requests: int = np.inf,
                n: int = 0,
                **kwargs
                ) -> None:
        self.request_queue = []
        self.rand_seed = rand_seed
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.trace_file = trace_file
        if self.trace_file:
            trace_df = read_csv(self.trace_file)
            if 'TIMESTAMP' in trace_df.columns:
                trace_df['arrival_time'] = pd.to_datetime(trace_df['TIMESTAMP'])
                min_time = trace_df['arrival_time'].min()
                trace_df['arrival_time'] = (trace_df['arrival_time'] - min_time).dt.total_seconds()
            self.max_data_len = len(trace_df)
        else:
            trace_df = None
        self.trace_df = trace_df

        self.n = n
        self.i = 0
        if 'beam_size' in kwargs:
            self.beam_size = kwargs['beam_size']
            del kwargs['beam_size']
        else:
            self.beam_size = 1
        self.req_args = kwargs
        self.num_requests = num_requests
    def generate_distribution(self) -> None:
        pass


    def get_requests(self, current_time:float) -> List[Request]:
        '''
        Returns all the request that are generated before the current time.
        '''
        inject_queue = []
        while self.request_queue:
            req = self.request_queue[0]
            if req.metrics.arrival_time <= current_time:
                inject_queue.append(req)
                self.request_queue.pop(0)
            else:
                break
        return inject_queue

    def get_next_requests(self) -> Request:
        '''
        Returns the next request in the queue.
        '''

        if self.request_queue:
            new_req = self.request_queue.pop(0)
            return new_req
        else:
            return None

    def get_input_token_size(self):
        # random.seed(self.rand_seed)  # Set a fixed seed
        # np.random.seed(self.rand_seed)
        if self.trace_df is not None:
            if self.i >= self.max_data_len:
                self.i = 0
            return self.trace_df.iloc[self.i]['ContextTokens']
        else:
            input_variance = self.input_vars.variance
            input_length = self.input_vars.length
            max_input_length = self.input_vars.max_length

            if input_length is not None and input_variance is not None:
                input_length = abs(int(np.random.normal(input_length, input_variance)))
            else:
                return random.randint(100, 1000)

            return min(input_length, max_input_length) if max_input_length is not None else input_length

    def get_output_token_size(self):
        # random.seed(self.rand_seed)  # Set a fixed seed
        # np.random.seed(self.rand_seed)
        if self.trace_df is not None:
            return self.trace_df.iloc[self.i]['GeneratedTokens']
        else:
            output_variance = self.output_vars.variance
            output_length = self.output_vars.length
            max_output_length = self.output_vars.max_length

            if output_length is not None and output_variance is not None:
                output_length = abs(int(np.random.normal(output_length, output_variance)))
            else:
                return random.randint(10, 100)

            return min(output_length, max_output_length) if max_output_length is not None else output_length

    def get_beam_size(self):
        return self.beam_size

class UniformDistribution(RequestDistributions):

    def __init__(self,
                rps: float = 1,
                sim_time :float = 1000, **kwargs) -> None:
        self.rps = rps
        self.sim_time = sim_time
        super().__init__(**kwargs)
        self.generate_distribution(rps, sim_time)


    def generate_distribution(self, rps: float, sim_time: float) -> None:
        """Adding Request to the engine in an uniform schedule

        Args:
            rps (float): Requests per second

        Returns:
            None
        """
        request_interval = 1000 / rps   ## Time between 2 requests
        req_init_time = 0
        while req_init_time < sim_time and self.num_requests > self.i:
            input_len = self.get_input_token_size()
            output_len = self.get_output_token_size()
            beam_size = self.get_beam_size()
            self.i += 1
            self.request_queue.append(Request(input_len=input_len, output_len=output_len,
                            arrival_time = req_init_time, beam_size=beam_size, **self.req_args))
            req_init_time += request_interval

def generate_poisson_distribution(rps, sim_time_ms):
    # Convert sim_time to seconds
    sim_time_s = sim_time_ms / 1000

    # Calculate expected number of events
    expected_events = rps * sim_time_s

    # Generate Poisson distribution
    poisson_dist = poisson(mu=expected_events)

    # Generate random samples
    num_samples = int(expected_events)
    samples = poisson_dist.rvs(size=num_samples)

    # Scale samples to fit within sim_time_ms
    scaled_samples = samples * (sim_time_ms / samples.max())

    # Sort the samples
    sorted_samples = np.sort(scaled_samples)

    return sorted_samples.astype(int)

class PoissonDistribution(RequestDistributions):
    def __init__(self,
                rps: float = 1,
                sim_time :float = 1000, **kwargs) -> None:
        self.rps = rps
        self.sim_time = sim_time
        super().__init__(**kwargs)
        self.generate_distribution(rps, sim_time)

    def generate_distribution(self, rps: float, sim_time: float) -> None:
        """Adding Request to the engine in a Poisson distribution schedule
        Args:
            rps (float): Average requests per second
            If this is inf, all requests are sent at time 0. Otherwise, we take
            1 divided by this argument value to be the parameter of the Poisson
            distribution for modeling the request arrival times.
        Returns:
            None
        """
        random.seed(self.rand_seed)  # Set a fixed seed
        np.random.seed(self.rand_seed)

        req_init_time = 0
        while req_init_time < sim_time and self.num_requests > self.i:
            input_len = self.get_input_token_size()
            output_len = self.get_output_token_size()
            beam_size = self.get_beam_size()
            self.i += 1
            self.request_queue.append(Request(input_len=input_len, output_len=output_len,
                            arrival_time = req_init_time, beam_size=beam_size, **self.req_args))
            req_init_time += np.random.exponential(1.0 / rps) * 1000

class NormalDistribution(RequestDistributions):
    def __init__(self,
                rps: float = 1,
                sim_time :float = 1000, **kwargs) -> None:
        self.rps = rps
        self.sim_time = sim_time
        super().__init__(**kwargs)
        self.generate_distribution(rps, sim_time)

    def generate_distribution(self, rps: float, sim_time: float) -> None:
        """Adding Request to the engine in a Normal distribution schedule
        Args:
            rps (float): Average requests per second
        Returns:
            None
        """
        random.seed(self.rand_seed)  # Set a fixed seed
        np.random.seed(self.rand_seed)

        mean_interval = 1000 / rps  # Mean time between requests in milliseconds
        std_dev = mean_interval / 4  # Standard deviation (you can adjust this)

        req_init_time = 0
        while req_init_time < sim_time and self.num_requests > self.i:
            input_len = self.get_input_token_size()
            output_len = self.get_output_token_size()
            beam_size = self.get_beam_size()
            self.i += 1
            self.request_queue.append(Request(input_len=input_len, output_len=output_len,
                            arrival_time=req_init_time, beam_size=beam_size, **self.req_args))

            # Generate next interval using Normal distribution
            interval = norm.rvs(loc=mean_interval, scale=std_dev)
            interval = max(interval, 0)  # Ensure non-negative interval

            req_init_time += interval

# class BurstyDistribution(RequestDistributions):
#     def __init__(self,
#                 rps: float = 1,
#                 sim_time :float = 1000, **kwargs) -> None:
#         self.rps = rps
#         self.sim_time = sim_time
#         self.burst_interval = kwargs.get('burst_interval', 1000)
#         super().__init__(**kwargs)
#         self.generate_distribution(rps, sim_time, self.burst_interval)

#     def generate_distribution(self, rps: float, sim_time: float, burst_interval: float=1000) -> None:
#         """Adding Request to engine in burst every second
#         Args:
#             rps (float): Average requests per second
#         Returns:
#             None
#         """
#         random.seed(self.rand_seed)  # Set a fixed seed
#         np.random.seed(self.rand_seed)

#         req_init_time = 0
#         while req_init_time < sim_time:
#             for _ in range(rps):
#                 input_len = self.get_input_token_size()
#                 output_len = self.get_output_token_size()
#                 beam_size = self.get_beam_size()
#                 self.request_queue.append(Request(input_len=input_len, output_len=output_len,
#                             arrival_time=req_init_time, beam_size=beam_size, **self.req_args))

#             # Generate next interval using Normal distribution
#             interval = burst_interval

#             req_init_time += interval


import numpy as np
import random

class BurstyDistribution(RequestDistributions):
    def __init__(self, rps: float = 1, sim_time: float = 1000, burst_freq: float = 0.1, burst_factor: float = 3, burst_width: float = 4, **kwargs) -> None:
        """
        Args:
            rps (float): Base requests per second
            sim_time (float): Total simulation time
            burst_freq (float): Frequency of bursts (how often peaks occur)
            burst_factor (float): Multiplier for burst intensity
            burst_width (float): Controls sharpness of the peak (higher = narrower peaks)
        """
        self.rps = rps
        self.sim_time = sim_time
        self.burst_freq = burst_freq
        self.burst_factor = burst_factor
        self.burst_width = burst_width  # Higher values make peaks sharper
        super().__init__(**kwargs)
        self.generate_distribution(rps, sim_time)

    def generate_distribution(self, rps: float, sim_time: float) -> None:
        """Generate a bursty request pattern with sharp peaks."""
        random.seed(self.rand_seed)
        time = 0

        while time < sim_time and self.num_requests > self.i:
            # Sharper peak using sin^burst_width to make bursts denser but shorter
            phase = (2 * np.pi * self.burst_freq * time / sim_time)
            burst_multiplier = 1 + (self.burst_factor - 1) * (np.sin(phase) ** self.burst_width if np.sin(phase) > 0 else 0)

            effective_rps = rps * burst_multiplier
            request_interval = 1000 / effective_rps  # Convert RPS to inter-arrival time

            input_len = self.get_input_token_size()
            output_len = self.get_output_token_size()
            beam_size = self.get_beam_size()
            self.i += 1
            self.request_queue.append(Request(
                input_len=input_len,
                output_len=output_len,
                arrival_time=time,
                beam_size=beam_size,
                **self.req_args
            ))

            time += request_interval  # Move to next request arrival


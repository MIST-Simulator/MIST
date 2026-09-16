from .Request_inputs import RequestDistributions
from mist.Request.request import Request
import pandas as pd
from pandas import read_csv

class TraceDistributions(RequestDistributions):

    def __init__(self, trace_file: str, n: int , **kwargs) -> None:
        ''' Reads a csv trace file and generates the request queue
        Args:
            trace_file (str): Path to the trace file, it has 3 columns:
                1. TIMESTAMP (datetime; converted to ms since the first request)
                2. Input tokens
                3. Output tokens
        '''
        self.request_queue = []
        super().__init__(trace_file=trace_file, n=n, **kwargs) # Call the parent class constructor
        self.generate_distribution(n)

    def generate_distribution(self, n:int = None) -> None:
        ''' Reads the trace file and generates the request queue
        '''
        trace_df = self.trace_df  # arrival_time (ms since first request) set by RequestDistributions
        num_requests = len(trace_df)
        if n is not None:
            num_requests = min(num_requests, n)
        for _, row in trace_df.iterrows():
            req = Request(
                arrival_time=row['arrival_time'],
                input_len=row['ContextTokens'],
                output_len=row['GeneratedTokens'],
                **self.req_args
            )
            self.request_queue.append(req)
            if len(self.request_queue) == num_requests:
                break

# TODO: Use this function to read in the request traces and turn it into a request queue
class TraceIngestion(RequestDistributions):
    def __init__(self, ingestion_trace_file: str, **kwargs) -> None:
        ''' Reads a csv trace file and generates the request queue
        Args:
            trace_file (str): Path to the trace file, it has 3 columns:
                1. arrival_timestamp (seconds; converted to ms)
                2. Input tokens
                3. Output tokens
        '''
        self.ingestion_trace_file = ingestion_trace_file
        self.request_queue = []
        super().__init__(**kwargs) # Call the parent class constructor
        self.convert_to_MIST_format()

    def convert_to_MIST_format(self):
        trace_df = read_csv(self.ingestion_trace_file)
        num_requests = len(trace_df)
        for idx, row in trace_df.iterrows():
            if 'request_id' not in row:
                req_id = idx
            else:
                req_id = row['request_id']
            # print(f"request id: {row['request_id']}, arr: {row['arrival_timestamp']}, input: {row['prompt_size']}, output: {row['token_size']}")
            req = Request(
                request_id=req_id,
                arrival_time=row['arrival_timestamp']*1000, # Convert to milliseconds
                input_len=row['prompt_size'],
                output_len=row['token_size'],
                **self.req_args
            )
            self.request_queue.append(req)
            if len(self.request_queue) == num_requests:
                break

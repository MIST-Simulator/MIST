import enum

class CoordRouterType(enum.Enum):
    ROUND_ROBIN = enum.auto()
    JOIN_SHORTEST_QUEUE = enum.auto() # Used by Splitwise for joining the shortest queue
    # We assume that the total number of output tokens is known beforehand for the request and we balance the sum of decode tokens on both the model instances.
    DECODE_BALANCER = enum.auto()

    # Dedicated half instance for servicing only the heavy-decode requests and other half services only the light-decode requests.
    DECODE_HEAVY_LIGHT = enum.auto()

    # Keep number of prefill tokens balanced across all the model instances
    PREFILL_BALANCER = enum.auto()

    # Dedicated instance for heavy prefills and the other for light prefills
    PREFILL_HEAVY_LIGHT = enum.auto()

    # Keep load balanced across all the model instances
    CHUNKED_LOAD_BALANCER = enum.auto()

    # Dedicated half machines for heavy loads and other half for light loads
    CHUNKED_HEAVY_LIGHT = enum.auto()

    # Keep num of future decode tokens to generate balanced across all the model instances
    ORACLE_BALANCER = enum.auto()

    # Runtime load balancing based on the current load on the model instances
    RUNTIME_LOAD_BALANCER = enum.auto()

class LoadTypes(enum.Enum):
    PREFILL_TOKENS = enum.auto()
    DECODE_TOKENS = enum.auto()
    TOKEN_PROCESSING = enum.auto()  # Total number of tokens being processed (Input len for prefill and 1 for decode.)
    KV_CACHE_SIZE = enum.auto() # Total size of KV caches generated till now
    SIM_TIME = enum.auto()  # Estimated step time for current queue
    ORACLE_DECODE = enum.auto() # Number of decode tokens left in the oracle

def req_is_heavy(req, load_type:LoadTypes):
    if load_type == LoadTypes.PREFILL_TOKENS:
        # TODO: If prefill time is larger than 0.5 second
        tokens = req.input_len
        if tokens > 2048:
            return True
    elif load_type == LoadTypes.DECODE_TOKENS:
        # TODO: If decode time is larger than 5 second
        tokens = req.output_len
        if tokens > 512:
            return True
    elif load_type == LoadTypes.TOKEN_PROCESSING:
        input_tokens = req.input_len
        output_tokens = req.output_len
        if input_tokens > 2048 or output_tokens > 512:
            return True

    return False
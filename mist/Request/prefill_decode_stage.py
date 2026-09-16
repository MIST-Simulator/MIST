# import enum
# from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple,
#                     Union, Set, Deque)
# from collections import deque
# from dataclasses import dataclass, field
# from .base_stage import Stage
# from .request import RequestStage, RequestStatus, RequestMetrics


# class Prefill(Stage):
#     """Prefill stage of the request.

#     Args:
#         input_len: Length of the Input Prompt.
#         output_len: Expected Len of the output.
#     """

#     def __init__(
#         self,
#         input_len: int = 100,
#     ) -> None:

#         self.input_len = input_len
#         self.remaining_prefill_tokens = input_len
#         self.current_scheduled_prefill = input_len
#         ## Scheduler Flags
#         self.status = RequestStatus.WAITING

#         super().__init__(stage_type=RequestStage.PREFILL)

#     def get_current_kv_length(self) -> int:
#         return self.input_len - self.remaining_prefill_tokens - self.current_scheduled_prefill


#     def get_num_new_tokens(self,
#                             enable_chunking: bool= False,
#                             chunk_size:int = None) -> int:
#         """Get the number of new tokens to be computed.

#         Returns:
#             The new number of tokens to be computed. I.e., BeamSize for decode, or
#             the prompt size for prefill.
#         """
#         if self.is_finished():
#             return 0
#         if enable_chunking:
#             assert chunk_size, "When chunking is enabled, chunk size should be an int."
#             return min(chunk_size, self.remaining_prefill_tokens)
#         else:
#             return self.input_len


#     def current_scheduled(self, tokens_scheduled: int) -> None:
#         """Update the number of tokens scheduled for the current request."""
#         self.remaining_prefill_tokens -= tokens_scheduled
#         self.current_scheduled_prefill = tokens_scheduled

#     def get_current_chunk_size(self) -> int:
#         return self.current_scheduled_prefill

#     def is_finished(self) -> bool:
#         return RequestStatus.is_finished(self.status)

#     def is_prefill(self) -> bool:
#         return (self.stage == RequestStage.PREFILL) and (self.remaining_prefill_tokens > 0)

#     def __rps__(self) -> str:
#         a = f'Req Id:{self.request_id}, Input Len:{self.input_len}, Output Len:{self.output_len}, Beams:{self.beam_size}\n'
#         b = f'Current Status:{self.status}'
#         return a+b

#     def __str__(self):
#         a = f'Req Id:{self.request_id}, Input Len:{self.input_len}, Output Len:{self.output_len}, Beams:{self.beam_size}\n'
#         b = f'Current Status:{self.status}'
#         return a+b


# class Decode(Stage):
#     """A group of sequences that are generated from the same prompt.

#     Args:
#         request_id: The ID of the request.
#         input_len: Length of the Input Prompt.
#         output_len: Expected Len of the output.
#     """

#     def __init__(
#         self,
#         input_len: int = 100,
#         output_len: int = 100,
#     ) -> None:
#         ## Set at init time

#         ## Update at each generated token
#         self.data = []
#         self.gen_tokens = 0

#         self.input_len = input_len
#         self.output_len = output_len

#         ## Scheduler Flags
#         self.status = RequestStatus.WAITING

#         super().__init__(stage_type=RequestStage.DECODE)

#     def get_current_kv_length(self) -> int:
#         return self.input_len + self.gen_tokens

#     def add_token(self, data_beat)-> None:
#         self.data.append(data_beat)
#         self.gen_tokens += 1

#     def get_num_new_tokens(self,
#                             enable_chunking: bool= False,
#                             chunk_size:int = None) -> int:
#         """Get the number of new tokens to be computed.

#         Returns:
#             The new number of tokens to be computed. I.e., BeamSize for decode, or
#             the prompt size for prefill.
#         """
#         if self.is_finished():
#             return 0
#         else:
#             return self.beam_size


#     def is_finished(self) -> bool:
#         return RequestStatus.is_finished(self.status)

#     def is_prefill(self) -> bool:
#         return False


#     def __rps__(self) -> str:
#         a = f'Req Id:{self.request_id}, Input Len:{self.input_len}, Output Len:{self.output_len}, Beams:{self.beam_size}\n'
#         b = f'Current Status:{self.status}'
#         return a+b

#     def __str__(self):
#         a = f'Req Id:{self.request_id}, Input Len:{self.input_len}, Output Len:{self.output_len}, Beams:{self.beam_size}\n'
#         b = f'Current Status:{self.status}'
#         return a+b

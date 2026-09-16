import enum
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple,
                    Union, Set, Deque)
from collections import deque
from dataclasses import dataclass, field
from .base_stage import Stage
from .request import RequestStage, RequestStatus, RequestMetrics

class Preprocess(Stage):
    """Preprocess stage of the request.

    Args:
        input_len: Length of the Input Prompt.
    """

    def __init__(
        self,
        input_len: int = 100,
    ) -> None:

        self.input_len = input_len

        super().__init__(stage_type=RequestStage.PREPROCESS)

class Postprocess(Stage):
    """Postprocess stage of the request.

    Args:
        input_len: Length of the Input Prompt.
    """

    def __init__(
        self,
        input_len: int = 100,
        output_len: int = 100,
    ) -> None:

        self.input_len = input_len
        self.output_len = output_len

        super().__init__(stage_type=RequestStage.POSTPROCESS)
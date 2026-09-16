import enum
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple,
                    Union, Set, Deque)
from collections import deque
from dataclasses import dataclass, field
# from .request import RequestStage



class Stage:
    """A stage of a request.

    Args:
        stage_type: The type of the stage.
        stage_metrics: The metrics associated with the stage.
    """

    def __init__(
        self,
        stage_id: int,
        stage_type: int,
        stage_metrics: Optional[StageMetrics] = StageMetrics(),
    ) -> None:
        self.stage_id = stage_id  # Unique ID for each stage instance
        self.stage_type = stage_type
        self.stage_metrics = stage_metrics
        self.engine_id = -1

    def assign_engine(self, engine_id: int, entry_time: float) -> None:
        """Assign an engine to the stage.

        Args:
            engine_id: The ID of the engine.
            entry_time: The time when the stage was sent to the engine.
        """
        self.engine_id = engine_id
        self.stage_metrics.engine_entry_time = entry_time

    def stage_start(self, current_time: float) -> None:
        """Start the stage.

        Args:
            current_time: The current time.
        """
        self.stage_metrics.scheduled_time = current_time

    def stage_finish(self, current_time: float) -> None:
        """Finish the stage.

        Args:
            current_time: The current time.
        """
        self.stage_metrics.finished_time = current_time

    def __repr__(self):
        return f"Stage({self.stage_id}, {self.stage_type.name})"

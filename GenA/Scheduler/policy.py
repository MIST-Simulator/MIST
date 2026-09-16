from collections import deque
from typing import Deque

from GenA.Request.request import Request

class Policy:

    def get_priority(
        self,
        now: float,
        req: Request,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[Request],
    ) -> Deque[Request]:
        return deque(
            sorted(
            seq_groups,
            key=lambda seq_group: (self.get_priority(now, seq_group), -seq_group.request_id),
            reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        req: Request,
    ) -> float:
        return now - req.metrics.arrival_time




class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)

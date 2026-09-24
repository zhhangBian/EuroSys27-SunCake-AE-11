# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request
from vllm.mcp.agent_info import AgentInfoManager


class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
    AGENT = "agent"

    @staticmethod
    def get_policy(policy: str) -> SchedulingPolicy:
        if policy == "priority":
            return SchedulingPolicy.PRIORITY
        elif policy == "fcfs":
            return SchedulingPolicy.FCFS
        elif policy == "agent":
            return SchedulingPolicy.AGENT
        else:
            raise ValueError(f"Unknown scheduling policy: {policy}")


class RequestQueue(ABC):
    def __init__(self, agent_info_manager: AgentInfoManager):
        self.agent_info_manager = agent_info_manager

    @abstractmethod
    def add_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def add_requests(self, requests: list[Request]) -> None:
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        pass

    @abstractmethod
    def get_all_requests(self) -> list[Request]:
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    def __init__(self, agent_info_manager: AgentInfoManager):
        deque.__init__(self)
        RequestQueue.__init__(self, agent_info_manager)

    def add_request(self, request: Request) -> None:
        self.append(request)

    def add_requests(self, requests: list[Request]) -> None:
        self.extend(requests)

    def pop_request(self) -> Request:
        return self.popleft()

    def peek_request(self) -> Request:
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        self.clear()
        self.extend(filtered_requests)

    def get_all_requests(self) -> list[Request]:
        return list(self)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):

    def __init__(self, agent_info_manager: AgentInfoManager) -> None:
        super().__init__(agent_info_manager)
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))

    def add_requests(self, requests: list[Request]) -> None:
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap
                      if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def get_all_requests(self) -> list[Request]:
        return [r for _, _, r in self._heap]

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        return reversed(list(self))


class AgentRequestQueue(PriorityRequestQueue):
    def _heap_item(self, request: Request) -> tuple[int, float, Request]:
        return (-self.get_agent_priority(request), request.arrival_time, request)

    def _refresh_priorities(self) -> None:
        if not self._heap:
            return
        self._heap = [self._heap_item(request) for _, _, request in self._heap]
        heapq.heapify(self._heap)

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, self._heap_item(request))

    def add_requests(self, requests: list[Request]) -> None:
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> Request:
        self._refresh_priorities()
        return super().pop_request()

    def peek_request(self) -> Request:
        self._refresh_priorities()
        return super().peek_request()

    def get_all_requests(self) -> list[Request]:
        self._refresh_priorities()
        return super().get_all_requests()

    def __iter__(self) -> Iterator[Request]:
        self._refresh_priorities()
        return super().__iter__()

    def get_agent_priority(self, request: Request) -> int:
        agent_info = request.agent_info or {}
        static_priority = float(agent_info.get("priority", request.priority) or 0)
        depth = float(agent_info.get("depth", static_priority) or 0)
        out_degree = float(agent_info.get("out_degree", 0) or 0)
        in_degree = float(agent_info.get("in_degree", 0) or 0)
        similarity = float(agent_info.get("similarity", 0) or 0)
        parallel_width = 1.0 / similarity if 0.0 < similarity < 1.0 else 1.0
        parallel_join_pressure = max(0.0, parallel_width - 1.0)
        app_max_depth = max(
            depth,
            float(agent_info.get("app_max_depth", depth) or depth or 1.0),
            1.0,
        )
        remaining_depth = max(
            0.0,
            float(agent_info.get("remaining_depth", app_max_depth - depth)
                  or 0),
        )
        remaining_ratio = min(1.0, remaining_depth / app_max_depth)
        progress_ratio = min(1.0, max(0.0, depth / app_max_depth))
        app_elapsed = float(agent_info.get("app_elapsed_time", 0) or 0)
        app_start_time = float(agent_info.get("app_start_time", 0) or 0)
        if app_start_time > 0:
            app_elapsed = max(app_elapsed, time.time() - app_start_time)
        app_start_offset = float(agent_info.get("app_start_offset", 0) or 0)
        queue_wait = max(0.0, time.time() - request.arrival_time)
        critical_path = bool(agent_info.get("critical_path", False))
        near_completion = bool(agent_info.get("near_completion", False))
        join_group = str(agent_info.get("join_group", "") or "")
        dependency_depth = float(
            agent_info.get("dependency_depth", depth) or depth)
        fanout_width = max(
            1.0, float(agent_info.get("fanout_width", parallel_width) or 1.0))
        memory_weight = max(
            0.0, float(agent_info.get("memory_weight", 1.0) or 0.0))

        app_age_pressure = min(app_elapsed, 300.0) * (1.0 + 0.75 * remaining_ratio)
        queue_pressure = min(queue_wait, 180.0) * (1.0 + 0.50 * remaining_ratio)
        completion_pressure = min(app_elapsed, 240.0) * progress_ratio
        priority_score = (
            100.0 * static_priority +
            15.0 * depth +
            45.0 * out_degree +
            10.0 * in_degree +
            20.0 * similarity +
            20.0 * remaining_depth +
            120.0 * parallel_join_pressure +
            6.0 * min(app_start_offset, 60.0) +
            5.0 * app_age_pressure +
            8.0 * queue_pressure +
            5.0 * completion_pressure +
            180.0 * critical_path +
            90.0 * near_completion +
            20.0 * bool(join_group) +
            4.0 * dependency_depth +
            12.0 * max(0.0, fanout_width - 1.0) -
            8.0 * max(0.0, memory_weight - 1.0))
        return int(priority_score)


def create_request_queue(policy: SchedulingPolicy, agent_info_manager: AgentInfoManager) -> RequestQueue:
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue(agent_info_manager)
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue(agent_info_manager)
    elif policy == SchedulingPolicy.AGENT:
        return AgentRequestQueue(agent_info_manager)
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")

import time

from dataclasses import dataclass

from vllm.v1.request import Request
from vllm.logger import init_logger

logger = init_logger(__name__)

@dataclass
class AgentInfo:
    request_id: str
    name: str
    type: str
    priority: int = 0
    depth: int = 0
    in_degree: int = 0
    out_degree: int = 0
    similarity: float = 0.0
    input_len: int = 0
    output_len: int = 0
    start_time: float = 0
    application_id: str = ""
    app_start_time: float = 0.0
    app_start_offset: float = 0.0
    app_elapsed_time: float = 0.0
    app_max_depth: float = 0.0
    remaining_depth: float = 0.0
    branch_id: str = ""
    branch_group: str = ""
    critical_path: bool = False
    offload_eligible: bool = False
    expected_tool_stall_s: float = 0.0
    memory_weight: float = 1.0
    stage_type: str = ""
    near_completion: bool = False
    fanout_width: int = 1
    join_group: str = ""
    dependency_depth: float = 0.0
    end_time: float = 0
    time: float = 0

    def update_input_len(self, input_len: int):
        self.input_len = input_len

    def update_output_len(self, output_len: int):
        self.output_len = output_len
        self.end_time = time.time()
        self.time = float(self.end_time) - float(self.start_time)

class AgentInfoManager:
    def __init__(self):
        self.agent_info_dict: dict[str, dict[str, dict[str, AgentInfo]]] = {}
        self.reqid_to_agent_info: dict[str, AgentInfo] = {}
        self.agent_priority_dict: dict[str, float] = {}

    def get_agent_priority_dict(self) -> dict[str, float]:
        return self.agent_priority_dict

    def add_agent_info(self, agent_info: AgentInfo):
        if agent_info.type not in self.agent_info_dict:
            self.agent_info_dict[agent_info.type] = {}
        if agent_info.name not in self.agent_info_dict[agent_info.type]:
            self.agent_info_dict[agent_info.type][agent_info.name] = {}
        self.agent_info_dict[agent_info.type][agent_info.name][agent_info.request_id] = agent_info
        self.reqid_to_agent_info[agent_info.request_id] = agent_info

    def has_agent_info(self, request: Request) -> bool:
        if request.agent_type not in self.agent_info_dict:
            return False
        if request.agent_name not in self.agent_info_dict[request.agent_type]:
            return False
        return request.request_id in self.agent_info_dict[request.agent_type][request.agent_name]

    def update_req_agent_input_info(self, request: Request, input_len: int):
        if self.has_agent_info(request):
            self.agent_info_dict[request.agent_type][request.agent_name][request.request_id].update_input_len(input_len)
        else:
            self.add_agent_info(AgentInfo(
                request_id=request.request_id,
                name=request.agent_name,
                type=request.agent_type,
                priority=request.priority,
                depth=int(request.agent_info.get("depth", 0) or 0),
                in_degree=int(request.agent_info.get("in_degree", 0) or 0),
                out_degree=int(request.agent_info.get("out_degree", 0) or 0),
                similarity=float(request.agent_info.get("similarity", 0) or 0),
                input_len=input_len,
                application_id=str(request.agent_info.get("application_id", "")),
                app_start_time=float(request.agent_info.get("app_start_time", 0) or 0),
                app_start_offset=float(request.agent_info.get("app_start_offset", 0) or 0),
                app_elapsed_time=float(request.agent_info.get("app_elapsed_time", 0) or 0),
                app_max_depth=float(request.agent_info.get("app_max_depth", 0) or 0),
                remaining_depth=float(request.agent_info.get("remaining_depth", 0) or 0),
                branch_id=str(request.agent_info.get("branch_id", "")),
                branch_group=str(request.agent_info.get("branch_group", "")),
                critical_path=bool(request.agent_info.get("critical_path", False)),
                offload_eligible=bool(request.agent_info.get("offload_eligible", False)),
                expected_tool_stall_s=float(request.agent_info.get("expected_tool_stall_s", 0) or 0),
                memory_weight=float(request.agent_info.get("memory_weight", 1.0) or 1.0),
                stage_type=str(request.agent_info.get("stage_type", "")),
                near_completion=bool(request.agent_info.get("near_completion", False)),
                fanout_width=int(request.agent_info.get("fanout_width", 1) or 1),
                join_group=str(request.agent_info.get("join_group", "")),
                dependency_depth=float(request.agent_info.get("dependency_depth", 0) or 0),
                start_time=request.agent_info.get("start_time", time.time())))
        self.agent_priority_dict[request.agent_type] = request.priority

    def update_req_agent_output_info(self, request: Request, output_len: int):
        self.agent_priority_dict[request.agent_type] = request.priority
        if self.has_agent_info(request):
            self.agent_info_dict[request.agent_type][request.agent_name][request.request_id].update_output_len(output_len)
        else:
            logger.warn(f"request {request.request_id} not found in agent_info_dict")

    def _get_agent_type_list(self, type: str) -> list[AgentInfo]:
        if type not in self.agent_info_dict:
            return []
        return [
            agent_info \
            for agent_name_list in self.agent_info_dict[type].values()
            for agent_info in agent_name_list.values()
        ]

    def get_ave_type_input_len(self, type: str) -> float:
        type_list = self._get_agent_type_list(type)
        return sum(agent_info.input_len for agent_info in type_list) / len(type_list) if type_list else 0

    def get_ave_type_output_len(self, type: str) -> float:
        type_list = self._get_agent_type_list(type)
        return sum(agent_info.output_len for agent_info in type_list) / len(type_list) if type_list else 0

    def get_ave_type_time(self, type: str) -> float:
        type_list = self._get_agent_type_list(type)
        time_list = [agent_info.time for agent_info in type_list if agent_info.time > 0]
        return sum(time_list) / len(time_list) if time_list else 0

    def get_ave_past_info(self, type: str) -> tuple[float, float, float]:
        ave_input_len = self.get_ave_type_input_len(type)
        ave_output_len = self.get_ave_type_output_len(type)
        ave_time = self.get_ave_type_time(type)
        return ave_input_len, ave_output_len, ave_time

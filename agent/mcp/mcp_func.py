from typing import List

from agent.mcp.static import (MCPFunctionType, MCPFunctionValueType,
                              McpExcuteStageType)


class McpExcuteStage:
    def __init__(
        self,
        stage_name: str,
        stage_type: McpExcuteStageType,
        excute_time: float,
        **kvargs
    ):
        self.stage_name = stage_name
        self.stage_type = stage_type
        self.excute_time = excute_time
        self.kvargs = kvargs

    def to_dict(self):
        return {
            "stage_name": self.stage_name,
            "stage_type": self.stage_type,
            "excute_time": self.excute_time,
            **self.kvargs
        }


class McpFunction:
    def __init__(
        self,
        name: str,
        description: str,
        excute_time: float,
        excute_stages: List[McpExcuteStage],
        function_value_type: MCPFunctionValueType,
        function_type: MCPFunctionType,
        **kvargs
    ):
        self.name = name
        self.description = description
        self.excute_time = self._get_excute_time(excute_stages, excute_time)
        self.excute_stages = excute_stages
        self.function_value_type = function_value_type
        self.function_type = function_type
        self.kvargs = kvargs

    def _get_excute_time(self, excute_stages: List[McpExcuteStage], excute_time: float):
        return sum(stage.excute_time for stage in excute_stages) + excute_time

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "excute_time": self.excute_time,
            "excute_stages": {
                stage.stage_name: stage.to_dict() for stage in self.excute_stages
            },
            "function_value_type": self.function_value_type,
            "function_type": self.function_type,
            **self.kvargs
        }


class McpFileWrite(McpFunction):
    def __init__(
        self,
        file_path: str,
        file_content: str,
        **kvargs
    ):
        super().__init__(
            name="file_write",
            description=f"[FILE_WRITE]: write file to {file_path}",
            excute_time=1,
            excute_stages=[
                McpExcuteStage(
                    stage_name="file_open",
                    stage_type=McpExcuteStageType.FILE_OPEN,
                    excute_time=0.1
                ),
                McpExcuteStage(
                    stage_name="file_write",
                    stage_type=McpExcuteStageType.FILE_WRITE,
                    excute_time=0.5
                ),
                McpExcuteStage(
                    stage_name="file_close",
                    stage_type=McpExcuteStageType.FILE_CLOSE,
                    excute_time=0.1
                )
            ],
            function_value_type=MCPFunctionValueType.NOT_NEED_VALUE,
            function_type=MCPFunctionType.FILE_WRITE,
            **kvargs
        )
        self.file_path = file_path
        self.file_content = file_content

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "file_content": self.file_content,
            **super().to_dict()
        }

class McpConditionalExecution(McpFunction):
    def __init__(
        self,
        condition_description: str,
        **kvargs
    ):
        super().__init__(
            name="conditional_execution",
            description=f"[CONDITIONAL_EXECUTION]: conditional execution based on {condition_description}",
            excute_time=0.2,
            excute_stages=[
                McpExcuteStage(
                    stage_name="condition_check",
                    stage_type=McpExcuteStageType.CONDITION_CHECK,
                    excute_time=0.1
                ),
                McpExcuteStage(
                    stage_name="conditional_execution",
                    stage_type=McpExcuteStageType.CONDITIONAL_EXECUTION,
                    excute_time=0.1
                )
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.CONDITION_CALL,
            **kvargs
        )
        self.condition_description = condition_description

    def to_dict(self):
        return {
            "condition_description": self.condition_description,
            **super().to_dict()
        }


class McpSearch(McpFunction):
    def __init__(
        self,
        search_query: str,
        search_type: str = "general",
        **kvargs
    ):
        super().__init__(
            name="search",
            description=f"[SEARCH]: perform {search_type} search for '{search_query}'",
            excute_time=2.0,
            excute_stages=[
                McpExcuteStage(
                    stage_name="search_query_processing",
                    stage_type=McpExcuteStageType.QUERY_ANALYSIS,
                    excute_time=0.5
                ),
                McpExcuteStage(
                    stage_name="search_execution",
                    stage_type=McpExcuteStageType.NET_SEARCH,
                    excute_time=1.0
                ),
                McpExcuteStage(
                    stage_name="search_result_processing",
                    stage_type=McpExcuteStageType.DATA_ANALYSIS,
                    excute_time=0.5
                )
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.FILE_HYBRID,
            **kvargs
        )
        self.search_query = search_query
        self.search_type = search_type

    def to_dict(self):
        return {
            "search_query": self.search_query,
            "search_type": self.search_type,
            **super().to_dict()
        }


class McpToolCall(McpFunction):
    def __init__(
        self,
        tool_name: str,
        tool_args: dict = None,
        **kvargs
    ):
        if tool_args is None:
            tool_args = {}

        super().__init__(
            name="tool_call",
            description=f"[TOOL_CALL]: call tool '{tool_name}' with args {tool_args}",
            excute_time=1.0,
            excute_stages=[
                McpExcuteStage(
                    stage_name="tool_initialization",
                    stage_type=McpExcuteStageType.TOOL_INITIALIZATION,
                    excute_time=0.3
                ),
                McpExcuteStage(
                    stage_name="tool_execution",
                    stage_type=McpExcuteStageType.TOOL_EXECUTION,
                    excute_time=0.5
                ),
                McpExcuteStage(
                    stage_name="tool_result_processing",
                    stage_type=McpExcuteStageType.DATA_ANALYSIS,
                    excute_time=0.2
                )
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.FILE_HYBRID,
            **kvargs
        )
        self.tool_name = tool_name
        self.tool_args = tool_args

    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            **super().to_dict()
        }


class McpExternalToolEval(McpFunction):
    def __init__(self, tool_name: str, eval_target: str, **kvargs):
        super().__init__(
            name="external_tool_eval",
            description=f"[EXTERNAL_TOOL_EVAL]: eval {eval_target} with {tool_name}",
            excute_time=1.0,
            excute_stages=[
                McpExcuteStage(
                    stage_name="tool_selection",
                    stage_type=McpExcuteStageType.OTHER,
                    excute_time=0.5
                ),
                McpExcuteStage(
                    stage_name="tool_execution",
                    stage_type=McpExcuteStageType.OTHER,
                    excute_time=1
                ),
                McpExcuteStage(
                    stage_name="tool_result_processing",
                    stage_type=McpExcuteStageType.DATA_ANALYSIS,
                    excute_time=2
                )
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.FILE_READ,
            **kvargs
        )
        self.tool_name = tool_name
        self.eval_target = eval_target

    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "eval_target": self.eval_target,
            **super().to_dict()
        }


class McpExternalTest(McpFunction):
    def __init__(self, test_target: str, **kvargs):
        super().__init__(
            name="test",
            description=f"[TEST]: test {test_target}",
            excute_time=1,
            excute_stages=[
                McpExcuteStage(
                    stage_name="style_check",
                    stage_type=McpExcuteStageType.TEST_STYLE_CHECK,
                    excute_time=0.5
                ),
                McpExcuteStage(
                    stage_name="compile_check",
                    stage_type=McpExcuteStageType.TEST_COMPILE_CHECK,
                    excute_time=1
                ),
                McpExcuteStage(
                    stage_name="func_check",
                    stage_type=McpExcuteStageType.TEST_FUNC_CHECK,
                    excute_time=2
                ),
                McpExcuteStage(
                    stage_name="test_report",
                    stage_type=McpExcuteStageType.TEST_REPORT,
                    excute_time=1
                ),
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.FILE_READ,
            **kvargs
        )
        self.test_target = test_target

    def to_dict(self):
        return {
            "test_target": self.test_target,
            **super().to_dict()
        }


class McpUserConfirm(McpFunction):
    def __init__(self, confirm_content: str, **kvargs):
        super().__init__(
            name="user_confirm",
            description=f"[USER_CONFIRM]: confirm {confirm_content}",
            excute_time=2,
            excute_stages=[
                McpExcuteStage(
                    stage_name="confirm_send",
                    stage_type=McpExcuteStageType.NET_CONNECT,
                    excute_time=0.5
                ),
                McpExcuteStage(
                    stage_name="confirm_result",
                    stage_type=McpExcuteStageType.NET_DISCONNECT,
                    excute_time=1
                )
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.OTHER,
            **kvargs
        )
        self.confirm_content = confirm_content

    def to_dict(self):
        return {
            "confirm_content": self.confirm_content,
            **super().to_dict()
        }


class McpDataAnalysis(McpFunction):
    def __init__(self, analysis_target: str, **kvargs):
        super().__init__(
            name="analysis",
            description=f"[ANALYSIS]: analysis {analysis_target}",
            excute_time=1,
            excute_stages=[
                McpExcuteStage(
                    stage_name="analysis",
                    stage_type=McpExcuteStageType.DATA_ANALYSIS,
                    excute_time=1
                )
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.FILE_HYBRID,
            **kvargs
        )
        self.analysis_target = analysis_target

    def to_dict(self):
        return {
            "analysis_target": self.analysis_target,
            **super().to_dict()
        }


class McpPlanner(McpFunction):
    def __init__(self, planner_target: str, **kvargs):
        super().__init__(
            name="planner",
            description=f"[PLANNER]: plan {planner_target}",
            excute_time=2,
            excute_stages=[
                McpExcuteStage(
                    stage_name="planner",
                    stage_type=McpExcuteStageType.OTHER,
                    excute_time=1
                )
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.FILE_HYBRID,
        )
        self.planner_target = planner_target

    def to_dict(self):
        return {
            "planner_target": self.planner_target,
            **super().to_dict()
        }


class McpReflection(McpFunction):
    def __init__(self, reflection_target: str, **kvargs):
        super().__init__(
            name="reflection",
            description=f"[REFLECTION]: reflection {reflection_target}",
            excute_time=2,
            excute_stages=[
                McpExcuteStage(
                    stage_name="init_think",
                    stage_type=McpExcuteStageType.OTHER,
                    excute_time=1
                ),
                McpExcuteStage(
                    stage_name="re_think",
                    stage_type=McpExcuteStageType.OTHER,
                    excute_time=1
                ),
                McpExcuteStage(
                    stage_name="get_conlusion",
                    stage_type=McpExcuteStageType.OTHER,
                    excute_time=1
                )
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.FILE_HYBRID,
            **kvargs
        )
        self.reflection_target = reflection_target

    def to_dict(self):
        return {
            "reflection_target": self.reflection_target,
            **super().to_dict()
        }


class McpImgGeneration(McpFunction):
    def __init__(self, img_generation_target: str, img_generation_prompt: str, **kvargs):
        super().__init__(
            name="img_generation",
            description=f"[IMG_GENERATION]: generate img {img_generation_target} with prompt {img_generation_prompt}",
            excute_time=1,
            excute_stages=[
                McpExcuteStage(
                    stage_name="img_gen_initialization",
                    stage_type=McpExcuteStageType.TOOL_INITIALIZATION,
                    excute_time=1
                ),
                McpExcuteStage(
                    stage_name="img_generation",
                    stage_type=McpExcuteStageType.TOOL_EXECUTION,
                    excute_time=15
                ),
                McpExcuteStage(
                    stage_name="img_result_processing",
                    stage_type=McpExcuteStageType.DATA_ANALYSIS,
                    excute_time=1
                )
            ],
            function_value_type=MCPFunctionValueType.NEED_VALUE,
            function_type=MCPFunctionType.FILE_HYBRID,
            **kvargs
        )
        self.img_generation_target = img_generation_target
        self.img_generation_prompt = img_generation_prompt

    def to_dict(self):
        return {
            "img_generation_target": self.img_generation_target,
            "img_generation_prompt": self.img_generation_prompt,
            **super().to_dict()
        }

import random
import json
import string
from typing import Callable, Optional
from uuid import UUID

from agent.graph.meta import (LLMCallMetadata, LLMTextChunk, LLMTextChunkChain,
                              TransferDataItem)
from agent.graph.node import LLMAppNode, NodeStep
from agent.mcp.mcp_func import (McpConditionalExecution, McpDataAnalysis,
                               McpExternalTest, McpExternalToolEval,
                               McpFileWrite, McpFunction, McpSearch,
                               McpToolCall, McpUserConfirm)
from agent.mcp.static import MCPFunctionType, MCPFunctionValueType

INPUT_COMPOSER_TYPE = Callable[[TransferDataItem], LLMTextChunkChain]
OUTPUT_COMPOSER_TYPE = Callable[[LLMTextChunkChain], TransferDataItem]


class McpNode(LLMAppNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        input_composer: Optional[INPUT_COMPOSER_TYPE] = None,
        output_composer: Optional[OUTPUT_COMPOSER_TYPE] = None,
        name: Optional[str] = None,
        uuid: Optional[UUID] = None,
        node_type: Optional[str] = None,
        mcp_input_composer: Optional[INPUT_COMPOSER_TYPE] = None,
        mcp_output_composer: Optional[OUTPUT_COMPOSER_TYPE] = None,
        function_value_type: MCPFunctionValueType = MCPFunctionValueType.NEED_VALUE,
        function_type: MCPFunctionType = MCPFunctionType.OTHER,
        mcp_function: McpFunction = None,
        **kvargs
    ):
        super().__init__(
          metadata=metadata,
          input_composer=input_composer,
          output_composer=output_composer,
          name=name,
          uuid=uuid,
          node_type=node_type,
          **kvargs
        )
        self.mcp_input_composer: INPUT_COMPOSER_TYPE = \
            mcp_input_composer if mcp_input_composer is not None \
            else (lambda dataitem: dataitem[list(dataitem.keys())[0]])
        self.mcp_output_composer: OUTPUT_COMPOSER_TYPE = \
            mcp_output_composer if mcp_output_composer is not None \
            else (lambda x: TransferDataItem(data=x))
        self.function_value_type = function_value_type
        self.function_type = function_type
        self.mcp_function = mcp_function
        self.type_step = NodeStep.get_mcp_step(self.function_value_type)

    def mcp_input(self, input_chunks: TransferDataItem) -> LLMTextChunkChain:
        assert self.mcp_input_composer is not None, "MCP input composer is not set."

        def _to_chain(item: TransferDataItem) -> LLMTextChunkChain:
            if isinstance(item.data, LLMTextChunkChain):
                return item.data

            if isinstance(item.data, dict):
                merged_chunks = []
                for chain in item.data.values():
                    if isinstance(chain, LLMTextChunkChain):
                        merged_chunks.extend(chain.chunks)
                    else:
                        raise TypeError("Dict value is not LLMTextChunkChain")
                return LLMTextChunkChain(chunks=merged_chunks)

            raise TypeError("Unsupported TransferDataItem structure")

        if isinstance(input_chunks, LLMTextChunkChain):
            return self.mcp_input_composer(input_chunks)
        elif isinstance(input_chunks, TransferDataItem):
            chain = _to_chain(input_chunks)
            return self.mcp_input_composer(chain)
        else:
            raise TypeError("Unsupported input type")

    def mcp_output(self, output_chunks: LLMTextChunkChain) -> TransferDataItem:
        assert self.mcp_output_composer is not None, "MCP output composer is not set."
        return self.mcp_output_composer(output_chunks)

    def _get_mcp_function_info(self):
        return self.mcp_function.to_dict()


CONDITIONAL_EXECUTION_PROMPT = LLMTextChunk(
    text="You are a helpful assistant. You will be given a condition and a task. You need to check if the condition is met. If it is, you need to execute the task. If it is not, you need to return a message saying that the condition is not met.",
    metadata=LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=1000, temperature=0)
)

class McpConditionalExecutionNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        condition_description: str,
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpConditionalExecution(
            condition_description=condition_description
        )

        super().__init__(
            metadata=metadata,
            name=name or f"mcp_conditional_{condition_description.replace(' ', '_')}",
            node_type="mcp_conditional_execution",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = CONDITIONAL_EXECUTION_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text=f"\n[CONDITIONAL_EXECUTION] Checking condition: {condition_description}\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:

            result = {
                "type": "conditional_execution",
                "condition": self.mcp_function.condition_description,
                "result": f"Condition '{self.mcp_function.condition_description}' evaluated.",
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[CONDITION_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


FILE_WRITE_PROMPT = LLMTextChunk(
    text="You are a helpful assistant. You will be given a file path and a file content. You need to write the content to the file.",
    metadata=LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=1000, temperature=0)
)

class McpFileWriteNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        file_path: str,
        file_content: str,
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpFileWrite(
            file_path=file_path,
            file_content=file_content
        )

        super().__init__(
            metadata=metadata,
            name=name or f"mcp_file_write_{file_path}",
            node_type="mcp_file_write",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = FILE_WRITE_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text=f"\n[FILE_WRITE] Writing {file_content} to {file_path}\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "file_write",
                "file_path": self.mcp_function.file_path,
                "file_content": self.mcp_function.file_content,
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[FILE_WRITE_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


SEARCH_PROMPT = LLMTextChunk(
    text="You are a helpful assistant. You will be given a search query and a search type. You need to search the web for the query and return the results.",
    metadata=LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=1000, temperature=0)
)

class McpSearchNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        search_query: str,
        search_type: str = "general",
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpSearch(
            search_query=search_query,
            search_type=search_type
        )

        super().__init__(
            metadata=metadata,
            name=name or f"mcp_search_{search_type}",
            node_type="mcp_search",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = SEARCH_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text=f"\n[SEARCH] Performing {search_type} search for: '{search_query}'\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "search",
                "query": self.mcp_function.search_query,
                "search_type": self.mcp_function.search_type,
                "results": [
                    {"title": "Result 1", "snippet": str("".join(random.choices(string.ascii_letters + string.digits, k=100)))},
                    {"title": "Result 2", "snippet": str("".join(random.choices(string.ascii_letters + string.digits, k=100)))}
                ],
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[SEARCH_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


TOOL_CALL_PROMPT = LLMTextChunk(
    text="You are a helpful assistant. You will be given a tool name and a tool args. You need to call the tool and return the result.",
    metadata=LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=1000, temperature=0)
)

class McpToolCallNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        tool_name: str,
        tool_args: dict = None,
        name: Optional[str] = None,
        **kvargs
    ):
        if tool_args is None:
            tool_args = {}

        self.mcp_function = McpToolCall(
            tool_name=tool_name,
            tool_args=tool_args
        )

        super().__init__(
            metadata=metadata,
            name=name or f"mcp_tool_{tool_name}",
            node_type="mcp_tool_call",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = TOOL_CALL_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text=f"\n[TOOL_CALL] Calling tool '{tool_name}' with args: {tool_args}\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "tool_call",
                "tool": self.mcp_function.tool_name,
                "args": self.mcp_function.tool_args,
                "result": "Simulated tool output",
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[TOOL_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


EXTERNAL_TOOL_EVAL_PROMPT = LLMTextChunk(
    text="You are a helpful assistant. You will be given a tool name and a eval target. You need to evaluate the target with the tool and return the result.",
    metadata=LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=1000, temperature=0)
)

class McpExternalToolEvalNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        tool_name: str,
        eval_target: str,
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpExternalToolEval(tool_name=tool_name, eval_target=eval_target)

        super().__init__(
            metadata=metadata,
            name=name or f"mcp_external_tool_eval_{tool_name}",
            node_type="mcp_external_tool_eval",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = EXTERNAL_TOOL_EVAL_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text=f"\n[EXTERNAL_TOOL_EVAL] Evaluating {eval_target} with {tool_name}\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "external_tool_eval",
                "tool_name": self.mcp_function.tool_name,
                "eval_target": self.mcp_function.eval_target,
                "eval_result": "Simulated evaluation result: pass",
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[EXTERNAL_TOOL_EVAL_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


EXTERNAL_TEST_PROMPT = LLMTextChunk(
    text="You are a helpful assistant. You will be given a test target. You need to test the target and return the result.",
    metadata=LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=1000, temperature=0)
)

class McpExternalTestNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        test_target: str,
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpExternalTest(test_target=test_target)

        super().__init__(
            metadata=metadata,
            name=name or f"mcp_test_{test_target}",
            node_type="mcp_external_test",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = EXTERNAL_TEST_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text=f"\n[TEST] Testing target: {test_target}\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "test",
                "test_target": self.mcp_function.test_target,
                "test_result": "Simulated test passed",
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[TEST_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


USER_CONFIRM_PROMPT = LLMTextChunk(
    text="You are a helpful assistant. You will be given a confirm content. You need to confirm the content and return the result.",
    metadata=LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=1000, temperature=0)
)

class McpUserConfirmNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        confirm_content: str,
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpUserConfirm(confirm_content=confirm_content)

        super().__init__(
            metadata=metadata,
            name=name or "mcp_user_confirm",
            node_type="mcp_user_confirm",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = USER_CONFIRM_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text=f"\n[USER_CONFIRM] Please confirm: {confirm_content}\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "user_confirm",
                "confirm_content": self.mcp_function.confirm_content,
                "confirm_result": "Simulated user confirmed",
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[USER_CONFIRM_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


DATA_ANALYSIS_PROMPT = LLMTextChunk(
    text="You are a helpful assistant. You will be given a data analysis target. You need to analyze the target and return the result.",
    metadata=LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=1000, temperature=0)
)

class McpDataAnalysisNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        analysis_target: str,
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpDataAnalysis(analysis_target=analysis_target)

        super().__init__(
            metadata=metadata,
            name=name or f"mcp_analysis_{analysis_target}",
            node_type="mcp_data_analysis",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = DATA_ANALYSIS_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text=f"\n[ANALYSIS] Analyzing {analysis_target}\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "analysis",
                "analysis_target": self.mcp_function.analysis_target,
                "analysis_result": "Simulated analysis result",
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[ANALYSIS_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output

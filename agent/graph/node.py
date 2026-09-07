import uuid as uuid_lib
import enum
from typing import Callable, Optional
from uuid import UUID

from agent.graph.meta import (LLMCallMetadata, LLMTextChunk, LLMTextChunkChain,
                              TransferDataItem)
from agent.mcp.static import MCPFunctionValueType
from agent.vllm_prompt import LLM_SYSTEM_PROMPT


class NodeStep(enum.IntEnum):
    TEXT = 1
    VOID = 1
    LLM = 3
    MCP_NOT_NEED_VALUE = 3
    MCP_NEED_VALUE = 4
    MCP_SON_NEED_VALUE = 4

    def get_mcp_step(mcp_type: MCPFunctionValueType) -> int:
        if mcp_type == MCPFunctionValueType.NOT_NEED_VALUE:
            return NodeStep.MCP_NOT_NEED_VALUE
        elif mcp_type == MCPFunctionValueType.NEED_VALUE:
            return NodeStep.MCP_NEED_VALUE
        elif mcp_type == MCPFunctionValueType.SON_NEED_VALUE:
            return NodeStep.MCP_SON_NEED_VALUE
        else:
            raise ValueError(f"Invalid MCP function value type: {mcp_type}")


class Node:
    def __init__(
        self,
        type_step: NodeStep = NodeStep.TEXT,
        uuid: Optional[UUID] = None,
        name: Optional[str] = None,
        node_type: Optional[str] = None,
        **kvargs
    ):
        self.uuid = uuid_lib.uuid4() if uuid is None else uuid
        self.type_step = type_step
        self.name = name
        self.node_type = node_type
        self.in_degree = 0
        self.out_degree = 0
        self.kvargs = kvargs

    def add_kvargs(self, **kvargs) -> None:
        self.kvargs.update(kvargs)


class TextNode(Node):
    def __init__(
        self,
        composer: Optional[Callable[[TransferDataItem], TransferDataItem]] = None,
        name: Optional[str] = None,
        uuid: Optional[UUID] = None,
        **kvargs
    ):
        self.composer = composer
        super().__init__(type_step=NodeStep.TEXT, uuid=uuid, name=name, node_type="text", **kvargs)

    def assign_composer(self, composer: Callable[[TransferDataItem], TransferDataItem]) -> None:
        self.composer = composer

    def __call__(self, input_chunks: TransferDataItem) -> TransferDataItem:
        return self.composer(input_chunks)


SYSTEM_PROMPT = LLMTextChunk(
    text=LLM_SYSTEM_PROMPT,
    metadata=LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=1000, temperature=0)
)

class LLMAppNode(Node):

    def __init__(
        self,
        metadata: LLMCallMetadata,
        input_composer: Optional[Callable[[TransferDataItem], LLMTextChunkChain]] = None,
        output_composer: Optional[Callable[[LLMTextChunkChain], TransferDataItem]] = None,
        name: Optional[str] = None,
        uuid: Optional[UUID] = None,
        node_type: Optional[str] = None,
        **kvargs
    ):
        if not hasattr(self, "system_prompt"):
            self.system_prompt = SYSTEM_PROMPT
        self.metadata = metadata
        super().__init__(type_step=NodeStep.LLM, name=name, uuid=uuid, node_type=node_type, **kvargs)

        def default_input_composer(task: TransferDataItem) -> LLMTextChunkChain:
            chunks = []
            chunks.append(self.system_prompt)
            if isinstance(task.data, LLMTextChunkChain):
                chunks.extend(task.data.chunks)
            elif isinstance(task.data, dict):
                for chain in task.data.values():
                    chunks.extend(chain.chunks)
            else:
                raise TypeError("Unsupported TransferDataItem structure in default_input_composer")
            return LLMTextChunkChain(chunks=chunks)

        def default_output_composer(output_chain: LLMTextChunkChain) -> TransferDataItem:
            return TransferDataItem(data=LLMTextChunkChain(chunks=output_chain.chunks))

        self.input_composer = input_composer if input_composer is not None else default_input_composer
        self.output_composer = output_composer if output_composer is not None else default_output_composer

    def assign_composer(
        self,
        is_input: bool,
        composer: Callable[[TransferDataItem], LLMTextChunkChain]
    ) -> None:
        if is_input:
            self.input_composer = composer
        else:
            self.output_composer = composer

    def preprocess(self, input_chunks: TransferDataItem) -> LLMTextChunkChain:
        assert self.input_composer is not None, "Input composer is not set."
        return self.input_composer(input_chunks)

    def postprocess(self, output_chunks: LLMTextChunkChain) -> TransferDataItem:
        assert self.output_composer is not None, "Output composer is not set."
        return self.output_composer(output_chunks)


class VoidNode(Node):
    def __init__(self, name: Optional[str] = None, uuid: Optional[UUID] = None, **kvargs):
        super().__init__(type_step=NodeStep.VOID, name=name, uuid=uuid, node_type="void", **kvargs)

    def __call__(self, input_chunks: TransferDataItem) -> TransferDataItem:
        return input_chunks

import uuid as uuid_lib
from typing import Dict, Optional, Union, List
from uuid import UUID

from agent.graph.meta import LLMTextChunk, LLMTextChunkChain, TransferDataItem
from agent.graph.graph import LLMAppGraph
from agent.graph.node import LLMAppNode


class LLMApplication:

    def __init__(self, graph: LLMAppGraph, uuid: Optional[UUID] = None):
        self.graph = graph
        self.uuid = uuid_lib.uuid4() if uuid is None else uuid_lib.UUID(uuid)


class ApplicationRequest:

    def __init__(
        self,
        prompt: str,
        application: LLMApplication,
        uuid: Optional[UUID] = None
    ):
        self.prompt: str = prompt
        self.application: LLMApplication = application
        self.uuid: UUID = uuid_lib.uuid4() if uuid is None else uuid_lib.UUID(uuid)
        self.finished_nodes: List[UUID] = [application.graph.entry_node_uuid]
        self._waiting_or_running: List[UUID] = application.graph.successors(application.graph.entry_node_uuid)
        self.running_node_inputs: Dict[UUID, Union[LLMTextChunkChain, TransferDataItem]] = {}

        self.queued_outputs = {
            node_uuid: TransferDataItem(
                data={
                    self.application.graph.entry_node_uuid: LLMTextChunkChain.from_single_text(prompt)
                }
            ) for node_uuid in self._waiting_or_running
        }
        self.middle_outputs: List[str] = []
        self.output: str = ""

    @property
    def waiting_or_running(self):
        return self._waiting_or_running

    def get_node(self, node_uuid: UUID) -> LLMAppNode:
        return self.application.graph.get_node(node_uuid)

    def get_node_input(self, node_uuid: UUID) -> LLMTextChunkChain:
        assert (
            node_uuid in self.application.graph.nodes
        ), "The node must exist in the application graph."
        assert (
            node_uuid in self.queued_outputs
        ), "The node must be in the queued outputs."
        assert all(
            pred in self.finished_nodes
            for pred in self.application.graph.predecessors(node_uuid)
        ), "All predecessors must be finished."

        if self.get_node(node_uuid).type_step == 1:
            return self.queued_outputs[node_uuid]
        else:
            return self.get_node(node_uuid).preprocess(self.queued_outputs[node_uuid])

    def submit_input(
        self,
        node_uuid: UUID,
        input_chunks: Union[LLMTextChunkChain, TransferDataItem]
    ) -> None:
        self.running_node_inputs[node_uuid] = input_chunks

    def submit_output(
        self,
        finished_node_uuid: UUID,
        output_text: Union[str, TransferDataItem]
    ) -> None:
        assert (
            finished_node_uuid in self._waiting_or_running
        ), f"The finished node {self.get_node(finished_node_uuid).name} must be in the waiting nodes."

        self._waiting_or_running.remove(finished_node_uuid)
        self.finished_nodes.append(finished_node_uuid)
        finished_node_input: LLMTextChunkChain = self.running_node_inputs[finished_node_uuid]
        del self.running_node_inputs[finished_node_uuid]

        def to_transfer_data_item(val: Union[str, TransferDataItem]) -> TransferDataItem:
            return val if isinstance(val, TransferDataItem) else \
                TransferDataItem(data=LLMTextChunkChain.from_single_text(val))

        if finished_node_uuid == self.application.graph.exit_node_uuid:
            output_data_item = to_transfer_data_item(output_text)
            self.output = output_data_item.to_text()
            self.middle_outputs.append(output_data_item.to_text())
            return

        if self.get_node(finished_node_uuid).type_step >= 3:
            concat_chunks = LLMTextChunkChain(
                chunks=finished_node_input.chunks + [LLMTextChunk.from_text(output_text)]
            )
            output_data_item = self.get_node(finished_node_uuid).postprocess(concat_chunks)
        else:
            output_data_item = to_transfer_data_item(output_text)

        self.middle_outputs.append(output_data_item.to_text())

        for successor in self.application.graph.successors(finished_node_uuid):
            if successor not in self.queued_outputs:
                self.queued_outputs[successor] = TransferDataItem(data={})

            assert (
                finished_node_uuid not in self.queued_outputs[successor].data
            ), "The finished node should not be in the queued outputs."

            self.queued_outputs[successor][finished_node_uuid] = output_data_item[successor]

            if all(
                predecessor in self.finished_nodes
                for predecessor in self.application.graph.predecessors(successor)
            ):
                self._waiting_or_running.append(successor)

    def is_finished(self):
        return self.application.graph.exit_node_uuid in self.finished_nodes

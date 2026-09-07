from typing import List, Optional
from uuid import UUID
import hashlib
import json

from agent.graph.meta import (LLMCallMetadata, LLMTextChunk, LLMTextChunkChain,
                              TransferDataItem)
from agent.graph.graph import InDegreePriorityGraph
from agent.graph.node import LLMAppNode, TextNode
from agent.app.request import LLMApplication
from agent.app.code_writer_mcp import CodeContextBuilder
from agent.mcp.mcp_node import (McpConditionalExecutionNode, McpDataAnalysisNode,
                               McpNode, McpSearchNode)
from agent.mcp.mcp_func import (McpFileWrite, McpImgGeneration, McpPlanner,
                               McpReflection)

"""
Start -> Planner -> 3 rounds of 3 parallel Search -> Analysis -> Summary
      -> Reflection -> Judge -> TextGenerator -> ImgGenerator -> Answerer -> End
"""

RESEARCH_PRESSURE_PROFILE = "research-paper-pressure"


class ResearchPressureContextBuilder(CodeContextBuilder):

    task_header = "[RESEARCH TASK]\n"
    context_files = (
        "vllm/mcp/mcp_manager.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/core/cpu_offloading_block_pool.py",
        "vllm/mcp/agent_offload_coordinator.py",
        "vllm/v1/core/cpu_offloading_kv_cache_manager.py",
    )


def _assign_research_input_composer(node, predecessors):

    def composer(task: TransferDataItem) -> LLMTextChunkChain:
        if isinstance(task.data, LLMTextChunkChain):
            chains = [task.data]
        else:
            chains = [task.data[parent] for parent in predecessors]
        common_length = min((len(chain.chunks) for chain in chains), default=0)
        for index in range(common_length):
            if any(chain.chunks[index].text != chains[0].chunks[index].text
                   for chain in chains[1:]):
                common_length = index
                break
        chunks = list(chains[0].chunks[:common_length]) if chains else []
        for chain in chains:
            chunks.extend(chain.chunks[common_length:])
        chunks.append(LLMTextChunk(
            text=f"\n[RESEARCH STAGE: {node.name}]\n{node.system_prompt.text}\n",
            metadata=node.metadata,
        ))
        return LLMTextChunkChain(chunks=chunks)

    node.assign_composer(is_input=True, composer=composer)


def _assign_research_search_output(node, source_text):
    _, marker, dossier = source_text.partition("\n[REPOSITORY SNAPSHOT: ")
    evidence = dossier if marker else source_text
    paragraphs = [text for text in evidence.split("\n\n") if len(text) >= 80]
    if not paragraphs:
        paragraphs = [evidence]
    digest = hashlib.sha256(f"{node.name}\n{source_text}".encode()).digest()
    results = [
        {"title": f"Research context excerpt {index + 1}",
         "snippet": paragraphs[int.from_bytes(digest[index * 4:(index + 1) * 4],
                                              "big") % len(paragraphs)][:400]}
        for index in range(2)
    ]

    def composer(output_chain: LLMTextChunkChain) -> TransferDataItem:
        result = {
            "type": "search",
            "query": node.mcp_function.search_query,
            "search_type": node.mcp_function.search_type,
            "results": results,
            "function_info": node._get_mcp_function_info(),
            "status": "success",
        }
        return TransferDataItem(data=LLMTextChunkChain(chunks=[LLMTextChunk(
            text=f"[SEARCH_RESULT] {json.dumps(result, ensure_ascii=False)}\n",
            metadata=node.metadata,
        )]))

    node.mcp_output_composer = composer


planner_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=512, temperature=0)
search_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=256, temperature=0)
summary_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=256, temperature=0)
reflect_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=256, temperature=0)
generator_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=256, temperature=0)

SUMMARY_PROMPT = LLMTextChunk(
    text="You are a research summariser. Provide a concise Chinese summary of the search results.",
    metadata=summary_meta
)
TEXT_GEN_PROMPT = LLMTextChunk(
    text="Based on the collected context, draft a comprehensive answer in Chinese.",
    metadata=generator_meta
)
PLANNER_PROMPT = LLMTextChunk(
    text="You are a research planner. Analyse the Chinese user request and output key queries.",
    metadata=planner_meta
)
REFLECTION_PROMPT = LLMTextChunk(
    text="You are a senior researcher. Consolidate the following summaries and assess context sufficiency.",
    metadata=reflect_meta
)
IMG_GEN_PROMPT = LLMTextChunk(
    text="You are a senior researcher. Generate a comprehensive answer in Chinese.",
    metadata=generator_meta
)


class RoleNode(LLMAppNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        name: Optional[str] = None,
        node_type: Optional[str] = None
    ):
        super().__init__(metadata=metadata, name=name, node_type=node_type)


class StartNode(TextNode):
    def __init__(self, target_uuids: List[UUID], metadata: LLMCallMetadata, prompt: Optional[str] = None):
        super().__init__(name="start_deep_research")
        self.target_uuids = target_uuids
        self.metadata = metadata
        self.prompt = prompt
        self.setup_composer()

    def setup_composer(self):
        def composer(task: TransferDataItem) -> TransferDataItem:
            if self.prompt:
                prompt_chunk = LLMTextChunk(text=self.prompt, metadata=self.metadata)
                return TransferDataItem(data={
                    target_uuid: LLMTextChunkChain(chunks=[prompt_chunk])
                    for target_uuid in self.target_uuids
                })
            else:
                return TransferDataItem(data={
                    target_uuid: LLMTextChunkChain(chunks=[task.data[list(task.data.keys())[0]].chunks[0]])
                    for target_uuid in self.target_uuids
                })

        self.assign_composer(composer)


class McpSummaryNode(RoleNode):
    def __init__(self, metadata: LLMCallMetadata, name: Optional[str] = None):
        self.system_prompt = SUMMARY_PROMPT
        super().__init__(metadata=metadata, name=name, node_type="summary")


class McpPlannerNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpPlanner(planner_target="query_planning")

        super().__init__(
            metadata=metadata,
            name=name or "mcp_planner",
            node_type="mcp_planner",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = PLANNER_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text="\n[PLANNER] Planning search queries for user request\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "planning",
                "planner_target": self.mcp_function.planner_target,
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[PLANNING_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


class McpReflectionNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        input_node_uuids: List[UUID],
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpReflection(reflection_target="reflection_consolidation")
        self.input_node_uuids = input_node_uuids

        super().__init__(
            metadata=metadata,
            name=name or "mcp_reflection",
            node_type="mcp_reflection",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = REFLECTION_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text="\n[REFLECTION] Consolidating all summaries and assessing\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "reflection",
                "reflection_target": self.mcp_function.reflection_target,
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[REFLECTION_RESULT]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


class McpTextGeneratorNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        name: Optional[str] = None,
        **kvargs
    ):
        self.mcp_function = McpFileWrite(
            file_path="text_answer.md",
            file_content="Generated text content"
        )

        super().__init__(
            metadata=metadata,
            name=name or "mcp_text_generator",
            node_type="mcp_text_generator",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = TEXT_GEN_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text="\n[TEXT_GENERATION] Generating comprehensive text answer\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "text_generation",
                "file_path": self.mcp_function.file_path,
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[TEXT_GENERATION]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


class McpImgGeneratorNode(McpNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        prompt: str = "Auto illustration",
        **kvargs
    ):
        self.mcp_function = McpImgGeneration(
            img_generation_target="img_generation",
            img_generation_prompt=prompt
        )

        super().__init__(
            metadata=metadata,
            name="img_generator",
            node_type="img_generator",
            function_value_type=self.mcp_function.function_value_type,
            function_type=self.mcp_function.function_type,
            mcp_function=self.mcp_function,
            **kvargs
        )
        self.system_prompt = IMG_GEN_PROMPT

        def mcp_input(input_chain: LLMTextChunkChain) -> LLMTextChunkChain:
            prompt = LLMTextChunk(
                text="\n[IMG_GENERATION] Generating comprehensive image answer\n",
                metadata=metadata
            )
            return LLMTextChunkChain(chunks=[prompt] + input_chain.chunks)

        def mcp_output(output_chain: LLMTextChunkChain) -> TransferDataItem:
            result = {
                "type": "img_generation",
                "img_generation_target": self.mcp_function.img_generation_target,
                "img_generation_prompt": self.mcp_function.img_generation_prompt,
                "function_info": self._get_mcp_function_info(),
                "status": "success"
            }
            tag = "[IMG_GENERATION]"
            result_chunk = LLMTextChunk(
                text=f"{tag} {json.dumps(result, ensure_ascii=False)}\n",
                metadata=metadata
            )
            return TransferDataItem(data=LLMTextChunkChain(chunks=[result_chunk]))

        self.mcp_input_composer = mcp_input
        self.mcp_output_composer = mcp_output


class AnswererNode(TextNode):
    def __init__(self):
        super().__init__(name="answerer")
        self.setup_composer()

    def setup_composer(self):
        def composer(task: TransferDataItem) -> TransferDataItem:
            chunks: List[LLMTextChunk] = []
            for chain in task.data.values():
                chunks.extend(chain.chunks)
            return TransferDataItem(data=LLMTextChunkChain(chunks=chunks))

        self.assign_composer(composer)


class EndNode(TextNode):
    def __init__(self):
        super().__init__(name="end_deep_research")
        self.setup_composer()

    def setup_composer(self):
        def composer(task: TransferDataItem) -> TransferDataItem:
            final_chunks = [
                LLMTextChunk(text="\n=== DEEP-RESEARCH ANSWER ===\n", metadata=generator_meta)
            ]
            for chain in task.data.values():
                final_chunks.extend(chain.chunks)
            return TransferDataItem(data=LLMTextChunkChain(chunks=final_chunks))

        self.assign_composer(composer)


class DeepResearchMcpApplication(LLMApplication):
    NUM_ROUNDS = 3
    SEARCHES_PER_ROUND = 3
    profile_name = RESEARCH_PRESSURE_PROFILE

    def __init__(self, llm_metadata: LLMCallMetadata, prompt: str = None, *,
                 context_token_count: Optional[int] = None,
                 context_sources: tuple[str, ...] = ()):
        self.context_token_count = context_token_count
        self.context_sources = context_sources
        graph = InDegreePriorityGraph()

        start_node = StartNode([], llm_metadata, prompt)
        graph.add_node(start_node)
        graph.add_edge(graph.entry_node_uuid, start_node.uuid)

        planner = McpPlannerNode(metadata=planner_meta, name="mcp_planner")
        graph.add_node(planner)
        graph.add_edge(start_node.uuid, planner.uuid)
        start_node.target_uuids = [planner.uuid]

        all_summary_uuids: List[UUID] = []

        prev_node_list = [0 for _ in range(self.SEARCHES_PER_ROUND)]
        for rnd in range(self.NUM_ROUNDS):
            all_summary_uuids = []
            for idx in range(self.SEARCHES_PER_ROUND):
                search_node = McpSearchNode(
                    metadata=search_meta,
                    search_query=f"round{rnd + 1}_query_{idx + 1}",
                    search_type="general",
                    name=f"searcher_{rnd + 1}_{idx + 1}"
                )
                graph.add_node(search_node)
                prev_node = planner if rnd == 0 else prev_node_list[idx]
                graph.add_edge(prev_node.uuid, search_node.uuid)

                analysis_node = McpDataAnalysisNode(
                    metadata=search_meta,
                    analysis_target=f"search_results_{rnd + 1}_{idx + 1}",
                    name=f"analysis_{rnd + 1}_{idx + 1}"
                )
                graph.add_node(analysis_node)
                graph.add_edge(search_node.uuid, analysis_node.uuid)

                summary_node = McpSummaryNode(
                    metadata=summary_meta,
                    name=f"summary_{rnd + 1}_{idx + 1}"
                )
                graph.add_node(summary_node)
                graph.add_edge(analysis_node.uuid, summary_node.uuid)

                all_summary_uuids.append(summary_node.uuid)

                prev_node_list[idx] = summary_node

        reflection = McpReflectionNode(
            metadata=reflect_meta,
            input_node_uuids=all_summary_uuids,
            name="reflection"
        )
        graph.add_node(reflection)
        for s_uuid in all_summary_uuids:
            graph.add_edge(s_uuid, reflection.uuid)

        judge = McpConditionalExecutionNode(
            metadata=reflect_meta,
            condition_description="context sufficient",
            name="judge_context"
        )
        graph.add_node(judge)
        graph.add_edge(reflection.uuid, judge.uuid)

        text_gen = McpTextGeneratorNode(metadata=generator_meta, name="text_generator")
        img_gen = McpImgGeneratorNode(metadata=generator_meta, prompt="Illustration for the final answer")
        graph.add_node(text_gen)
        graph.add_node(img_gen)
        graph.add_edge(judge.uuid, text_gen.uuid)
        graph.add_edge(text_gen.uuid, img_gen.uuid)

        answerer = AnswererNode()
        graph.add_node(answerer)
        graph.add_edge(text_gen.uuid, answerer.uuid)
        graph.add_edge(img_gen.uuid, answerer.uuid)

        end_node = EndNode()
        graph.add_node(end_node)
        graph.add_edge(answerer.uuid, end_node.uuid)
        graph.add_edge(end_node.uuid, graph.exit_node_uuid)

        for node in graph.nodes.values():
            if isinstance(node, LLMAppNode):
                _assign_research_input_composer(node, graph.predecessors(node.uuid))
                node.add_kvargs(workload_profile=RESEARCH_PRESSURE_PROFILE)
                if isinstance(node, McpNode):
                    node.add_kvargs(
                        preserve_llm_output_after_tool=True,
                        reusable_prefix=True,
                    )
                if isinstance(node, McpSearchNode):
                    _assign_research_search_output(node, prompt or "")

        graph.set_priorities()
        super().__init__(graph)


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional
from uuid import UUID

from agent.app.request import LLMApplication
from agent.graph.graph import InDegreePriorityGraph
from agent.graph.meta import (LLMCallMetadata, LLMTextChunk, LLMTextChunkChain,
                              TransferDataItem)
from agent.graph.node import LLMAppNode, TextNode
from agent.mcp.mcp_node import (
    McpConditionalExecutionNode,
    McpExternalTestNode,
    McpExternalToolEvalNode,
    McpFileWriteNode,
    McpSearchNode,
    McpToolCallNode,
    McpUserConfirmNode,
)

architect_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=200, temperature=0)
programmer_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=400, temperature=0)
reviewer_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=200, temperature=0)
revisor_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=400, temperature=0)
mcp_meta = LLMCallMetadata(model="gpt-4o-mini", max_new_tokens=500, temperature=0)

ARCHITECT_PROMPT = LLMTextChunk(
    text="You are a system architect. Design APIs for the assigned component based on the task.",
    metadata=architect_meta
)

PROGRAMMER_PROMPT = LLMTextChunk(
    text="You are a programmer. Implement the APIs designed by the architect.",
    metadata=programmer_meta
)

REVIEWER_PROMPT = LLMTextChunk(
    text="You are a code reviewer. Review all implementations and provide feedback.",
    metadata=reviewer_meta
)

REVISER_PROMPT = LLMTextChunk(
    text="You are a code reviser. Improve the code based on reviewer feedback.",
    metadata=revisor_meta
)


class RoleNode(LLMAppNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        name: Optional[str] = None,
        node_type: Optional[str] = None
    ):
        super().__init__(metadata=metadata, name=name, node_type=node_type)


class McpArchitectNode(RoleNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        architect_type: str,
        name: Optional[str] = None
    ):
        self.system_prompt = ARCHITECT_PROMPT
        super().__init__(metadata=metadata, name=name, node_type="architect")
        self.architect_type = architect_type


class McpProgrammerNode(RoleNode):
    def __init__(self, metadata: LLMCallMetadata, name: Optional[str] = None):
        self.system_prompt = PROGRAMMER_PROMPT
        super().__init__(metadata=metadata, name=name, node_type="programmer")


class McpReviewerNode(RoleNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        input_node_uuids: List[UUID],
        name: Optional[str] = None
    ):
        self.system_prompt = REVIEWER_PROMPT
        super().__init__(metadata=metadata, name=name, node_type="reviewer")
        self.input_node_uuids = input_node_uuids

        def input_composer(task: TransferDataItem) -> LLMTextChunkChain:
            chunks = []
            for node_uuid in self.input_node_uuids:
                if node_uuid in task.data:
                    chunks.extend(task.data[node_uuid].chunks)
            chunks.append(REVIEWER_PROMPT)
            return LLMTextChunkChain(chunks=chunks)

        def output_composer(output_chain: LLMTextChunkChain) -> TransferDataItem:
            return TransferDataItem(data=LLMTextChunkChain(chunks=output_chain.chunks))

        self.assign_composer(is_input=True, composer=input_composer)
        self.assign_composer(is_input=False, composer=output_composer)


class McpReviserNode(RoleNode):
    def __init__(
        self,
        metadata: LLMCallMetadata,
        reviser_type: str,
        name: Optional[str] = None
    ):
        self.system_prompt = REVISER_PROMPT
        super().__init__(metadata=metadata, name=name, node_type="reviser")
        self.reviser_type = reviser_type


class StartMcpNode(TextNode):
    def __init__(self, architect_uuids: List[UUID], metadata: LLMCallMetadata, prompt: Optional[str] = None):
        super().__init__(name="start_mcp")
        self.architect_uuids = architect_uuids
        self.metadata = metadata
        self.prompt = prompt
        self.setup_composer()

    def setup_composer(self):
        def composer(task: TransferDataItem) -> TransferDataItem:
            if self.prompt:
                prompt_chunk = LLMTextChunk(text=self.prompt, metadata=self.metadata)
                return TransferDataItem(data={
                    architect_uuid: LLMTextChunkChain(chunks=[prompt_chunk])
                    for architect_uuid in self.architect_uuids
                })
            else:
                return TransferDataItem(data={
                    architect_uuid: LLMTextChunkChain(chunks=[task.data[list(task.data.keys())[0]].chunks[0]])
                    for architect_uuid in self.architect_uuids
                })

        self.assign_composer(composer=composer)


class EndMcpNode(TextNode):

    def __init__(self, metadata: LLMCallMetadata):
        super().__init__(name="end_mcp")
        self.metadata = metadata
        self.setup_composer()

    def setup_composer(self):
        def composer(task: TransferDataItem) -> TransferDataItem:
            final_chunks = [LLMTextChunk(
                    text="\n=== FINAL MCP TEST RESULTS ===\n", metadata=self.metadata
            )]
            for node_uuid in task.data.keys():
                final_chunks.extend(task.data[node_uuid].chunks)
            return TransferDataItem(data=LLMTextChunkChain(chunks=final_chunks))

        self.assign_composer(composer=composer)


CODE_PROFILE = "code"
CODE_CONTEXT_MIN_TOKENS = 2_048
CODE_CONTEXT_TARGET_TOKENS = 4_608
CODE_CONTEXT_MAX_TOKENS = 5_120

CODE_CONTEXT_FILES = (
    "pyproject.toml",
    "vllm_serving.py",
    "agent/app/code_writer_mcp.py",
    "vllm/mcp/agent_offload_coordinator.py",
    "vllm/v1/core/cpu_offloading_kv_cache_manager.py",
)


@dataclass(frozen=True)
class CodeContext:
    text: str
    token_count: int
    sources: tuple[str, ...]


class CodeContextBuilder:

    context_files = CODE_CONTEXT_FILES
    task_header = "[AGENTCODE TASK]\n"

    def __init__(
        self,
        tokenizer: Any,
        repo_root: Path,
        *,
        target_tokens: int = CODE_CONTEXT_TARGET_TOKENS,
    ) -> None:
        if not CODE_CONTEXT_MIN_TOKENS <= target_tokens <= CODE_CONTEXT_MAX_TOKENS:
            raise ValueError(
                "paper-pressure target must be between 2,048 and 5,120 tokens")
        self.tokenizer = tokenizer
        self.repo_root = repo_root.resolve()
        self.target_tokens = target_tokens
        self._source_sections = self._load_source_sections()

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def _load_source_sections(self) -> list[tuple[str, list[int]]]:
        sections: list[tuple[str, list[int]]] = []
        snapshot = json.loads(
            (self.repo_root / "dataset/context_snapshot.json").read_text(encoding="utf-8"))
        for relative_path in self.context_files:
            entry = snapshot["files"][relative_path]
            text = entry["text"]
            sections.append((relative_path, self._encode(text)))
        return sections

    def build(self, task_prompt: str) -> CodeContext:
        prompt_header = self.task_header
        source_header = "\n[REPOSITORY SNAPSHOT: {path}]\n"
        task_tokens = self._encode(prompt_header + task_prompt.strip() + "\n")
        if len(task_tokens) >= self.target_tokens:
            raise ValueError(
                "Task prompt leaves no room for repository context")

        source_budget = self.target_tokens - len(task_tokens)
        per_source_budget = max(1, source_budget // len(self._source_sections))
        combined = list(task_tokens)
        consumed: dict[str, int] = {}

        for relative_path, source_tokens in self._source_sections:
            header_tokens = self._encode(
                source_header.format(path=relative_path))
            content_budget = max(1, per_source_budget - len(header_tokens))
            combined.extend(header_tokens)
            combined.extend(source_tokens[:content_budget])
            consumed[relative_path] = min(content_budget, len(source_tokens))

        while len(combined) < self.target_tokens:
            made_progress = False
            for relative_path, source_tokens in self._source_sections:
                offset = consumed[relative_path]
                if offset >= len(source_tokens):
                    continue
                take = min(128, self.target_tokens - len(combined),
                           len(source_tokens) - offset)
                combined.extend(source_tokens[offset:offset + take])
                consumed[relative_path] += take
                made_progress = True
                if len(combined) >= self.target_tokens:
                    break
            if not made_progress:
                break

        combined = combined[:self.target_tokens]
        if len(combined) < CODE_CONTEXT_MIN_TOKENS:
            raise ValueError(
                "real AgentCode and repository material is below 2,048 tokens")
        context_text = self._decode(combined)
        token_count = len(self._encode(context_text))
        if not CODE_CONTEXT_MIN_TOKENS <= token_count <= CODE_CONTEXT_MAX_TOKENS:
            raise ValueError(
                f"paper-pressure context has {token_count} tokenizer tokens")
        return CodeContext(
            text=context_text,
            token_count=token_count,
            sources=tuple(path for path, _ in self._source_sections),
        )


def _set_tool_duration(node: Any, duration_s: float) -> None:
    node.mcp_function.excute_time = float(duration_s)
    node.add_kvargs(expected_tool_stall_s=float(duration_s))


def _assign_deduplicating_input_composer(node: Any) -> None:

    def composer(task: TransferDataItem) -> LLMTextChunkChain:
        if isinstance(task.data, LLMTextChunkChain):
            chains = [task.data]
        elif isinstance(task.data, dict):
            chains = list(task.data.values())
        else:
            raise TypeError("Unsupported paper-pressure join input")

        if not chains:
            return LLMTextChunkChain(chunks=[node.system_prompt])
        common_length = min(len(chain.chunks) for chain in chains)
        for index in range(common_length):
            first_text = chains[0].chunks[index].text
            if any(chain.chunks[index].text != first_text
                   for chain in chains[1:]):
                common_length = index
                break

        chunks = [node.system_prompt]
        chunks.extend(chains[0].chunks[:common_length])
        for chain in chains:
            chunks.extend(chain.chunks[common_length:])
        return LLMTextChunkChain(chunks=chunks)

    node.assign_composer(is_input=True, composer=composer)


def _annotate(
    node: Any,
    *,
    stage_type: str,
    branch_id: str = "shared",
    branch_group: str = "shared",
    join_group: str = "",
    critical_path: bool = False,
    near_completion: bool = False,
    offload_eligible: bool = False,
    memory_weight: float = 1.0,
    fanout_width: int = 3,
) -> Any:
    node.add_kvargs(
        workload_profile=CODE_PROFILE,
        stage_type=stage_type,
        branch_id=branch_id,
        branch_group=branch_group,
        join_group=join_group,
        critical_path=critical_path,
        near_completion=near_completion,
        offload_eligible=offload_eligible,
        memory_weight=memory_weight,
        fanout_width=fanout_width,
    )
    return node


def _role_validation_node(
    *,
    role: str,
    branch_id: str,
    critical_path: bool,
    duration_s: float,
    name: str,
) -> McpToolCallNode:
    metadata = programmer_meta if role == "programmer" else revisor_meta
    system_prompt = PROGRAMMER_PROMPT if role == "programmer" else REVISER_PROMPT
    node = McpToolCallNode(
        metadata=metadata,
        tool_name="branch_patch_validation",
        tool_args={
            "branch": branch_id,
            "checks": ["lint", "test", "dependencies"]
        },
        name=name,
    )
    node.system_prompt = system_prompt
    node.node_type = role
    node.add_kvargs(
        preserve_llm_output_after_tool=True,
        reusable_prefix=True,
    )
    _set_tool_duration(node, duration_s)
    return _annotate(
        node,
        stage_type="branch_validation",
        branch_id=branch_id,
        branch_group="implementation",
        join_group="implementation_review",
        critical_path=critical_path,
        offload_eligible=not critical_path,
        memory_weight=1.6,
    )


class CodeWriterMcpApplication(LLMApplication):

    profile_name = CODE_PROFILE

    def __init__(
        self,
        llm_metadata: LLMCallMetadata,
        prompt: str,
        *,
        context_token_count: int,
        context_sources: tuple[str, ...],
    ) -> None:
        self.context_token_count = context_token_count
        self.context_sources = context_sources
        graph = InDegreePriorityGraph()

        start = _annotate(
            StartMcpNode([], llm_metadata, prompt),
            stage_type="start",
            critical_path=True,
        )
        graph.add_node(start)
        graph.add_edge(graph.entry_node_uuid, start.uuid)

        enter_node = start
        for index in range(2):
            architect = _annotate(
                McpArchitectNode(
                    metadata=architect_meta,
                    architect_type="planner",
                    name=f"architect_{index + 1}",
                ),
                stage_type="planning",
                critical_path=True,
            )
            graph.add_node(architect)
            graph.add_edge(enter_node.uuid, architect.uuid)
            if index == 0:
                start.architect_uuids = [architect.uuid]
                start.setup_composer()

            search = _annotate(
                McpSearchNode(
                    metadata=mcp_meta,
                    search_query=
                    "repository APIs, dependencies, and related solutions",
                    search_type="repository_and_web",
                    name=f"search_{index + 1}",
                ),
                stage_type="search",
                critical_path=True,
            )
            _set_tool_duration(search, 4.0)
            graph.add_node(search)
            graph.add_edge(architect.uuid, search.uuid)

            judger = _annotate(
                McpConditionalExecutionNode(
                    metadata=mcp_meta,
                    condition_description=
                    "judge whether the implementation plan is ready",
                    name=f"judger_{index + 1}",
                ),
                stage_type="plan_review",
                critical_path=True,
            )
            graph.add_node(judger)
            graph.add_edge(search.uuid, judger.uuid)
            enter_node = judger

        plan_write = _annotate(
            McpFileWriteNode(
                metadata=mcp_meta,
                file_path="plan.txt",
                file_content="implementation plan",
                name="file_write_plan",
            ),
            stage_type="file_write",
            critical_path=True,
        )
        _set_tool_duration(plan_write, 0.1)
        graph.add_node(plan_write)
        graph.add_edge(enter_node.uuid, plan_write.uuid)

        branch_writes = []
        for branch_index in range(1, 4):
            branch_id = f"programmer_{branch_index}"
            critical = branch_index == 1
            validate = _role_validation_node(
                role="programmer",
                branch_id=branch_id,
                critical_path=critical,
                duration_s=8.0,
                name=f"{branch_id}_validate_patch",
            )
            graph.add_node(validate)
            graph.add_edge(plan_write.uuid, validate.uuid)

            repair = _annotate(
                McpProgrammerNode(
                    metadata=programmer_meta,
                    name=f"{branch_id}_repair",
                ),
                stage_type="branch_repair",
                branch_id=branch_id,
                branch_group="implementation",
                join_group="implementation_review",
                critical_path=critical,
                memory_weight=1.7,
            )
            graph.add_node(repair)
            graph.add_edge(validate.uuid, repair.uuid)

            code_write = _annotate(
                McpFileWriteNode(
                    metadata=mcp_meta,
                    file_path=f"code_{branch_index}.py",
                    file_content=f"validated implementation {branch_index}",
                    name=f"code_write_{branch_index}",
                ),
                stage_type="file_write",
                branch_id=branch_id,
                branch_group="implementation",
                join_group="implementation_review",
                critical_path=critical,
            )
            _set_tool_duration(code_write, 0.1)
            graph.add_node(code_write)
            graph.add_edge(repair.uuid, code_write.uuid)
            branch_writes.append(code_write)

        reviewers = []
        for reviewer_index in range(1, 3):
            reviewer = _annotate(
                McpReviewerNode(
                    metadata=reviewer_meta,
                    input_node_uuids=[node.uuid for node in branch_writes],
                    name=f"reviewer_{reviewer_index}",
                ),
                stage_type="review",
                branch_id=f"reviewer_{reviewer_index}",
                branch_group="review",
                join_group="revision",
                critical_path=reviewer_index == 1,
                memory_weight=2.0,
            )
            _assign_deduplicating_input_composer(reviewer)
            graph.add_node(reviewer)
            for code_write in branch_writes:
                graph.add_edge(code_write.uuid, reviewer.uuid)
            reviewers.append(reviewer)

        revised_writes = []
        for revision_index in range(1, 3):
            branch_id = f"reviser_{revision_index}"
            critical = revision_index == 1
            validate = _role_validation_node(
                role="reviser",
                branch_id=branch_id,
                critical_path=critical,
                duration_s=10.0,
                name=f"{branch_id}_validate_patch",
            )
            validate.add_kvargs(
                branch_group="revision",
                join_group="final_merge",
                stage_type="revision_validation",
            )
            _assign_deduplicating_input_composer(validate)
            graph.add_node(validate)
            for reviewer in reviewers:
                graph.add_edge(reviewer.uuid, validate.uuid)

            repair = _annotate(
                McpReviserNode(
                    metadata=revisor_meta,
                    reviser_type="default",
                    name=f"{branch_id}_repair",
                ),
                stage_type="revision_repair",
                branch_id=branch_id,
                branch_group="revision",
                join_group="final_merge",
                critical_path=critical,
                near_completion=True,
                memory_weight=2.2,
            )
            graph.add_node(repair)
            graph.add_edge(validate.uuid, repair.uuid)

            revised_write = _annotate(
                McpFileWriteNode(
                    metadata=mcp_meta,
                    file_path=f"revised_code_{revision_index}.py",
                    file_content=f"validated revision {revision_index}",
                    name=f"revised_code_write_{revision_index}",
                ),
                stage_type="file_write",
                branch_id=branch_id,
                branch_group="revision",
                join_group="final_merge",
                critical_path=critical,
                near_completion=True,
            )
            _set_tool_duration(revised_write, 0.1)
            graph.add_node(revised_write)
            graph.add_edge(repair.uuid, revised_write.uuid)
            revised_writes.append(revised_write)

        external_eval = _annotate(
            McpExternalToolEvalNode(
                metadata=mcp_meta,
                tool_name="external_patch_evaluator",
                eval_target="all reviewed code",
                name="external_eval",
            ),
            stage_type="external_evaluation",
            branch_id="external_test",
            branch_group="verification",
            join_group="final_merge",
            critical_path=True,
            memory_weight=2.0,
        )
        _set_tool_duration(external_eval, 4.5)
        graph.add_node(external_eval)
        graph.add_edge(reviewers[1].uuid, external_eval.uuid)

        user_confirm = _annotate(
            McpUserConfirmNode(
                metadata=mcp_meta,
                confirm_content="confirm the external evaluation report",
                name="user_confirm",
            ),
            stage_type="user_confirmation",
            branch_id="external_test",
            branch_group="verification",
            join_group="final_merge",
            critical_path=True,
            near_completion=True,
        )
        graph.add_node(user_confirm)
        graph.add_edge(external_eval.uuid, user_confirm.uuid)

        external_test = _annotate(
            McpExternalTestNode(
                metadata=llm_metadata,
                test_target="merged implementation",
                name="test_node",
            ),
            stage_type="external_test",
            branch_id="external_test",
            branch_group="verification",
            join_group="final_merge",
            critical_path=True,
            near_completion=True,
            memory_weight=2.2,
        )
        _set_tool_duration(external_test, 5.5)
        graph.add_node(external_test)
        graph.add_edge(user_confirm.uuid, external_test.uuid)

        end = _annotate(
            EndMcpNode(metadata=llm_metadata),
            stage_type="final_merge",
            join_group="final_merge",
            critical_path=True,
            near_completion=True,
        )
        graph.add_node(end)
        graph.add_edge(external_test.uuid, end.uuid)
        for revised_write in revised_writes:
            graph.add_edge(revised_write.uuid, end.uuid)
        graph.add_edge(end.uuid, graph.exit_node_uuid)

        graph.set_priorities()
        app_max_depth = max(
            node.kvargs.get("depth", 0) for node in graph.nodes.values())
        for node in graph.nodes.values():
            depth = int(node.kvargs.get("depth", 0) or 0)
            node.add_kvargs(
                app_max_depth=app_max_depth,
                remaining_depth=max(0, app_max_depth - depth),
                dependency_depth=depth,
            )
        super().__init__(graph)

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from uuid import UUID

import httpx

from agent.app.request import ApplicationRequest
from agent.graph.meta import LLMTextChunk, LLMTextChunkChain, TransferDataItem
from agent.graph.node import Node
from agent.mcp.mcp_node import McpNode


REQUEST_TIMEOUT = 600000


@dataclass
class ParrotNodeTrace:
    name: str
    node_type: Optional[str]
    latency: float
    llm_latency: float = 0.0
    tool_latency: float = 0.0
    residual_latency: float = 0.0
    base_tool_latency: float = 0.0
    scheduled_tool_latency: float = 0.0
    is_mcp: bool = False
    input_chars: int = 0
    output_chars: int = 0
    request_id: Optional[int] = None
    output_var_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ParrotApplicationResult:
    app_latency: float
    app_start_time: float
    app_end_time: float
    request_info: dict[str, dict[str, Any]]
    middle_outputs: list[str] = field(default_factory=list)
    output: str = ""


def apply_tool_latency_noise(base_latency: float,
                             profile: str = "none",
                             scale: float = 0.0) -> float:
    if base_latency <= 0 or profile == "none" or scale <= 0:
        return max(0.0, base_latency)

    if profile == "gaussian":
        sampled = random.gauss(base_latency, base_latency * scale)
    elif profile == "uniform":
        delta = base_latency * scale
        sampled = random.uniform(base_latency - delta, base_latency + delta)
    else:
        raise ValueError(f"Unsupported tool noise profile: {profile}")

    return max(0.0, sampled)


class ParrotHttpClient:
    def __init__(self,
                 *,
                 base_url: str,
                 model_path: str,
                 output_criteria: str = "latency") -> None:
        self.base_url = base_url.rstrip("/")
        self.model_path = model_path
        self.output_criteria = output_criteria
        self.session_id: Optional[int] = None
        self.session_auth = "1"
        self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    async def close(self) -> None:
        await self._client.aclose()

    async def register_session(self) -> None:
        response = await self._client.post(f"{self.base_url}/v1/session",
                                           json={})
        response.raise_for_status()
        payload = response.json()
        self.session_id = int(payload["session_id"])
        self.session_auth = str(payload.get("session_auth", "1"))

    async def remove_session(self) -> None:
        if self.session_id is None:
            return
        response = await self._client.request(
            "DELETE",
            f"{self.base_url}/v1/session/{self.session_id}",
            json={"session_auth": self.session_auth},
        )
        response.raise_for_status()
        self.session_id = None

    async def get_semantic_variable(self, var_id: str) -> str:
        if self.session_id is None:
            raise RuntimeError("Parrot session has not been registered")
        response = await self._client.request(
            "GET",
            f"{self.base_url}/v1/semantic_var/{var_id}",
            json={
                "session_id": self.session_id,
                "session_auth": self.session_auth,
                "criteria": self.output_criteria,
            },
        )
        response.raise_for_status()
        return str(response.json().get("content", ""))

    async def semantic_completion(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ) -> tuple[str, int, str]:
        if self.session_id is None:
            raise RuntimeError("Parrot session has not been registered")

        payload = {
            "session_id": self.session_id,
            "session_auth": self.session_auth,
            "template": "{{prompt}}{{output}}",
            "parameters": [
                {
                    "name": "prompt",
                    "is_output": False,
                    "value": prompt,
                },
                {
                    "name": "output",
                    "is_output": True,
                    "sampling_config": {
                        "max_gen_length": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "ignore_tokenizer_eos": True,
                    },
                },
            ],
            "models": [self.model_path],
            "output_criteria": self.output_criteria,
        }
        response = await self._client.post(f"{self.base_url}/v1/semantic_call",
                                           json=payload)
        response.raise_for_status()
        data = response.json()
        output_var_id = None
        for param in data.get("param_info", []):
            if param.get("is_output"):
                output_var_id = str(param["var_id"])
                break
        if output_var_id is None:
            raise RuntimeError(f"Parrot response missing output var: {data}")
        text = await self.get_semantic_variable(output_var_id)
        return text, int(data["request_id"]), output_var_id


class ParrotGraphExecutor:
    def __init__(
        self,
        *,
        parrot_url: str,
        model_path: str,
        max_new_tokens_cap: int = 0,
        tool_noise_profile: str = "none",
        tool_noise_scale: float = 0.0,
        output_criteria: str = "latency",
        node_concurrency: int = 1,
        llm_concurrency: int = 1,
    ) -> None:
        self.parrot_url = parrot_url
        self.model_path = model_path
        self.max_new_tokens_cap = max_new_tokens_cap
        self.tool_noise_profile = tool_noise_profile
        self.tool_noise_scale = tool_noise_scale
        self.output_criteria = output_criteria
        self.node_concurrency = max(1, int(node_concurrency))
        self.llm_concurrency = max(1, int(llm_concurrency))
        self._llm_semaphore = asyncio.Semaphore(self.llm_concurrency)

    def _max_tokens_for_node(self, node: Node) -> int:
        max_tokens = int(getattr(node.metadata, "max_new_tokens", 16))
        if self.max_new_tokens_cap > 0:
            max_tokens = min(max_tokens, self.max_new_tokens_cap)
        return max(1, max_tokens)

    async def _run_llm(
        self,
        *,
        client: ParrotHttpClient,
        node: Node,
        prompt: str,
    ) -> tuple[str, int, str, float]:
        llm_start = time.time()
        async with self._llm_semaphore:
            text, request_id, output_var_id = await client.semantic_completion(
                prompt=prompt,
                max_tokens=self._max_tokens_for_node(node),
                temperature=float(
                    getattr(node.metadata, "temperature", 0.0) or 0.0),
            )
        return text, request_id, output_var_id, time.time() - llm_start

    async def _run_node(
        self,
        *,
        client: ParrotHttpClient,
        app_req: ApplicationRequest,
        node_uuid: UUID,
        all_start_time: float,
    ) -> tuple[UUID, Any, ParrotNodeTrace]:
        node = app_req.get_node(node_uuid)
        node_input = app_req.get_node_input(node_uuid)
        app_req.submit_input(node_uuid, node_input)
        start = time.time()

        if node.type_step == 1:
            output = node(node_input)
            latency = time.time() - start
            return node_uuid, output, ParrotNodeTrace(
                name=node.name or str(node.uuid),
                node_type=node.node_type,
                latency=latency,
                residual_latency=latency,
                input_chars=len(node_input.to_text()),
                output_chars=len(output.to_text())
                if isinstance(output, TransferDataItem) else len(str(output)),
            )

        input_chain = node_input
        prompt = input_chain.to_text()
        output_text, request_id, output_var_id, llm_latency = await self._run_llm(
            client=client,
            node=node,
            prompt=prompt,
        )

        final_output: Any = output_text
        tool_latency = 0.0
        base_tool_latency = 0.0
        scheduled_tool_latency = 0.0
        if isinstance(node, McpNode):
            concat_chain = LLMTextChunkChain(
                chunks=input_chain.chunks + [LLMTextChunk.from_text(output_text)])
            postprocess_item = node.postprocess(concat_chain)
            mcp_input_chain = node.mcp_input(postprocess_item)

            base_tool_latency = float(node.mcp_function.excute_time)
            scheduled_tool_latency = apply_tool_latency_noise(
                base_tool_latency,
                self.tool_noise_profile,
                self.tool_noise_scale,
            )
            tool_start = time.time()
            await asyncio.sleep(scheduled_tool_latency)
            tool_latency = time.time() - tool_start

            mcp_output_chain = node.mcp_output(mcp_input_chain)
            final_output = mcp_output_chain.to_text()
            if node.kvargs.get("preserve_llm_output_after_tool", False):
                separator = "" if output_text.endswith("\n") else "\n"
                final_output = output_text + separator + final_output

        latency = time.time() - start
        output_chars = (len(final_output.to_text())
                        if isinstance(final_output, TransferDataItem)
                        else len(str(final_output)))
        return node_uuid, final_output, ParrotNodeTrace(
            name=node.name or str(node.uuid),
            node_type=node.node_type,
            latency=latency,
            llm_latency=llm_latency,
            tool_latency=tool_latency,
            residual_latency=max(0.0, latency - llm_latency - tool_latency),
            base_tool_latency=base_tool_latency,
            scheduled_tool_latency=scheduled_tool_latency,
            is_mcp=isinstance(node, McpNode),
            input_chars=len(prompt),
            output_chars=output_chars,
            request_id=request_id,
            output_var_id=output_var_id,
        )

    async def run_application(self,
                              app_req: ApplicationRequest,
                              *,
                              all_start_time: Optional[float] = None
                              ) -> ParrotApplicationResult:
        if all_start_time is None:
            all_start_time = time.time()
        app_start = time.time()
        request_info: dict[str, dict[str, Any]] = {}

        client = ParrotHttpClient(base_url=self.parrot_url,
                                  model_path=self.model_path,
                                  output_criteria=self.output_criteria)
        await client.register_session()
        try:
            tasks: list[asyncio.Task] = []
            running: list[UUID] = []

            def schedule_ready_nodes() -> None:
                for waiting_uuid in list(app_req.waiting_or_running):
                    if len(running) >= self.node_concurrency:
                        return
                    if waiting_uuid == app_req.application.graph.exit_node_uuid:
                        continue
                    if waiting_uuid in running:
                        continue
                    try:
                        app_req.get_node_input(waiting_uuid)
                    except AssertionError:
                        continue
                    tasks.append(
                        asyncio.create_task(
                            self._run_node(client=client,
                                           app_req=app_req,
                                           node_uuid=waiting_uuid,
                                           all_start_time=all_start_time)))
                    running.append(waiting_uuid)

            schedule_ready_nodes()
            while tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)
                for task in done:
                    try:
                        finished_uuid, output, trace = task.result()
                    except Exception:
                        for pending_task in tasks:
                            pending_task.cancel()
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                    request_info[trace.name] = asdict(trace)
                    app_req.submit_output(finished_uuid, output)
                    running.remove(finished_uuid)
                schedule_ready_nodes()

            exit_uuid = app_req.application.graph.exit_node_uuid
            if exit_uuid in app_req.queued_outputs and not app_req.output:
                app_req.output = app_req.queued_outputs[exit_uuid].to_text()

            app_end = time.time()
            return ParrotApplicationResult(
                app_latency=app_end - app_start,
                app_start_time=app_start - all_start_time,
                app_end_time=app_end - all_start_time,
                request_info=request_info,
                middle_outputs=[str(item) for item in app_req.middle_outputs],
                output=app_req.output,
            )
        finally:
            try:
                await client.remove_session()
            except Exception:
                pass
            await client.close()

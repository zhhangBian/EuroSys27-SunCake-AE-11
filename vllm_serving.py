import argparse
import asyncio
import hashlib
import json
import random
import time
import numpy as np
import os
import httpx
from pathlib import Path

from openai import AsyncOpenAI
from typing import Any, Dict, List, Optional, Tuple

from agent.graph.meta import LLMCallMetadata, LLMTextChunk, LLMTextChunkChain
from agent.graph.node import Node
from agent.app.request import ApplicationRequest
from agent.mcp.mcp_node import McpNode
from agent.vllm_prompt import CODE_WRITER_PROMPT

from agent.app.code_writer_mcp import (
    CodeWriterMcpApplication,
    CODE_PROFILE,
    CodeContextBuilder,
)
from agent.app.deep_research_mcp import (
    DeepResearchMcpApplication,
    ResearchPressureContextBuilder,
)
from ae.protocol import request_metadata

all_start_time = time.time()

OPENAI_BASE = None
MODEL_PATH = None
OPENAI_CLIENT = None

MCP_URL = None
MCP_FINISHED_URL = None

REQUEST_TIMEOUT = 600000

HTTPX_CLIENT = httpx.AsyncClient()

APP_REQ_LATENCIES = []
APPLICATION_INFO = {}
IO_RECORD = {}


def sample_actual_tool_latency(base_latency: float, args: argparse.Namespace) -> float:
    profile = getattr(args, "tool_prediction_error_profile", "none")
    scale = float(getattr(args, "tool_prediction_error_scale", 0.0) or 0.0)
    if base_latency <= 0 or profile == "none" or scale <= 0:
        return max(0.0, base_latency)

    if profile == "gaussian":
        sampled = random.gauss(base_latency, base_latency * scale)
    elif profile == "uniform":
        delta = base_latency * scale
        sampled = random.uniform(base_latency - delta, base_latency + delta)
    else:
        raise ValueError(f"Unsupported tool prediction-error profile: {profile}")

    return max(0.0, sampled)


class DatasetReader:
    def __init__(self, dataset_path: str):
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)
        self.index = 0
        self.dataset_len = len(self.dataset)

    def get_prompt(self) -> str:
        prompt = self.dataset[str(self.index % self.dataset_len)]["prompt"]
        self.index += 1
        return prompt


dataset_reader = None


def generate_requests_app(
    dataset_path: str,
    num_requests: int,
    task: str,
    kvargs: Dict[str, Any],
) -> List[ApplicationRequest]:
    prompt = CODE_WRITER_PROMPT
    global dataset_reader
    dataset_reader = DatasetReader(dataset_path)

    if task == CODE_PROFILE:
        application = CodeWriterMcpApplication
        builder_type = CodeContextBuilder
    elif task == "research":
        application = DeepResearchMcpApplication
        builder_type = ResearchPressureContextBuilder
    else:
        raise ValueError(f"Invalid task: {task}")

    from vllm.tokenizers import get_tokenizer

    context_builder = builder_type(
        get_tokenizer(MODEL_PATH), Path(__file__).resolve().parent
    )
    requests = []
    for _ in range(num_requests):
        context = context_builder.build(dataset_reader.get_prompt())
        app_request = ApplicationRequest(
            prompt=prompt,
            application=application(
                kvargs["llm_metadata"],
                prompt=context.text,
                context_token_count=context.token_count,
                context_sources=context.sources,
            ),
        )
        synthetic_node_ids = {
            app_request.application.graph.entry_node_uuid,
            app_request.application.graph.exit_node_uuid,
        }
        executable_nodes = [
            node
            for node_uuid, node in app_request.application.graph.nodes.items()
            if node_uuid not in synthetic_node_ids
        ]
        app_request.frozen_workload_contract = {
            "application": application.__name__,
            "task": task,
            "application_prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "dataset_prompt_sha256": hashlib.sha256(
                context.text.encode("utf-8")
            ).hexdigest(),
            "expected_dag_nodes": sorted(
                (
                    {
                        "name": node.name,
                        "type": node.node_type,
                        "is_mcp": isinstance(node, McpNode),
                    }
                    for node in executable_nodes
                ),
                key=lambda item: item["name"],
            ),
        }
        requests.append(app_request)
    return requests


def compose_mcp_final_output(
    node: McpNode, generated_output: str, tool_output: str
) -> str:
    if not node.kvargs.get("preserve_llm_output_after_tool", False):
        return tool_output
    separator = "" if generated_output.endswith("\n") else "\n"
    return generated_output + separator + tool_output


async def async_post(url, json_data):
    response = await HTTPX_CLIENT.post(url, json=json_data, timeout=35.0)
    response.raise_for_status()
    payload = response.json()
    if payload.get("disposition") not in {"applied", "duplicate", "late_finish"}:
        raise RuntimeError(f"Tool lifecycle event was not accepted: {payload}")
    return payload


async def send_application_request(
    app_req: ApplicationRequest,
    id: int,
    args: argparse.Namespace,
    arrival_offset_s: float = 0.0,
) -> None:
    tasks = []
    running_node_uuids = []
    app_start_time = time.time()
    app_req_times, app_req_lock = [], asyncio.Lock()
    app_max_depth = max(
        node.kvargs.get("depth", 0) for node in app_req.application.graph.nodes.values()
    )
    synthetic_node_ids = {
        app_req.application.graph.entry_node_uuid,
        app_req.application.graph.exit_node_uuid,
    }
    expected_nodes = [
        node
        for node_uuid, node in app_req.application.graph.nodes.items()
        if node_uuid not in synthetic_node_ids
    ]
    expected_node_names = [node.name for node in expected_nodes]
    APPLICATION_INFO[id] = {
        "app_start_time": app_start_time - all_start_time,
        "arrival_offset_s": arrival_offset_s,
        "request_info": {},
        "workload_profile": getattr(app_req.application, "profile_name", "canonical"),
        "context_token_count": getattr(
            app_req.application, "context_token_count", None
        ),
        "context_sources": list(getattr(app_req.application, "context_sources", ())),
        "expected_node_names": sorted(expected_node_names),
        "expected_node_names_unique": (
            len(expected_node_names) == len(set(expected_node_names))
        ),
        "frozen_workload_contract": getattr(app_req, "frozen_workload_contract", {}),
    }
    IO_RECORD[id] = {}
    smoke_max_finished_nodes = max(
        0, int(getattr(args, "smoke_max_finished_nodes", 0) or 0)
    )

    async def call_vllm_api(prompt: str, node: Node, MODEL_PATH: str, max_tokens: int):
        url = f"http://localhost:{args.port}/v1/completions"
        headers = {"Content-Type": "application/json"}
        agent_info = {
            "application_id": id,
            "app_start_time": app_start_time,
            "app_start_offset": app_start_time - all_start_time,
            "app_elapsed_time": time.time() - app_start_time,
            "name": node.name,
            "type": node.node_type,
            "start_time": str(time.time()),
            "priority": node.kvargs.get("priority", 0),
            "in_degree": node.kvargs.get("in_degree", 0),
            "out_degree": node.kvargs.get("out_degree", 0),
            "similarity": node.kvargs.get("similarity", 0),
            "depth": node.kvargs.get("depth", 0),
            "app_max_depth": app_max_depth,
            "remaining_depth": max(0, app_max_depth - node.kvargs.get("depth", 0)),
        }
        for field in (
            "workload_profile",
            "branch_id",
            "branch_group",
            "critical_path",
            "offload_eligible",
            "expected_tool_stall_s",
            "memory_weight",
            "stage_type",
            "near_completion",
            "fanout_width",
            "join_group",
            "dependency_depth",
            "reusable_prefix",
            "preserve_llm_output_after_tool",
        ):
            if field in node.kvargs:
                agent_info[field] = node.kvargs[field]
        expected_stall = float(agent_info.get("expected_tool_stall_s", 0.0) or 0.0)
        if expected_stall > 0:
            agent_info["resume_deadline"] = time.time() + expected_stall
        if isinstance(node, McpNode):
            agent_info.setdefault(
                "offload_eligible",
                bool(node.kvargs.get("preserve_llm_output_after_tool", False)),
            )

        data = {
            "model": MODEL_PATH,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
        }
        metadata = None
        if args.protocol == "tokencake":
            metadata = request_metadata(agent_info)
            data["request_id"] = metadata["lifecycle_id"]
            data["vllm_xargs"] = {"tokencake": metadata}

        resp = await HTTPX_CLIENT.post(
            url, headers=headers, json=data, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        result = resp.json()
        result["lifecycle_id"] = metadata["lifecycle_id"] if metadata else None
        return result

    async def send_request_with_timeout(
        node: Node, prompt: str, max_tokens: int
    ) -> Tuple[Optional[Any], str]:
        try:
            completion_task = asyncio.create_task(
                call_vllm_api(prompt, node, MODEL_PATH, max_tokens)
            )

            completion = await asyncio.wait_for(
                completion_task, timeout=REQUEST_TIMEOUT
            )
            return completion, ""
        except asyncio.TimeoutError:
            completion_task.cancel()
            try:
                await completion_task
            except asyncio.CancelledError:
                print(f"[timeout][{node.name}] request cancelled")
            return None, "request timeout"
        except Exception as e:
            return None, (f"request error: {type(e).__name__}: {e!r}")

    async def send_llm_request(
        node: Node,
        prompt: str,
        max_new_tokens: int,
        request_start_time: float,
    ) -> Tuple[str, str, Dict[str, Any]]:
        llm_completion, error_msg = await send_request_with_timeout(
            node, prompt, max_new_tokens
        )
        if llm_completion is not None:
            completion = llm_completion
            completion_id = completion["id"]
            output_text = completion["choices"][0]["text"]

            choice = completion["choices"][0]
            usage = completion.get("usage") or {}
            completion_metadata = {
                "lifecycle_id": completion.get("lifecycle_id"),
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "generated_tokens": int(usage.get("completion_tokens", 0) or 0),
                "processed_tokens": int(
                    usage.get("total_tokens", 0)
                    or (
                        (usage.get("prompt_tokens", 0) or 0)
                        + (usage.get("completion_tokens", 0) or 0)
                    )
                ),
            }
            if args.record_output:
                IO_RECORD[id][node.name] = {
                    "input": prompt,
                    "output": output_text,
                    "latency": time.time() - request_start_time,
                    "request_id": completion_id,
                    **completion_metadata,
                }
            return output_text, completion_id, completion_metadata
        raise RuntimeError(f"request error: {node.name}: {error_msg}")

    async def send_request(node: Node, node_input: LLMTextChunkChain):
        if node.type_step == 1:
            request_start_time = time.time()
            output = node(node_input)
            request_end_time = time.time()
            APPLICATION_INFO[id]["request_info"][node.name] = {
                "name": node.name,
                "type": node.node_type,
                "execution_kind": "local",
                "finish_reason": "local",
                "request_id": None,
                "latency": request_end_time - request_start_time,
                "llm_latency": 0.0,
                "tool_latency": 0.0,
                "residual_latency": request_end_time - request_start_time,
                "base_tool_latency": 0.0,
                "scheduled_tool_latency": 0.0,
                "estimated_tool_latency": 0.0,
                "sampled_actual_tool_latency": 0.0,
                "tool_prediction_error_s": 0.0,
                "is_mcp": False,
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "processed_tokens": 0,
                "start_time": request_start_time - all_start_time,
                "end_time": request_end_time - all_start_time,
            }
            return node.uuid, output

        merged_input_text = node_input.to_text()
        max_new_tokens = node.metadata.max_new_tokens
        max_new_tokens_cap = int(getattr(args, "max_new_tokens_cap", 0) or 0)
        if max_new_tokens_cap > 0:
            max_new_tokens = min(max_new_tokens, max_new_tokens_cap)

        request_start_time = time.time()
        llm_latency = 0.0
        tool_latency = 0.0
        scheduled_tool_latency = 0.0
        base_tool_latency = 0.0
        estimated_tool_latency = 0.0
        actual_tool_latency = 0.0

        if args.debug:
            output_text = f"[DEBUG-OUTPUT][{node.name}]'s output\n"
            completion_id = f"debug-{node.name}-{node.uuid}"
            completion_metadata = {
                "finish_reason": "stop",
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "processed_tokens": 0,
            }
        else:
            llm_start_time = time.time()
            (output_text, completion_id, completion_metadata) = await send_llm_request(
                node, merged_input_text, max_new_tokens, request_start_time
            )
            llm_latency = time.time() - llm_start_time

        final_output = output_text
        if isinstance(node, McpNode):
            concat_chain = LLMTextChunkChain(
                chunks=node_input.chunks + [LLMTextChunk.from_text(output_text)]
            )
            postprocess_item = node.postprocess(concat_chain)

            mcp_input_chain = node.mcp_input(postprocess_item)
            if args.debug:
                print(
                    f"\n\n[MCP] {node.mcp_function.name}'s mcp input:\n{mcp_input_chain.to_text()}"
                )
                print("=" * 100)

            base_tool_latency = float(node.mcp_function.excute_time)
            estimated_tool_latency = base_tool_latency
            actual_tool_latency = sample_actual_tool_latency(base_tool_latency, args)
            scheduled_tool_latency = actual_tool_latency
            request_id = completion_metadata.get("lifecycle_id")

            tool_start_time = time.time()
            if (
                not args.debug
                and not args.disable_mcp_notifications
                and args.protocol == "tokencake"
            ):
                mcp_req = {
                    "event": "stall_started",
                    "lifecycle_id": request_id,
                    "kind": node.mcp_function.name,
                }
                if estimated_tool_latency > 0:
                    mcp_req["estimated_duration_s"] = estimated_tool_latency
                print(f"[MCP] Offload: {mcp_req}")
                await async_post(MCP_URL, mcp_req)

                print(
                    "[MCP] Simulate execution: "
                    f"estimated={estimated_tool_latency}s, "
                    f"actual={actual_tool_latency}s"
                )
                await asyncio.sleep(actual_tool_latency)

                await async_post(
                    MCP_FINISHED_URL,
                    {"event": "stall_finished", "lifecycle_id": request_id},
                )
            elif not args.debug:
                print(
                    f"[MCP] Notifications disabled; simulate execution: "
                    f"sleep {actual_tool_latency}s"
                )
                await asyncio.sleep(actual_tool_latency)
            else:
                print(
                    f"\t[MCP] {node.mcp_function.name} running time: "
                    f"{actual_tool_latency}s"
                )
                if args.debug_sleep:
                    await asyncio.sleep(actual_tool_latency)
            tool_latency = time.time() - tool_start_time

            mcp_output_chain = node.mcp_output(mcp_input_chain)
            final_output = compose_mcp_final_output(
                node, output_text, mcp_output_chain.to_text()
            )
            if args.debug:
                print(
                    f"\n\n[MCP] {node.mcp_function.name}'s mcp final output:\n{final_output}"
                )
                print("=" * 100)

        request_end_time = time.time()
        request_latency = request_end_time - request_start_time

        APPLICATION_INFO[id]["request_info"][node.name] = {
            "name": node.name,
            "type": node.node_type,
            "latency": request_latency,
            "llm_latency": llm_latency,
            "tool_latency": tool_latency,
            "residual_latency": max(0.0, request_latency - llm_latency - tool_latency),
            "base_tool_latency": base_tool_latency,
            "scheduled_tool_latency": scheduled_tool_latency,
            "estimated_tool_latency": estimated_tool_latency,
            "sampled_actual_tool_latency": actual_tool_latency,
            "tool_prediction_error_s": (actual_tool_latency - estimated_tool_latency),
            "is_mcp": isinstance(node, McpNode),
            "execution_kind": "llm",
            "request_id": completion_id,
            "lifecycle_id": completion_metadata.get("lifecycle_id"),
            "finish_reason": completion_metadata["finish_reason"],
            "prompt_tokens": completion_metadata["prompt_tokens"],
            "generated_tokens": completion_metadata["generated_tokens"],
            "processed_tokens": completion_metadata["processed_tokens"],
            "start_time": request_start_time - all_start_time,
            "end_time": request_end_time - all_start_time,
        }

        async with app_req_lock:
            app_req_times.append((max_new_tokens, request_latency))

        return node.uuid, final_output

    for waiting_node_uuid in app_req.waiting_or_running:
        if waiting_node_uuid == app_req.application.graph.exit_node_uuid:
            continue
        waiting_node = app_req.get_node(waiting_node_uuid)
        node_input = app_req.get_node_input(waiting_node_uuid)
        app_req.submit_input(waiting_node_uuid, node_input)
        tasks.append(asyncio.create_task(send_request(waiting_node, node_input)))
        running_node_uuids.append(waiting_node_uuid)

    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        tasks = list(pending)

        for task in done:
            finish_node_uuid, output = task.result()

            app_req.submit_output(finish_node_uuid, output)

            running_node_uuids.remove(finish_node_uuid)

        if (
            smoke_max_finished_nodes > 0
            and len(app_req.finished_nodes) >= smoke_max_finished_nodes
        ):
            for pending_task in tasks:
                pending_task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            tasks = []
            print(
                f"[SMOKE] stopping application {id} after {len(app_req.finished_nodes)} finished nodes"
            )
            break

        for waiting_node_uuid in app_req.waiting_or_running:
            if waiting_node_uuid == app_req.application.graph.exit_node_uuid:
                continue
            if waiting_node_uuid not in running_node_uuids:
                waiting_node = app_req.get_node(waiting_node_uuid)
                node_input = app_req.get_node_input(waiting_node_uuid)
                app_req.submit_input(waiting_node_uuid, node_input)
                tasks.append(
                    asyncio.create_task(send_request(waiting_node, node_input))
                )
                running_node_uuids.append(waiting_node_uuid)

        if app_req.is_finished() and not tasks:
            break

    app_end_time = time.time()
    app_latency = app_end_time - app_start_time

    APPLICATION_INFO[id]["app_end_time"] = app_end_time - all_start_time
    APPLICATION_INFO[id]["app_latency"] = app_latency
    observed_node_names = sorted(APPLICATION_INFO[id]["request_info"])
    APPLICATION_INFO[id]["observed_node_names"] = observed_node_names
    APPLICATION_INFO[id]["application_internal_finished"] = app_req.is_finished()
    APPLICATION_INFO[id]["app_finished"] = (
        APPLICATION_INFO[id]["expected_node_names_unique"]
        and sorted(expected_node_names) == observed_node_names
    )

    APP_REQ_LATENCIES.append(app_latency)
    print(f"Finished {len(APP_REQ_LATENCIES)} requests")


async def benchmark(
    app_reqs: List[ApplicationRequest], args: argparse.Namespace
) -> None:
    global OPENAI_CLIENT

    OPENAI_CLIENT = AsyncOpenAI(
        base_url=OPENAI_BASE, api_key="EMPTY", timeout=REQUEST_TIMEOUT
    )

    tasks: List[asyncio.Task] = []

    request_rate = args.request_rate

    arrival_offsets = None
    if getattr(args, "arrival_trace_file", None):
        with open(args.arrival_trace_file, "r", encoding="utf-8") as handle:
            arrival_payload = json.load(handle)
        arrival_offsets = arrival_payload.get("offsets_s", arrival_payload)
        if not isinstance(arrival_offsets, list) or len(arrival_offsets) != len(
            app_reqs
        ):
            raise ValueError("arrival trace must contain one offset per application")
        arrival_offsets = [float(value) for value in arrival_offsets]
        if arrival_offsets != sorted(arrival_offsets) or any(
            value < 0 for value in arrival_offsets
        ):
            raise ValueError(
                "arrival trace offsets must be non-negative and non-decreasing"
            )

    if request_rate == float("inf") and arrival_offsets is None:
        for i, app_req in enumerate(app_reqs):
            task = asyncio.create_task(send_application_request(app_req, i, args, 0.0))
            tasks.append(task)
    else:
        sub_tasks = []
        interval = 1.0 / request_rate

        async def schedule_request(i, app_req, expected_time, arrival_offset):
            current_time = time.time()
            wait_time = expected_time - current_time

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            return await send_application_request(app_req, i, args, arrival_offset)

        start_time = time.time()
        for i, app_req in enumerate(app_reqs):
            arrival_offset = (
                arrival_offsets[i] if arrival_offsets is not None else i * interval
            )
            task = asyncio.create_task(
                schedule_request(
                    i, app_req, start_time + arrival_offset, arrival_offset
                )
            )
            sub_tasks.append(task)

        tasks = sub_tasks

    await asyncio.gather(*tasks)

    output_file = os.path.join(
        args.output_dir, f"app_qps_{request_rate}_num_{len(app_reqs)}.json"
    )
    with open(output_file, "w") as f:
        json.dump(APPLICATION_INFO, f)


def serve(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    APP_REQ_LATENCIES.clear()
    APPLICATION_INFO.clear()
    IO_RECORD.clear()

    app_reqs = generate_requests_app(
        dataset_path=args.dataset,
        num_requests=args.num_requests,
        task=args.task,
        kvargs={
            "llm_metadata": LLMCallMetadata(
                model="gpt-4o-mini",
                max_new_tokens=500,
                temperature=0,
            )
        },
    )

    benchmark_start_time = time.time()
    asyncio.run(benchmark(app_reqs, args))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_requests / benchmark_time:.2f} requests/s")

    avg_latency = np.mean(APP_REQ_LATENCIES)
    print(f"Average latency: {avg_latency:.2f} s")
    max_, p99, p90, p50 = (
        np.max(APP_REQ_LATENCIES),
        np.percentile(APP_REQ_LATENCIES, 99),
        np.percentile(APP_REQ_LATENCIES, 90),
        np.percentile(APP_REQ_LATENCIES, 50),
    )
    print(f"Max latency: {max_:.2f} s")
    print(f"99th percentile latency: {p99:.2f} s")
    print(f"90th percentile latency: {p90:.2f} s")
    print(f"50th percentile latency: {p50:.2f} s")

    if args.record_output:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(IO_RECORD, f)
        print(f"Output record saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--debug_sleep", type=bool, default=False)
    parser.add_argument("--record_output", type=bool, default=True)
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"results/record/output_record_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )

    parser.add_argument("--port", type=str, default="8073")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, default="dataset/agentcodeclean_new.json"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=[CODE_PROFILE, "research"],
        default=CODE_PROFILE,
    )

    parser.add_argument("--request_rate", type=float, default=float("inf"))
    parser.add_argument("--num_requests", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument(
        "--tool_prediction_error_profile",
        "--tool_noise_profile",
        dest="tool_prediction_error_profile",
        type=str,
        choices=["none", "gaussian", "uniform"],
        default="none",
    )
    parser.add_argument(
        "--tool_prediction_error_scale",
        "--tool_noise_scale",
        dest="tool_prediction_error_scale",
        type=float,
        default=0.0,
    )
    parser.add_argument("--smoke_max_finished_nodes", type=int, default=0)
    parser.add_argument("--max_new_tokens_cap", type=int, default=0)
    parser.add_argument("--arrival_trace_file", type=str, default="")
    parser.add_argument("--disable_mcp_notifications", action="store_true")
    parser.add_argument(
        "--protocol", choices=["tokencake", "openai"], default="tokencake"
    )
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    OPENAI_BASE = f"http://localhost:{args.port}/v1"
    MCP_URL = f"http://localhost:{args.port}/v1/tokencake/events"
    MCP_FINISHED_URL = MCP_URL

    os.makedirs(args.output_dir, exist_ok=True)

    serve(args)

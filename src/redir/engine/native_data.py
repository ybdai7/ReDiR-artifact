"""Deterministic native-tool serialization for MT-AgentRisk saved states."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
import re
from typing import Any

from redir.server.native_messages import normalize_native_messages
from redir.server.qwen_native_protocol import parse_qwen_native_response


DEFAULT_FILESYSTEM_TOOL_NAMES = (
    "read_file",
    "read_text_file",
    "read_media_file",
    "read_multiple_files",
    "write_file",
    "create_directory",
    "list_directory",
    "list_directory_with_sizes",
    "directory_tree",
    "move_file",
    "search_files",
    "get_file_info",
    "list_allowed_directories",
)

_MCP_ACTION_RE = re.compile(
    r"I am interacting with the MCP server with name:\s*```\s*(?P<name>[^`]+?)\s*```"
    r"\s*and arguments:\s*```\s*(?P<arguments>.*?)\s*```",
    re.DOTALL,
)


def _finish_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the task or stop safely with a final message.",
            "parameters": {
                "type": "object",
                "required": ["message"],
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Final message to send to the user.",
                    }
                },
            },
        },
    }


def _filesystem_schema(name: str) -> dict[str, Any] | None:
    path = {"type": "string", "description": "Absolute path"}
    path_tools = {
        "read_file",
        "read_text_file",
        "read_media_file",
        "list_directory",
        "list_directory_with_sizes",
        "create_directory",
        "directory_tree",
        "get_file_info",
    }
    if name in path_tools:
        return {
            "type": "object",
            "properties": {"path": path},
            "required": ["path"],
        }
    if name == "write_file":
        return {
            "type": "object",
            "properties": {"path": path, "content": {"type": "string"}},
            "required": ["path", "content"],
        }
    if name == "move_file":
        return {
            "type": "object",
            "properties": {"source": path, "destination": path},
            "required": ["source", "destination"],
        }
    if name == "search_files":
        return {
            "type": "object",
            "properties": {"path": path, "pattern": {"type": "string"}},
            "required": ["path", "pattern"],
        }
    if name == "read_multiple_files":
        return {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["paths"],
        }
    if name == "list_allowed_directories":
        return {"type": "object", "properties": {}}
    return None


def _core_openhands_tools() -> list[dict[str, Any]]:
    from openhands.agenthub.codeact_agent.tools import (
        BrowserTool,
        FinishTool,
        IPythonTool,
        ThinkTool,
        create_cmd_run_tool,
        create_str_replace_editor_tool,
    )

    return [
        deepcopy(create_cmd_run_tool()),
        deepcopy(ThinkTool),
        deepcopy(FinishTool),
        deepcopy(BrowserTool),
        deepcopy(IPythonTool),
        deepcopy(create_str_replace_editor_tool()),
    ]


def _generic_tool(name: str) -> dict[str, Any]:
    schema = _filesystem_schema(name) or {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"OpenHands tool: {name}",
            "parameters": schema,
        },
    }


def native_tools_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct the exact 20-tool native set used by the OpenHands eval."""

    provided = {
        str((tool.get("function") or {}).get("name")): deepcopy(tool)
        for tool in record.get("available_tools") or []
        if isinstance(tool, dict) and (tool.get("function") or {}).get("name")
    }
    declared_names = [
        str(name)
        for name in record.get("available_tool_names") or []
        if str(name) and str(name) != "edit_file"
    ]
    for name in provided:
        if name != "edit_file" and name not in declared_names:
            declared_names.append(name)
    for name in DEFAULT_FILESYSTEM_TOOL_NAMES:
        if name not in declared_names:
            declared_names.append(name)

    tools = _core_openhands_tools()
    core_names = [str(tool["function"]["name"]) for tool in tools]
    for name in declared_names:
        if name in core_names:
            continue
        tool = provided.get(name) or _generic_tool(name)
        function = tool.get("function") or {}
        parameters = function.get("parameters") or {}
        if not parameters.get("properties") and _filesystem_schema(name) is not None:
            function["parameters"] = _filesystem_schema(name)
        tools.append(tool)
    if "fetch" not in core_names and "fetch" not in declared_names:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "fetch",
                    "description": "Fetch a URL and return its content.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        )
    if not any(tool["function"]["name"] == "finish" for tool in tools):
        tools.insert(0, _finish_tool())
    return tools


def _mcp_action_message(content: str) -> dict[str, Any] | None:
    match = _MCP_ACTION_RE.search(content)
    if match is None:
        return None
    raw_arguments = match.group("arguments").strip()
    try:
        arguments = ast.literal_eval(raw_arguments)
    except (SyntaxError, ValueError):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return {
        "role": "assistant",
        "content": content[: match.start()].strip(),
        "tool_calls": [
            {
                "id": "matched-state-replay",
                "type": "function",
                "function": {
                    "name": match.group("name").strip(),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def legacy_messages_to_native(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reserialize legacy assistant actions while preserving state semantics."""

    converted: list[dict[str, Any]] = []
    for raw_message in messages:
        message = deepcopy(raw_message)
        if message.get("role") != "assistant" or message.get("tool_calls"):
            converted.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str):
            converted.append(message)
            continue
        mcp_action = _mcp_action_message(content)
        if mcp_action is not None:
            converted.append(mcp_action)
            continue
        parsed = parse_qwen_native_response(
            content,
            tools,
            tool_call_id_factory=lambda: "matched-state-replay",
        )
        if not parsed.tool_calls:
            converted.append(message)
            continue
        native_message: dict[str, Any] = {
            "role": "assistant",
            "content": parsed.content or "",
            "tool_calls": parsed.tool_calls,
        }
        if parsed.reasoning_content is not None:
            native_message["reasoning"] = parsed.reasoning_content
        converted.append(native_message)
    return normalize_native_messages(converted)

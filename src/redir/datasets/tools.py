"""Filesystem tool schemas used by the MT-AgentRisk collection and test flow."""

from __future__ import annotations

from typing import Any, Iterable


FILESYSTEM_TOOL_NAMES = [
    "read_file",
    "read_text_file",
    "read_media_file",
    "read_multiple_files",
    "write_file",
    "edit_file",
    "create_directory",
    "list_directory",
    "list_directory_with_sizes",
    "directory_tree",
    "move_file",
    "search_files",
    "get_file_info",
    "list_allowed_directories",
]


def tool_names_for_dependencies(dependencies: Iterable[str]) -> list[str]:
    names: list[str] = []
    for dependency in dependencies:
        if dependency == "mcp-filesystem":
            names.extend(name for name in FILESYSTEM_TOOL_NAMES if name not in names)
    return names


def tool_schemas_for_names(names: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


__all__ = ["FILESYSTEM_TOOL_NAMES", "tool_names_for_dependencies", "tool_schemas_for_names"]

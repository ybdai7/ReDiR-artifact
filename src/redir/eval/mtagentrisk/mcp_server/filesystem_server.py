#!/usr/bin/env python
"""Filesystem MCP server for local MT-AgentRisk runs.

The OpenHands runtime sees the shared directory as /workspace. This server runs
on the host, so it maps /workspace paths back to MTAGENTRISK_SHARED_WORKSPACE.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any

from fastmcp import FastMCP


WORKSPACE_ALIAS = "/workspace"
DEFAULT_PORT = 19090

mcp = FastMCP("mtagentrisk-filesystem")


def _workspace_root() -> Path:
    root = os.environ.get("MTAGENTRISK_SHARED_WORKSPACE")
    if not root:
        root = str(Path.cwd() / "outputs" / "shared_workspace")
    return Path(root).expanduser().resolve()


def _resolve_workspace_path(path: str) -> Path:
    if not path:
        raise ValueError("path is required")

    root = _workspace_root()
    path_text = str(path).replace("\\", "/")

    if path_text == WORKSPACE_ALIAS:
        target = root
    elif path_text.startswith(f"{WORKSPACE_ALIAS}/"):
        target = root / path_text[len(WORKSPACE_ALIAS) + 1 :]
    else:
        raw = Path(path_text).expanduser()
        target = raw if raw.is_absolute() else root / raw

    resolved = target.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path outside allowed workspace: {path}")
    return resolved


def _display_path(path: Path) -> str:
    root = _workspace_root()
    resolved = path.resolve(strict=False)
    if resolved == root:
        return WORKSPACE_ALIAS
    try:
        return f"{WORKSPACE_ALIAS}/{resolved.relative_to(root).as_posix()}"
    except ValueError:
        return str(resolved)


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _display_path(path),
        "type": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "permissions": oct(stat.st_mode & 0o777),
    }


@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the allowed workspace."""
    return _resolve_workspace_path(path).read_text(encoding="utf-8")


@mcp.tool()
def read_text_file(path: str) -> str:
    """Read a UTF-8 text file from the allowed workspace."""
    return read_file(path)


@mcp.tool()
def read_media_file(path: str) -> str:
    """Read a binary file and return base64 content with a best-effort MIME type."""
    file_path = _resolve_workspace_path(path)
    mime_type, _ = mimetypes.guess_type(file_path.name)
    payload = {
        "path": _display_path(file_path),
        "mime_type": mime_type or "application/octet-stream",
        "base64": base64.b64encode(file_path.read_bytes()).decode("ascii"),
    }
    return json.dumps(payload, indent=2)


@mcp.tool()
def read_multiple_files(paths: list[str]) -> str:
    """Read multiple UTF-8 text files from the allowed workspace."""
    parts: list[str] = []
    for item in paths:
        file_path = _resolve_workspace_path(item)
        parts.append(f"--- {_display_path(file_path)} ---\n{file_path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text content to a file in the allowed workspace."""
    file_path = _resolve_workspace_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {_display_path(file_path)}"


@mcp.tool()
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace text in a UTF-8 file in the allowed workspace."""
    file_path = _resolve_workspace_path(path)
    text = file_path.read_text(encoding="utf-8")
    if old_text not in text:
        raise ValueError("old_text was not found")
    updated = text.replace(old_text, new_text, 1)
    file_path.write_text(updated, encoding="utf-8")
    return f"Edited {_display_path(file_path)}"


@mcp.tool()
def create_directory(path: str) -> str:
    """Create a directory under the allowed workspace."""
    dir_path = _resolve_workspace_path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return f"Created {_display_path(dir_path)}"


@mcp.tool()
def list_directory(path: str = WORKSPACE_ALIAS) -> str:
    """List immediate children of a directory in the allowed workspace."""
    dir_path = _resolve_workspace_path(path)
    if not dir_path.is_dir():
        raise NotADirectoryError(path)

    entries = []
    for child in sorted(dir_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        kind = "DIR" if child.is_dir() else "FILE"
        entries.append(f"[{kind}] {child.name}")
    return "\n".join(entries) if entries else "<empty>"


@mcp.tool()
def list_directory_with_sizes(path: str = WORKSPACE_ALIAS) -> str:
    """List immediate children of a directory with file sizes."""
    dir_path = _resolve_workspace_path(path)
    if not dir_path.is_dir():
        raise NotADirectoryError(path)

    entries = []
    for child in sorted(dir_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        kind = "DIR" if child.is_dir() else "FILE"
        size = "-" if child.is_dir() else str(child.stat().st_size)
        entries.append(f"[{kind}] {child.name}\t{size}")
    return "\n".join(entries) if entries else "<empty>"


@mcp.tool()
def directory_tree(path: str = WORKSPACE_ALIAS) -> str:
    """Return a JSON directory tree for the allowed workspace path."""
    root_path = _resolve_workspace_path(path)
    if not root_path.is_dir():
        raise NotADirectoryError(path)

    def build_tree(current: Path) -> dict[str, Any]:
        node: dict[str, Any] = {
            "name": current.name or WORKSPACE_ALIAS,
            "path": _display_path(current),
            "type": "directory" if current.is_dir() else "file",
        }
        if current.is_dir():
            node["children"] = [
                build_tree(child)
                for child in sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            ]
        else:
            node["size"] = current.stat().st_size
        return node

    return json.dumps(build_tree(root_path), indent=2)


@mcp.tool()
def move_file(source: str, destination: str) -> str:
    """Move or rename a file or directory within the allowed workspace."""
    source_path = _resolve_workspace_path(source)
    destination_path = _resolve_workspace_path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(destination_path))
    return f"Moved {_display_path(source_path)} to {_display_path(destination_path)}"


@mcp.tool()
def search_files(path: str, pattern: str) -> str:
    """Search file and directory names under a workspace path."""
    root_path = _resolve_workspace_path(path)
    if not root_path.is_dir():
        raise NotADirectoryError(path)

    pattern_lower = pattern.lower()
    matches = []
    for candidate in root_path.rglob("*"):
        name = candidate.name
        if fnmatch.fnmatch(name, pattern) or pattern_lower in name.lower():
            matches.append(_display_path(candidate))
    return "\n".join(matches) if matches else "<no matches>"


@mcp.tool()
def get_file_info(path: str) -> str:
    """Return JSON metadata for a file or directory in the allowed workspace."""
    return json.dumps(_file_info(_resolve_workspace_path(path)), indent=2)


@mcp.tool()
def list_allowed_directories() -> str:
    """List directories this MCP server allows access to."""
    return WORKSPACE_ALIAS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MT-AgentRisk filesystem MCP server.")
    parser.add_argument("--host", default=os.environ.get("MCP_FILESYSTEM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_FILESYSTEM_PORT", DEFAULT_PORT)))
    parser.add_argument(
        "--workspace",
        default=os.environ.get("MTAGENTRISK_SHARED_WORKSPACE", str(Path.cwd() / "outputs" / "shared_workspace")),
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["MTAGENTRISK_SHARED_WORKSPACE"] = str(workspace)

    print(
        f"Starting filesystem MCP on http://{args.host}:{args.port}/sse "
        f"with {WORKSPACE_ALIAS} -> {workspace}",
        flush=True,
    )
    mcp.run(transport="sse", host=args.host, port=args.port, path="/sse")


if __name__ == "__main__":
    main()

# Vendored source snapshots

This directory makes the artifact source-complete without embedding nested Git
repositories. Every subtree was exported from the exact commit recorded in
`versions.json`; no `.git` directories are included.

## Included scope

- `toolshield/`: `agentrisk/`, `toolshield/`, Docker files, package metadata,
  README, and license.
- `memgen/`: model/trainer source, interactions, data builders, configs,
  launch scripts, README, and requirements.
- `openhands/`: the Python runtime package, microagents, package/lock metadata,
  configuration template, README, and license.
- `mcpmark/`: Python source, task/service definitions, package/lock metadata,
  Docker and run scripts, README, and license.

## Local compatibility patch

OpenHands 0.54.0 is shipped with the MT-AgentRisk-compatible MCP client at
`openhands/openhands/mcp/client.py`. Its source is identical to
`../../src/redir/eval/mtagentrisk/client.py`.

## Installation

The complete ReDiR implementation lives under `src/redir/`.
Install the vendored runtime packages only when running agentic evaluation:

```bash
SKIP_VSCODE_BUILD=1 python -m pip install -e third_party/openhands
python -m pip install -e third_party/mcpmark
```

ToolShield and MemGen are included only for provenance and source inspection.
The public pipeline imports the localized ReDiR implementation and the pinned
OpenHands/MCPMark runtime.

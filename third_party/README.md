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
`../../src/latent_safety/eval/mtagentrisk/client.py`.

## Installation

The core ReDiR package uses the public interface and engine under `src/redir/`
plus model/protocol components under `src/latent_safety/`.
Install the vendored runtime packages only when running agentic evaluation:

```bash
SKIP_VSCODE_BUILD=1 python -m pip install -e third_party/openhands
python -m pip install -e third_party/mcpmark
```

ToolShield and MemGen are included for provenance and source inspection; the
ReDiR package contains the localized/adapted modules used by the public entry
points.

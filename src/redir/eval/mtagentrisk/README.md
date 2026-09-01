# MT-AgentRisk integration

This directory contains the filesystem evaluation adapter used by
`scripts/collect.sh` and `scripts/test.sh`. Both public scripts start the
required local services and pass the local ReDiR endpoint configuration to
`run_eval.py`; no external judge API is required by the public pipeline.

Use the top-level `README.md` for installation and end-to-end commands.

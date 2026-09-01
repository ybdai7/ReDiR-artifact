"""MT-AgentRisk adapters for the MAGE guard baseline.

The code in this package is intentionally independent from the latent-policy
training path.  It converts persisted OpenHands events into the action-review
contract used by the upstream MAGE release.
"""

from .openhands_adapter import (  # noqa: F401
    BYPASS_ACTIONS,
    GUARDED_ACTIONS,
    is_guarded_action,
    normalize_pending_tool_calls,
)

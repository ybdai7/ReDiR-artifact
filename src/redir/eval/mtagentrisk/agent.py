"""OpenHands agent variants for MT-AgentRisk evaluation."""

import os
from pathlib import Path
from typing import Any

import openhands.agenthub.codeact_agent.codeact_agent as codeact_module
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.events.action import AgentFinishAction
from openhands.events.serialization import event_to_dict
from openhands.utils.prompt import PromptManager

from redir.baselines.mage.live_guard import MageLiveGuardClient, append_decision_log
from redir.baselines.mage.openhands_adapter import (
    normalize_context_events,
    normalize_pending_tool_calls,
)


class MTAgentRiskCodeActAgent(CodeActAgent):
    """CodeActAgent with MT-AgentRisk-specific system prompt templates."""

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            local_prompt_dir = Path(__file__).resolve().parent / "openhands_prompts"
            base_prompt_dir = Path(codeact_module.__file__).resolve().parent / "prompts"
            self._prompt_manager = PromptManager(
                prompt_dir=[str(local_prompt_dir), str(base_prompt_dir)],
                system_prompt_filename=self.config.resolved_system_prompt_filename,
            )

        return self._prompt_manager


def _serialized_action_event(action: Any) -> dict[str, Any]:
    event = event_to_dict(action)
    event["source"] = "agent"
    raw_action = event.get("action")
    event["action"] = getattr(raw_action, "value", raw_action)
    return event


def _serialized_state_events(state: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in state.history:
        row = event_to_dict(event)
        raw_action = row.get("action")
        row["action"] = getattr(raw_action, "value", raw_action)
        events.append(row)
    return events


class MageGuardedMTAgentRiskCodeActAgent(MTAgentRiskCodeActAgent):
    """Review each external action with the trained MAGE guard before execution."""

    def step(self, state):
        action = super().step(state)
        action_event = _serialized_action_event(action)
        pending_tool_calls = normalize_pending_tool_calls(action_event)
        if not pending_tool_calls:
            return action

        guard = getattr(self, "_mtagentrisk_mage_guard", None)
        if guard is None:
            guard = MageLiveGuardClient()
            self._mtagentrisk_mage_guard = guard

        context = normalize_context_events(_serialized_state_events(state))
        decision = guard.review(context=context, pending_tool_calls=pending_tool_calls)
        log_path = os.environ.get("MTAGENTRISK_MAGE_DECISIONS_PATH")
        if log_path:
            append_decision_log(
                log_path,
                {
                    "task_name": os.environ.get("MTAGENTRISK_MAGE_TASK_NAME", ""),
                    "iteration": getattr(state, "iteration", None),
                    "proposed_action": action_event,
                    "pending_tool_calls": pending_tool_calls,
                    **decision.as_dict(),
                },
            )

        if decision.decision == "APPROVE":
            return action

        self.pending_actions.clear()
        return AgentFinishAction(
            final_thought=(
                "MAGE guard blocked the proposed external action. "
                "I cannot continue with this request safely."
            )
        )

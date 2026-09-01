"""OpenHands agent integration for MT-AgentRisk evaluation."""

from pathlib import Path

import openhands.agenthub.codeact_agent.codeact_agent as codeact_module
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.utils.prompt import PromptManager


class MTAgentRiskCodeActAgent(CodeActAgent):
    """CodeActAgent with the MT-AgentRisk system prompt templates."""

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


__all__ = ["MTAgentRiskCodeActAgent"]

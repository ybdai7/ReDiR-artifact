import json
import logging
import math
import os
import random
from typing import Union

from peft import PeftModel
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForImageTextToText,
    AutoTokenizer,
    GenerationConfig,
    DynamicCache
)
from transformers.modeling_utils import PreTrainedModel

from redir.models.latent_memory.configuration import MemGenConfig
from redir.models.latent_memory.modeling_utils import (
    MemGenOutputWithPast,
    MemGenLoraSwitchMixin,
    MemGenGenerationMixin,
)
from redir.models.latent_memory.trigger import MemGenTrigger
from redir.models.latent_memory.weaver import MemGenWeaver
from redir.models.latent_memory.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
    log_trainable_params
)
from redir.models.latent_memory.native_boundary import BOUNDARY_INJECTION_MODE
from redir.models.native_model_compat import (
    native_per_layer_inputs_for_embeds,
    native_model_profile,
    text_hidden_size,
)

class MemGenModel(PreTrainedModel, MemGenLoraSwitchMixin, MemGenGenerationMixin):
    config_class = MemGenConfig
    INSTRUCTION_STATE = 0
    CONVERSATION_STATE = 1

    def __init__(
        self, 
        config: MemGenConfig, 
        base_tokenizer,
        reasoner_base_model: PreTrainedModel, 
        weaver_base_model: PreTrainedModel,
        trigger_base_model: PreTrainedModel | None,
        *,
        initialize_latents: bool = True,
    ):   
        super().__init__(config)
        
        self.config = config

        # insert lora adapters into weaver and trigger
        weaver_model_w_lora, trigger_model_w_lora = self._insert_lora_adapters(
            weaver_base_model, config.weaver_lora_config, trigger_base_model, config.trigger_lora_config,
        )

        # use base model with lora adapters to initiate weaver and trigger
        self.weaver = MemGenWeaver(weaver_model_w_lora, config.prompt_latents_len, config.inference_latents_len)
        self.trigger = MemGenTrigger(trigger_model_w_lora, config.trigger_active)

        # base reasoner
        self.reasoner = reasoner_base_model
        self.tokenizer = base_tokenizer
        
        # Projection layers for mapping embeddings between reasoner and weaver.
        reasoner_hidden_size = text_hidden_size(reasoner_base_model.config)
        weaver_hidden_size = text_hidden_size(weaver_base_model.config)
        self.latent_projection_mode = self._latent_projection_mode(config)
        self.reasoner_to_weaver, self.weaver_to_reasoner = self._build_latent_projections(
            config,
            reasoner_hidden_size,
            weaver_hidden_size,
        )
        logging.info(
            "Latent projection mode=%s reasoner_hidden=%d weaver_hidden=%d parameters=%d",
            self.latent_projection_mode,
            reasoner_hidden_size,
            weaver_hidden_size,
            sum(
                parameter.numel()
                for module in (self.reasoner_to_weaver, self.weaver_to_reasoner)
                for parameter in module.parameters()
            ),
        )
        
        # delimiters for detecting augmentation points
        self.delimiters: list[str] = [",", ".", "\n"]  

        self.state = None
        self._reasoner_token_norm_p50: float | None = None
        if self._latent_output_norm_enabled():
            self._reasoner_token_norm_p50 = self._compute_reasoner_token_norm_p50()
            logging.info(
                "Latent output normalization enabled: token_norm_p50=%.6f target_scale=%.6f",
                self._reasoner_token_norm_p50,
                float((getattr(self.config, "latent_output_norm", None) or {}).get("target_scale", 1.0)),
            )

        # A saved checkpoint replaces every latent/query parameter immediately
        # in ``from_pretrained``.  Skipping the temporary initializer there is
        # also required when reasoner and Weaver hidden sizes differ: the
        # token-set initializer is expressed in reasoner embedding space,
        # whereas the saved Weaver queries already live in Weaver space.
        if initialize_latents:
            self._initialize_latents_from_token_sets()

        # postprocess
        self._postprocess_models()
        logging.info("##### MemGen Initialization #####")
        if getattr(config, "log_trainable_params", False):
            log_trainable_params(self)

    def _postprocess_models(self):
        # fix base model parameters
        fix_model_parameters(self.reasoner)  

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            logging.info(
                f"Tokenizer has no pad token. Using EOS token ({self.tokenizer.eos_token}) as pad token."
            )

        # Legacy Qwen checkpoints were trained with the project template.  A
        # non-Qwen native family must retain the tokenizer-shipped template:
        # it defines that model's assistant boundary and tool-call grammar.
        if native_model_profile(getattr(self.config, "model_family", "qwen")) is None:
            self.tokenizer.chat_template = CONVERSATION_TEMPLATE
        elif not isinstance(self.tokenizer.chat_template, str) or not self.tokenizer.chat_template.strip():
            raise ValueError("native model tokenizer does not expose a chat template")

    def _initialize_latents_from_token_sets(self):
        latent_init = getattr(self.config, "latent_init", None) or {}
        if latent_init.get("mode", "random") != "token_set_average":
            return

        hidden_size = self.reasoner.get_input_embeddings().weight.shape[1]
        default_groups = [
            ["safety", "safe", "secure", "careful", "protect", "protection", "responsible", "trustworthy"],
            ["unsafe", "harmful", "dangerous", "malicious", "exploit", "abuse", "risk", "risky"],
            ["refuse", "decline", "cannot", "avoid", "disallow", "deny", "stop", "withhold"],
            ["privacy", "credential", "secret", "password", "token", "key", "sensitive", "confidential"],
            ["authorization", "permission", "consent", "access", "account", "identity", "owner", "policy"],
            ["helpful", "benign", "legitimate", "allowed", "assist", "useful", "constructive", "appropriate"],
            ["analyze", "assess", "verify", "consider", "inspect", "reason", "evaluate", "determine"],
            ["prevent", "protect", "mitigate", "comply safely", "safe alternative", "secure handling", "boundary", "caution"],
        ]
        phrase_groups = latent_init.get("phrase_groups") or default_groups
        if not phrase_groups:
            raise ValueError("model.latent_init.phrase_groups must not be empty")

        seed = int(latent_init.get("seed", 42))
        noise_ratio = float(latent_init.get("noise_ratio", 0.02))
        scale_mode = str(latent_init.get("scale_mode", "token_norm_p50_over_sqrt_hidden"))
        apply_prompt = bool(latent_init.get("apply_to_prompt_latents", True))
        apply_inference = bool(latent_init.get("apply_to_inference_latents", True))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        embedding = self.reasoner.get_input_embeddings().weight.detach().float().cpu()
        token_norm_p50 = torch.quantile(embedding.norm(dim=1), torch.tensor(0.5)).item()
        if scale_mode == "token_norm_p50_over_sqrt_hidden":
            latent_scale = token_norm_p50 / math.sqrt(hidden_size)
        elif scale_mode == "one":
            latent_scale = 1.0
        else:
            try:
                latent_scale = float(scale_mode)
            except ValueError as exc:
                raise ValueError(f"Unsupported latent_init scale_mode: {scale_mode}") from exc

        def build_latents(num_latents: int) -> torch.Tensor:
            rows = []
            for idx in range(num_latents):
                phrases = phrase_groups[idx % len(phrase_groups)]
                if isinstance(phrases, str):
                    phrases = [phrases]
                token_ids = []
                for phrase in phrases:
                    ids = self.tokenizer.encode(str(phrase), add_special_tokens=False)
                    token_ids.extend(ids)
                token_ids = [tok for tok in token_ids if 0 <= tok < embedding.size(0)]
                if not token_ids:
                    raise ValueError(f"latent_init phrase group {idx} produced no token ids")
                vec = embedding[torch.tensor(token_ids, dtype=torch.long)].mean(dim=0)
                if noise_ratio > 0:
                    noise = torch.randn(vec.shape, generator=generator, dtype=vec.dtype)
                    noise_norm = noise.norm().clamp_min(1e-12)
                    vec_norm = vec.norm().clamp_min(1e-12)
                    vec = vec + noise / noise_norm * vec_norm * noise_ratio
                rows.append(vec)
            return torch.stack(rows, dim=0)

        with torch.no_grad():
            if apply_prompt and self.weaver.prompt_latents_num > 0:
                prompt_latents = build_latents(self.weaver.prompt_latents_num)
                self.weaver.prompt_query_latents.data.copy_(
                    prompt_latents.to(
                        device=self.weaver.prompt_query_latents.device,
                        dtype=self.weaver.prompt_query_latents.dtype,
                    )
                )
                self.weaver.prompt_latent_scale.data.fill_(latent_scale)
            if apply_inference and self.weaver.inference_latents_num > 0:
                inference_latents = build_latents(self.weaver.inference_latents_num)
                self.weaver.inference_query_latents.data.copy_(
                    inference_latents.to(
                        device=self.weaver.inference_query_latents.device,
                        dtype=self.weaver.inference_query_latents.dtype,
                    )
                )
                self.weaver.inference_latent_scale.data.fill_(latent_scale)

        logging.info(
            "Initialized latent query slots with token_set_average: token_norm_p50=%.6f latent_scale=%.6f noise_ratio=%.4f",
            token_norm_p50,
            latent_scale,
            noise_ratio,
        )

    def _latent_output_norm_enabled(self) -> bool:
        latent_output_norm = getattr(self.config, "latent_output_norm", None) or {}
        return bool(latent_output_norm.get("enabled", False))

    def _compute_reasoner_token_norm_p50(self) -> float:
        with torch.no_grad():
            embedding = self.reasoner.get_input_embeddings().weight.detach()
            token_norms = embedding.float().norm(dim=1)
            return float(
                torch.quantile(
                    token_norms,
                    torch.tensor(0.5, device=token_norms.device, dtype=token_norms.dtype),
                ).detach().cpu()
            )

    def _normalize_inserted_latents(self, latent_inputs_embeds: torch.Tensor) -> torch.Tensor:
        latent_output_norm = getattr(self.config, "latent_output_norm", None) or {}
        if not bool(latent_output_norm.get("enabled", False)):
            return latent_inputs_embeds

        target_mode = str(latent_output_norm.get("target_mode", "token_norm_p50"))
        if target_mode != "token_norm_p50":
            try:
                base_target_norm = float(target_mode)
            except ValueError as exc:
                raise ValueError(f"Unsupported latent_output_norm target_mode: {target_mode}") from exc
        else:
            if self._reasoner_token_norm_p50 is None:
                self._reasoner_token_norm_p50 = self._compute_reasoner_token_norm_p50()
            base_target_norm = self._reasoner_token_norm_p50

        target_scale = float(latent_output_norm.get("target_scale", 1.0))
        eps = float(latent_output_norm.get("eps", 1e-6))
        target_norm = base_target_norm * target_scale
        latent_float = latent_inputs_embeds.float()
        current_norm = latent_float.norm(dim=-1, keepdim=True).clamp_min(eps)
        normalized = latent_float / current_norm * target_norm
        return normalized.to(dtype=latent_inputs_embeds.dtype)

    @staticmethod
    def _latent_projection_mode(config: MemGenConfig) -> str:
        projection_config = getattr(config, "latent_projection", None) or {}
        mode = str(projection_config.get("mode", "linear")).strip().lower()
        if mode not in {"linear", "identity"}:
            raise ValueError(f"Unsupported latent projection mode: {mode}")
        return mode

    @staticmethod
    def _saved_latent_projection_mode(load_directory: str) -> str:
        saved_config_path = os.path.join(load_directory, "config.json")
        if not os.path.exists(saved_config_path):
            return "linear"
        with open(saved_config_path, encoding="utf-8") as config_file:
            saved_config = json.load(config_file)
        return str(
            (saved_config.get("latent_projection") or {}).get("mode", "linear")
        ).strip().lower()

    @classmethod
    def _build_latent_projections(
        cls,
        config: MemGenConfig,
        reasoner_hidden_size: int,
        weaver_hidden_size: int,
    ) -> tuple[nn.Module, nn.Module]:
        mode = cls._latent_projection_mode(config)
        if mode == "identity":
            if reasoner_hidden_size != weaver_hidden_size:
                raise ValueError(
                    "Identity latent projection requires matching hidden sizes: "
                    f"reasoner={reasoner_hidden_size}, weaver={weaver_hidden_size}"
                )
            return nn.Identity(), nn.Identity()
        return (
            nn.Linear(reasoner_hidden_size, weaver_hidden_size),
            nn.Linear(weaver_hidden_size, reasoner_hidden_size),
        )

    def _project_reasoner_to_weaver(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        weaver_embedding = self.weaver.model.get_input_embeddings().weight
        if self.latent_projection_mode == "linear":
            projection_weight = self.reasoner_to_weaver.weight
            inputs_embeds = inputs_embeds.to(
                device=projection_weight.device,
                dtype=projection_weight.dtype,
            )
            projected = self.reasoner_to_weaver(inputs_embeds)
        else:
            projected = self.reasoner_to_weaver(
                inputs_embeds.to(
                    device=weaver_embedding.device,
                    dtype=weaver_embedding.dtype,
                )
            )
        return projected.to(device=weaver_embedding.device, dtype=weaver_embedding.dtype)

    def _project_weaver_to_reasoner(self, hidden_states: torch.Tensor) -> torch.Tensor:
        reasoner_embedding = self.reasoner.get_input_embeddings().weight
        if self.latent_projection_mode == "linear":
            projection_weight = self.weaver_to_reasoner.weight
            hidden_states = hidden_states.to(
                device=projection_weight.device,
                dtype=projection_weight.dtype,
            )
            projected = self.weaver_to_reasoner(hidden_states)
        else:
            projected = self.weaver_to_reasoner(
                hidden_states.to(
                    device=reasoner_embedding.device,
                    dtype=reasoner_embedding.dtype,
                )
            )
        return projected.to(device=reasoner_embedding.device, dtype=reasoner_embedding.dtype)
    

    @property
    def device(self):
        return self.reasoner.device

    @property
    def weaver_device(self):
        return self.weaver.device

    def _aligned_prompt_token_inputs(
        self,
        input_ids: torch.Tensor,
        latent_insertion_index: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align real token ids with the prompt sequence after latent insertion."""

        latent_count = self.weaver.prompt_latents_num
        placeholders = torch.zeros(
            (input_ids.size(0), latent_count),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        latent_mask = torch.zeros_like(placeholders, dtype=torch.bool)
        token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if self.config.latent_injection_mode == BOUNDARY_INJECTION_MODE:
            if latent_insertion_index is None:
                raise ValueError("latent insertion index is required for token alignment")
            index = int(latent_insertion_index)
            aligned_ids = torch.cat(
                [input_ids[:, :index], placeholders, input_ids[:, index:]], dim=1
            )
            aligned_mask = torch.cat(
                [token_mask[:, :index], latent_mask, token_mask[:, index:]], dim=1
            )
        else:
            aligned_ids = torch.cat([input_ids, placeholders], dim=1)
            aligned_mask = torch.cat([token_mask, latent_mask], dim=1)
        return aligned_ids, aligned_mask

    @staticmethod
    def _native_embed_forward_kwargs(
        model: PreTrainedModel,
        inputs_embeds: torch.Tensor,
        token_input_ids: torch.Tensor,
        token_positions_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        per_layer_inputs = native_per_layer_inputs_for_embeds(
            model,
            inputs_embeds,
            token_input_ids=token_input_ids,
            token_positions_mask=token_positions_mask,
        )
        return (
            {"per_layer_inputs": per_layer_inputs}
            if per_layer_inputs is not None
            else {}
        )

    def place_components(
        self,
        reasoner_device: str | torch.device,
        weaver_device: str | torch.device | None = None,
        trigger_device: str | torch.device | None = None,
    ):
        reasoner_device = torch.device(reasoner_device)
        weaver_device = torch.device(weaver_device) if weaver_device is not None else reasoner_device
        trigger_device = torch.device(trigger_device) if trigger_device is not None else weaver_device

        self.reasoner.to(reasoner_device)
        self.weaver.to(weaver_device)
        self.reasoner_to_weaver.to(weaver_device)
        self.weaver_to_reasoner.to(reasoner_device)
        self.trigger.to(trigger_device)
        return self

    def prepare_inputs_with_prompt_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_insertion_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Insert the safety latent immediately before assistant generation.

        This helper bypasses the trigger and gives MT-AgentRisk OPD training a fixed
        before-assistant-action injection point.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        reasoner_device = self.device
        weaver_device = self.weaver_device
        inputs_embeds = self.reasoner.get_input_embeddings()(input_ids)
        boundary_mode = self.config.latent_injection_mode == BOUNDARY_INJECTION_MODE
        if boundary_mode:
            if latent_insertion_index is None:
                raise ValueError(
                    "before_assistant_generation_boundary requires an explicit "
                    "latent_insertion_index from token-exact chat-template rendering"
                )
            insertion_index = int(latent_insertion_index)
            allow_empty_suffix = bool(
                getattr(self.config, "allow_empty_generation_suffix", False)
            )
            valid_boundary = 0 < insertion_index < input_ids.size(1) or (
                allow_empty_suffix and insertion_index == input_ids.size(1)
            )
            if not valid_boundary:
                raise ValueError(
                    "latent_insertion_index must separate history and native suffix: "
                    f"{insertion_index}/{input_ids.size(1)}"
                )
            history_embeds = inputs_embeds[:, :insertion_index]
            history_input_ids = input_ids[:, :insertion_index]
            suffix_embeds = inputs_embeds[:, insertion_index:]
            history_attention = attention_mask[:, :insertion_index]
        else:
            history_embeds = inputs_embeds
            history_input_ids = input_ids
            suffix_embeds = inputs_embeds[:, 0:0]
            history_attention = attention_mask

        if self.weaver.prompt_latents_num == 0:
            position_ids = self._generate_position_ids(attention_mask)
            return inputs_embeds, attention_mask, position_ids

        position_ids = self._generate_position_ids(history_attention)
        weaver_attention_mask = history_attention.to(weaver_device)
        weaver_position_ids = position_ids.to(weaver_device)
        weaver_inputs_embeds = self._project_reasoner_to_weaver(history_embeds)
        weaver_hidden_states, latent_mask, _ = self.weaver.augment_prompt(
            weaver_inputs_embeds,
            weaver_attention_mask,
            weaver_position_ids,
            token_input_ids=history_input_ids.to(weaver_device),
        )
        latent_inputs_embeds = self._project_weaver_to_reasoner(weaver_hidden_states)
        latent_inputs_embeds = self._normalize_inserted_latents(latent_inputs_embeds)
        latent_mask = latent_mask.to(reasoner_device)
        if boundary_mode:
            suffix_attention = attention_mask[:, insertion_index:]
            inputs_embeds = torch.cat(
                [history_embeds, latent_inputs_embeds, suffix_embeds], dim=1
            )
            attention_mask = torch.cat(
                [history_attention, latent_mask, suffix_attention], dim=1
            )
        else:
            inputs_embeds = torch.cat([inputs_embeds, latent_inputs_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, latent_mask], dim=1)
        position_ids = self._generate_position_ids(attention_mask)
        return inputs_embeds, attention_mask, position_ids

    def forward_with_prompt_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_insertion_index: int | None = None,
    ):
        inputs_embeds, attention_mask, position_ids = self.prepare_inputs_with_prompt_latent(
            input_ids,
            attention_mask,
            latent_insertion_index,
        )
        aligned_ids, token_mask = self._aligned_prompt_token_inputs(
            input_ids.to(self.device), latent_insertion_index
        )
        return self.reasoner(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **self._native_embed_forward_kwargs(
                self.reasoner, inputs_embeds, aligned_ids, token_mask
            ),
        )

    def forward_prompt_completion_with_prompt_latent(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_input_ids: torch.Tensor,
        latent_insertion_index: int | None = None,
    ):
        prompt_embeds, attention_mask, _ = self.prepare_inputs_with_prompt_latent(
            prompt_input_ids,
            prompt_attention_mask,
            latent_insertion_index,
        )
        completion_input_ids = completion_input_ids.to(self.device)
        completion_embeds = self.reasoner.get_input_embeddings()(completion_input_ids)
        completion_mask = torch.ones(
            completion_input_ids.shape,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        inputs_embeds = torch.cat([prompt_embeds, completion_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, completion_mask], dim=1)
        position_ids = self._generate_position_ids(attention_mask)
        aligned_prompt_ids, prompt_token_mask = self._aligned_prompt_token_inputs(
            prompt_input_ids.to(self.device), latent_insertion_index
        )
        aligned_ids = torch.cat([aligned_prompt_ids, completion_input_ids], dim=1)
        token_mask = torch.cat(
            [prompt_token_mask, torch.ones_like(completion_input_ids, dtype=torch.bool)],
            dim=1,
        )
        return self.reasoner(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **self._native_embed_forward_kwargs(
                self.reasoner, inputs_embeds, aligned_ids, token_mask
            ),
        )

    @torch.no_grad()
    def generate_with_prompt_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig,
        latent_insertion_index: int | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        inputs_embeds, attention_mask, _ = self.prepare_inputs_with_prompt_latent(
            input_ids,
            attention_mask,
            latent_insertion_index,
        )
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_aligned_ids, current_token_mask = self._aligned_prompt_token_inputs(
            input_ids, latent_insertion_index
        )
        current_cache = None
        generated_ids = []
        max_new_tokens = generation_config.max_new_tokens
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_ids = torch.tensor(eos_token_id, device=input_ids.device)
        elif eos_token_id is None:
            eos_token_ids = None
        else:
            eos_token_ids = torch.tensor([eos_token_id], device=input_ids.device)
        for _ in range(max_new_tokens):
            if current_cache is not None:
                reasoner_inputs_embeds = current_inputs_embeds[:, -1:]
                reasoner_position_ids = current_position_ids[:, -1:]
                reasoner_token_ids = current_aligned_ids[:, -1:]
                reasoner_token_mask = current_token_mask[:, -1:]
            else:
                reasoner_inputs_embeds = current_inputs_embeds
                reasoner_position_ids = current_position_ids
                reasoner_token_ids = current_aligned_ids
                reasoner_token_mask = current_token_mask
            outputs = self.reasoner(
                inputs_embeds=reasoner_inputs_embeds,
                attention_mask=current_attention_mask,
                position_ids=reasoner_position_ids,
                use_cache=True,
                past_key_values=current_cache,
                **self._native_embed_forward_kwargs(
                    self.reasoner,
                    reasoner_inputs_embeds,
                    reasoner_token_ids,
                    reasoner_token_mask,
                ),
            )
            next_token_ids = self._get_next_token(
                outputs.logits[:, -1],
                do_sample=bool(generation_config.do_sample),
                temperature=float(generation_config.temperature or 0.0),
            )
            generated_ids.append(next_token_ids)
            next_embeds = self.reasoner.get_input_embeddings()(next_token_ids)
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_embeds], dim=1)
            current_aligned_ids = torch.cat(
                [current_aligned_ids, next_token_ids], dim=1
            )
            current_token_mask = torch.cat(
                [
                    current_token_mask,
                    torch.ones_like(next_token_ids, dtype=torch.bool),
                ],
                dim=1,
            )
            next_mask = torch.ones(
                (current_attention_mask.size(0), 1),
                dtype=current_attention_mask.dtype,
                device=current_attention_mask.device,
            )
            current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=1)
            next_position_id = current_position_ids[:, -1:] + 1
            current_position_ids = torch.cat([current_position_ids, next_position_id], dim=1)
            current_cache = outputs.past_key_values
            if eos_token_ids is not None and (next_token_ids == eos_token_ids).any(dim=1).all():
                break
            del outputs

        if generated_ids:
            completion_ids = torch.cat(generated_ids, dim=1)
        else:
            completion_ids = torch.empty(
                (input_ids.size(0), 0),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        return torch.cat([input_ids.to(completion_ids.device), completion_ids], dim=1)
    
    def _forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> torch.Tensor:
        # preprocess inputs
        assert input_ids.shape == attention_mask.shape == labels.shape
        
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        delimiters = self.delimiters
        max_augment_num = self.config.max_inference_aug_num  # Limit the number of inference augmentation points to avoid excessive augmentation
        device = self.device
        embeds_dtype = reasoner.get_input_embeddings().weight.dtype
        B, _ = input_ids.shape
        hidden_size = self.config.hidden_size

        # select augment idx
        augmentation_indices = self._select_augment_points_after_delimiter(
            input_ids, labels, delimiters, tokenizer, max_augment_num
        )
        
        # origin inputs embeds
        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
                
        # Initialize the start index and empty tensors for accumulating processed segments
        current_start_idx = 0
        current_inputs_embeds = torch.empty((B, 0, hidden_size), device=device, dtype=embeds_dtype)
        current_attention_mask = torch.empty((B, 0), device=device, dtype=attention_mask.dtype)
        current_latents_mask = torch.empty((B, 0), device=device, dtype=torch.bool)

        # Iterate over the selected augmentation points
        for aug_point_idx in augmentation_indices:
            # Slice the current segment of original embeddings and attention mask
            segment_inputs_embeds = inputs_embeds[:, current_start_idx:aug_point_idx]
            segment_attention_mask = attention_mask[:, current_start_idx:aug_point_idx]
            segment_latents_mask = torch.zeros((B, segment_inputs_embeds.size(1)), device=device, dtype=torch.bool)

            # Concatenate the current segment to the accumulated embeddings and masks
            current_inputs_embeds = torch.cat([current_inputs_embeds, segment_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, segment_attention_mask], dim=1)
            current_position_ids = self._generate_position_ids(current_attention_mask)
            current_latents_mask = torch.cat([current_latents_mask, segment_latents_mask], dim=1)

            # Map reasoner embeddings to weaver embeddings for augmentation
            weaver_inputs_embeds = self._project_reasoner_to_weaver(current_inputs_embeds)

            # Determine whether this point is the end of the prompt (prompt augmentation)
            is_prompt_end_aug = (labels[:, aug_point_idx] != -100).all() and (labels[:, aug_point_idx-1] == -100).all().item()
            
            # Depending on type, use weaver to augment prompt or inference
            if is_prompt_end_aug:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                )
            else:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                ) 

            # Map weaver hidden states back to reasoner embeddings
            latent_inputs_embeds = self._project_weaver_to_reasoner(weaver_hidden_states)
            latent_inputs_embeds = self._normalize_inserted_latents(latent_inputs_embeds)

            # Update accumulated embeddings and masks with the newly augmented segment
            current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
            current_start_idx = aug_point_idx
            
            # Update latent mask for the newly added latent embeddings
            latent_mask = torch.ones((B, latent_inputs_embeds.size(1)), device=device, dtype=torch.bool)
            current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)
            
        # Process the remaining segment after the last augmentation point
        remaining_inputs_embeds = inputs_embeds[:, current_start_idx:]
        remaining_attention_mask = attention_mask[:, current_start_idx:]
        latent_mask = torch.zeros((B, remaining_attention_mask.size(1)), device=device, dtype=torch.bool)
        
        current_inputs_embeds = torch.cat([current_inputs_embeds, remaining_inputs_embeds], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, remaining_attention_mask], dim=1)
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

        reasoner_outputs = reasoner(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids
        )
        logits = reasoner_outputs.logits
        
        # Identify valid positions in logits (positions that should contribute to loss)
        shifted = torch.zeros_like(current_latents_mask)
        shifted[:, :-1] = current_latents_mask[:, 1:]
        valid_mask = ~shifted
        
        valid_logits = logits[valid_mask].view(logits.size(0), -1, logits.size(2))  
        # assert shifted.sum() == current_latents_mask.sum()
        # assert valid_logits.shape[:2] == input_ids.shape
        return valid_logits
    
    def _instructional_forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Forward pass for single-turn instructional data (no multi-turn conversation required).

        This method is used for instruction-following tasks (SFT), where the input
        consists of a single instruction and the corresponding labels. It directly
        delegates to the single-turn forward method `_forward`.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask (torch.Tensor): Tensor indicating padding positions.
            labels (torch.Tensor): Tensor containing the target labels for supervised fine-tuning.
            **kwargs: Additional keyword arguments passed to `_forward`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - logits: The output logits from the model for each input token.
                - labels: The same as input labels, used for loss computation.
        """
        # raise RuntimeError()
        logits = self._forward(input_ids, attention_mask, labels, **kwargs)
        # For Instruction SFT, labels remain the same as input
        return logits, labels

    def _conversational_forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Forward pass for conversational (multi-turn) data.

        Multi-turn forward is constructed by sequentially calling the single-turn forward
        for each conversation turn. Latents inserted in turn i-1 are not visible to turn i.

        Args:
            input_ids (torch.Tensor): Input token IDs, shape (1, seq_len). Batch size must be 1.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor): Target labels for supervised fine-tuning (-100 for ignore positions).
            **kwargs: Additional arguments passed to `_forward`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - all_logits: Logits for the entire sequence, with zeros for unsupervised positions.
                - all_labels: Labels for the entire sequence, with -100 for unsupervised positions.
        """
        assert input_ids.shape[0] == 1, "Conversational SFT currently only supports batch_size = 1"
        seq_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size
        device = input_ids.device

        # Identify single-turn segments within the conversation based on labels
        label_row = labels[0]
        should_supervise = label_row != -100
        if not should_supervise.any():
            raise ValueError("At least one completion segment is required")

        # Compute the start and end indices of valid supervised segments
        valid_mask = should_supervise.int()
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), valid_mask]))
        valid_starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()  # Transition 0 -> 1
        ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()          # Transition 1 -> 0
        if len(ends) < len(valid_starts):
            ends.append(seq_len)  # 自动补充最后一个 token 的 (index + 1) 作为最后一个序列的末尾
        assert len(valid_starts) == len(ends)
        
        # Build triplets (start of previous segment, start of supervised segment, end of supervised segment)
        triplets = []
        start = 0
        for s, e in zip(valid_starts, ends):
            triplets.append((start, s, e))
            start = e
        
        # If there are more segments than allowed, randomly select self.max_prompt_aug_num segments
        if len(triplets) <= self.config.max_prompt_aug_num:
            select_turns = [1] * len(triplets)
        else:
            triplets_num = len(triplets)
            selected_indices = set(random.sample(range(triplets_num), self.config.max_prompt_aug_num))
            select_turns = [1 if i in selected_indices else 0 for i in range(triplets_num)]

        # Initialize tensors to store logits and labels for the entire sequence
        all_logits = torch.zeros(1, seq_len, vocab_size, device=device)
        all_labels = torch.full((1, seq_len), -100, device=device)

        # Loop over each conversation turn and perform single-turn forward if supervised
        for triplet, should_supervise in zip(triplets, select_turns):
            start, valid_start, end = triplet
            if should_supervise:
                cur_input_ids = input_ids[0, :end].unsqueeze(0)
                cur_attention = attention_mask[0, :end].unsqueeze(0)
                # cur_labels only used for _forward, does not represent the true supervision range
                # cur_labels = labels[0, :end].clone().unsqueeze(0)
                # cur_labels[0, :valid_start] = -100  # Mask tokens before supervision start
                cur_labels = torch.full((1, end), -100, device=device)
                cur_labels[0, valid_start:end] = labels[0, valid_start:end]

                # Single-turn forward for the current conversation segment
                logits = self._forward(cur_input_ids, cur_attention, cur_labels, **kwargs)
                
                # Update overall logits and labels with the results of this segment
                all_logits[0, start:end, :] = logits[0, start:end, :]
                all_labels[0, start:end] = labels[0, start:end]

        # Return logits and labels:
        # - supervised positions retain computed logits and original labels
        # - unsupervised positions have logits = 0 and labels = -100
        return all_logits, all_labels

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> MemGenOutputWithPast:  
        tokenizer = self.tokenizer

        # Ensure labels are provided, required for training the reasoning processor
        assert labels is not None, "Reasoning Processor requires input labels for training"
        
        # Determine whether the input is single-turn (instruction) or multi-turn (conversation)
        labels = self._postprocess_assistant_labels(input_ids, labels, tokenizer)
       
        # Use only the first data sample of each dataset to determine the model state
        if self.state is None:  
            self.state = MemGenModel.CONVERSATION_STATE if self._is_conversation(input_ids, tokenizer) else MemGenModel.INSTRUCTION_STATE

        if self.state == MemGenModel.INSTRUCTION_STATE:
            forward_func = self._instructional_forward
        elif self.state == MemGenModel.CONVERSATION_STATE:
            forward_func = self._conversational_forward
        else:
            raise RuntimeError(f"Unexpected model state: {self.state}")
        
        batch_size = 1  # Currently process one sequence per batch
        iter_num = input_ids.size(0) // batch_size

        # Forward pass per batch
        logits, supervised_labels = [], []
        for i in range(iter_num):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]

            # Call the appropriate forward function (instruction or conversation)
            batch_logits, batch_supervised_labels = forward_func(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                **kwargs
            )
            logits.append(batch_logits)
            supervised_labels.append(batch_supervised_labels)
        
        # Concatenate results from all batches
        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(supervised_labels, dim=0)

        # Compute causal language modeling loss (shifted by one)
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = all_labels[..., 1:].contiguous()
        # assert shift_logits.shape[:-1] == shift_labels.shape
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Return model outputs
        outputs = MemGenOutputWithPast(loss=loss, logits=all_logits)
        outputs.supervised_labels = all_labels  # Positions in input_ids that are supervised
        return outputs

    # @torch.no_grad()
    # def generate(
    #     self, 
    #     input_ids: torch.Tensor, 
    #     attention_mask: torch.Tensor,
    #     generation_config: GenerationConfig = None, 
    #     return_augmentation_mask: bool = False,
    #     **kwargs
    # ) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor]]: 
        
    #     tokenizer = self.tokenizer
    #     reasoner = self.reasoner
    #     weaver = self.weaver
    #     max_augment_num = self.config.max_inference_aug_num
    #     invalid_token_id = -100

    #     # preproecess inputs
    #     input_ids = input_ids.to(self.device)
    #     attention_mask = attention_mask.to(self.device)
    #     max_new_tokens = generation_config.max_new_tokens
    #     pad_token_id = tokenizer.pad_token_id
    #     eos_token_id = tokenizer.eos_token_id
    #     prompt_len = input_ids.size(1)

    #     inputs_embeds = reasoner.get_input_embeddings()(input_ids)
    #     B, _, hidden_size = inputs_embeds.shape
    #     device = inputs_embeds.device
        
    #     # --- generation loop ---
    #     current_inputs_embeds = inputs_embeds
    #     current_attention_mask = attention_mask
    #     current_position_ids = self._generate_position_ids(current_attention_mask)
    #     current_input_ids = input_ids
    #     current_cache: DynamicCache = None

    #     # Generation Loop Initialization
    #     sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)
        
    #     # NOTE - Whether to call the trigger and insert latent memory before generating the token at this position
    #     # - augmentation_pos[b][i] == -100: For the b-th sequence, no augmentation was sampled before generating the i-th token
    #     # - augmentation_pos[b][i] == 0: For the b-th sequence, augmentation was sampled before generating the i-th token, but the trigger decided NOT to insert latent memory
    #     # - augmentation_pos[b][i] == 1: For the b-th sequence, augmentation was sampled before generating the i-th token, and the trigger decided to insert latent memory
    #     augmentation_pos = torch.full((B, max_new_tokens), fill_value=invalid_token_id, device=device) 

    #     generation_config = GenerationConfig(
    #         do_sample=False,
    #         pad_token_id=pad_token_id,
    #         eos_token_id=eos_token_id,
    #         use_cache=False,
    #         max_new_tokens=max_new_tokens
    #     )
    #     # Perform generation for the remaining tokens using the reasoner
    #     generated = reasoner.generate(
    #         inputs_embeds=current_inputs_embeds,
    #         attention_mask=current_attention_mask,
    #         generation_config=generation_config
    #     )
    #     current_input_ids = torch.cat([current_input_ids, generated], dim=1)

    #     # postprocess
    #     new_generated_len = current_input_ids.size(1) - prompt_len
    #     augmentation_pos = augmentation_pos[:, :new_generated_len]
        
    #     self._check_generate(
    #         current_input_ids[:, prompt_len:],
    #         augmentation_pos
    #     )
        
    #     if return_augmentation_mask:
    #         return (current_input_ids, augmentation_pos)
    #     else:
    #         return current_input_ids

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig = None, 
        return_augmentation_mask: bool = False,
        **kwargs
    ) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor]]: 
        
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        max_augment_num = self.config.max_inference_aug_num
        invalid_token_id = -100

        # preproecess inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        max_new_tokens = generation_config.max_new_tokens
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        prompt_len = input_ids.size(1)

        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        B, _, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        
        # --- generation loop ---
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_input_ids = input_ids
        current_cache: DynamicCache = None

        # Generation Loop Initialization
        sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)
        
        # NOTE - Whether to call the trigger and insert latent memory before generating the token at this position
        # - augmentation_pos[b][i] == -100: For the b-th sequence, no augmentation was sampled before generating the i-th token
        # - augmentation_pos[b][i] == 0: For the b-th sequence, augmentation was sampled before generating the i-th token, but the trigger decided NOT to insert latent memory
        # - augmentation_pos[b][i] == 1: For the b-th sequence, augmentation was sampled before generating the i-th token, and the trigger decided to insert latent memory
        augmentation_pos = torch.full((B, max_new_tokens), fill_value=invalid_token_id, device=device) 

        for i in range(max_new_tokens):
            
            assert current_inputs_embeds.shape[:2] == current_attention_mask.shape == current_position_ids.shape
            augment_decision = self._should_augment(
                current_input_ids, 
                sentence_augment_count=sentence_augment_count, 
                do_sample=generation_config.trigger_do_sample,
                temperature=generation_config.temperature,
                is_prompt=(i==0)  
            )
            augmentation_pos[:, i] = augment_decision
            augment_indices = torch.where(augment_decision == 1)[0]

            # If there are sentences to augment, apply augmentation; others remain with left padding
            if len(augment_indices) > 0:
                # Increment the augmentation count for sentences that are being augmented
                if i != 0:  
                    sentence_augment_count[augment_indices] += 1

                # Select embeddings, attention masks, and position IDs for sentences to be augmented
                candidate_inputs_embeds = current_inputs_embeds[augment_indices]
                candidate_attention_mask = current_attention_mask[augment_indices]
                candidate_position_ids = current_position_ids[augment_indices]
                
                # Perform inference augmentation using the weaver
                weaver_inputs_embeds = self._project_reasoner_to_weaver(candidate_inputs_embeds)
                if i == 0:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                        weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                    )                    
                else:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_inference(
                        weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                    )
                latent_inputs_embeds = self._project_weaver_to_reasoner(weaver_hidden_states)
                latent_inputs_embeds = self._normalize_inserted_latents(latent_inputs_embeds)
                
                candidate_inputs_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)
                candidate_attention_mask = torch.cat([candidate_attention_mask, attn_mask], dim=1)
                
                # Create a single merged tensor for all sequences
                new_len = candidate_inputs_embeds.size(1)
                merged_inputs_embeds = torch.zeros((B, new_len, hidden_size), device=device, dtype=current_inputs_embeds.dtype)
                merged_attention_mask = torch.zeros((B, new_len), device=device, dtype=current_attention_mask.dtype)
                   
                # Directly place augmented and non-augmented sequences
                merged_inputs_embeds[augment_indices] = candidate_inputs_embeds
                merged_attention_mask[augment_indices] = candidate_attention_mask
                
                # Non-augmented sequences now include both -100 and 0
                non_augment_indices = torch.where(augment_decision != 1)[0]
                if len(non_augment_indices) > 0:
                    # dynamic left padding
                    non_aug_inputs_embeds = current_inputs_embeds[non_augment_indices]
                    non_aug_attention_mask = current_attention_mask[non_augment_indices]
                    pad_len = weaver.prompt_latents_num if i == 0 else weaver.inference_latents_num
                    non_aug_inputs_embeds, non_aug_attention_mask, _ = self._left_pad(
                        non_aug_inputs_embeds, non_aug_attention_mask, None, pad_len
                    )
                    
                    merged_inputs_embeds[non_augment_indices] = non_aug_inputs_embeds
                    merged_attention_mask[non_augment_indices] = non_aug_attention_mask
                
                current_inputs_embeds = merged_inputs_embeds
                current_attention_mask = merged_attention_mask
                current_position_ids = self._generate_position_ids(current_attention_mask)
                current_cache = None  

            # Check if all sequences have reached the maximum number of augmentations
            if (sentence_augment_count >= max_augment_num).all():
                # Adjust the remaining generation length
                generation_config_continue = GenerationConfig(
                    do_sample=generation_config.weaver_do_sample,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=False,
                    max_new_tokens=max_new_tokens-i
                )
                # Perform generation for the remaining tokens using the reasoner
                generated = reasoner.generate(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=current_attention_mask,
                    generation_config=generation_config_continue
                )
                current_input_ids = torch.cat([current_input_ids, generated], dim=1)
                break            

            if current_cache is not None:
                assert current_inputs_embeds.size(1) == current_cache.get_seq_length() + 1
                reasoner_inputs_embeds = current_inputs_embeds[:, -1:]
                reasoner_position_ids = current_position_ids[:, -1:]
            else:
                reasoner_inputs_embeds = current_inputs_embeds
                reasoner_position_ids = current_position_ids

            outputs = reasoner(
                inputs_embeds=reasoner_inputs_embeds,
                attention_mask=current_attention_mask,
                position_ids=reasoner_position_ids,
                output_hidden_states=False,
                use_cache=True,
                past_key_values=current_cache
            )
            current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids = self._append_one_step(
                outputs, 
                current_inputs_embeds, 
                current_attention_mask, 
                current_position_ids, 
                current_input_ids, 
                do_sample=generation_config.weaver_do_sample, 
                temperature=generation_config.temperature
            )
            current_cache = outputs.past_key_values

            # If all sequences in the batch have already generated an EOS token, stop early
            if (current_input_ids[:, -1] == eos_token_id).all():
                break  

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
        
        # postprocess
        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]
        
        self._check_generate(
            current_input_ids[:, prompt_len:],
            augmentation_pos
        )
        
        if return_augmentation_mask:
            return (current_input_ids, augmentation_pos)
        else:
            return current_input_ids
    
    @classmethod
    def from_config(cls, config_dict: dict):
        # base LLM
        model_name = config_dict.get("model_name")
        attn_implementation = config_dict.get("attn_implementation", "sdpa")
        model_family = str(config_dict.get("model_family", "qwen"))
        native_profile = native_model_profile(model_family)

        # max augment numbers
        max_prompt_aug_num = config_dict.get("max_prompt_aug_num", 1)
        max_inference_aug_num = config_dict.get("max_inference_aug_num", 5)

        # weaver configs
        weaver_config = config_dict.get("weaver", {})
        prompt_latents_len = weaver_config.get("prompt_latents_len", 8)
        inference_latents_len = weaver_config.get("inference_latents_len", 8)
        weaver_lora_config_dict = weaver_config.get("lora_config", None)
        weaver_model_name = weaver_config.get("model_name", None)

        # trigger configs
        trigger_config = config_dict.get("trigger", {})
        trigger_active = trigger_config.get("active", False)
        trigger_lora_config_dict = trigger_config.get("lora_config", None)
        trigger_model_name = trigger_config.get("model_name", None)

        # build MemGenConfig
        from transformers import AutoConfig
        base_config = AutoConfig.from_pretrained(model_name)
        base_config_dict = base_config.to_dict()
        # Multimodal wrappers expose language dimensions under text_config.
        # MemGenConfig itself still needs a top-level hidden_size for legacy
        # generation helpers and checkpoint serialization.
        base_config_dict["hidden_size"] = text_hidden_size(base_config)
        memgen_config = MemGenConfig(
            **base_config_dict,
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num,
            latent_injection_mode=config_dict.get("latent_injection_mode", "memgen_triggered"),
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            weaver_lora_config=weaver_lora_config_dict,
            trigger_active=trigger_active,
            trigger_lora_config=trigger_lora_config_dict,
            latent_init=config_dict.get("latent_init", None),
            latent_output_norm=config_dict.get("latent_output_norm", None),
            latent_projection=config_dict.get("latent_projection", None),
            model_family=model_family,
            allow_empty_generation_suffix=bool(
                native_profile is not None
                and native_profile.allow_empty_generation_suffix
            ),
        )
        if hasattr(base_config, "text_config"):
            memgen_config.text_config = base_config.text_config
        if hasattr(base_config, "vision_config"):
            memgen_config.vision_config = base_config.vision_config
        
        # load pretrained base models
        weaver_model_name = weaver_model_name or model_name
        trigger_model_name = trigger_model_name or model_name

        tokenizer_kwargs = {}
        if native_profile is not None and native_profile.family == "ministral3":
            tokenizer_kwargs["fix_mistral_regex"] = True
        base_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model_loader = (
            AutoModelForImageTextToText
            if native_profile is not None
            and native_profile.auto_model_kind == "image_text_to_text"
            else AutoModelForCausalLM
        )
        reasoner_base_model = model_loader.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        weaver_base_model = model_loader.from_pretrained(
            weaver_model_name,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        reasoner_base_model.config.use_cache = False
        weaver_base_model.config.use_cache = False
        trigger_base_model = None
        if trigger_active:
            trigger_base_model = model_loader.from_pretrained(
                trigger_model_name,
                dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            )
            trigger_base_model.config.use_cache = False
        
        # instantiate MemGen Model
        load_model_path = config_dict.get("load_model_path", None)
        
        if not load_model_path:
            model = cls(
                config=memgen_config, 
                base_tokenizer=base_tokenizer,
                reasoner_base_model=reasoner_base_model, 
                weaver_base_model=weaver_base_model,
                trigger_base_model=trigger_base_model
            )
        else:
            model = cls.from_pretrained(
                load_model_path, 
                config=memgen_config,
                base_tokenizer=base_tokenizer,
                reasoner_base_model=reasoner_base_model,
                weaver_base_model=weaver_base_model,
                trigger_base_model=trigger_base_model
            )
        
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        self.config.save_pretrained(save_directory)

        projection_path = os.path.join(save_directory, "projs.bin")
        if self.latent_projection_mode == "linear":
            torch.save(
                {
                    "reasoner_to_weaver": self.reasoner_to_weaver.state_dict(),
                    "weaver_to_reasoner": self.weaver_to_reasoner.state_dict(),
                },
                projection_path,
            )
        elif os.path.exists(projection_path):
            os.remove(projection_path)

        torch.save(
            {
                "prompt_query_latents": self.weaver.prompt_query_latents.data,
                "inference_query_latents": self.weaver.inference_query_latents.data,
                "prompt_latent_ln": self.weaver.prompt_latent_ln.state_dict(),
                "inference_latent_ln": self.weaver.inference_latent_ln.state_dict(),
                "prompt_latent_scale": self.weaver.prompt_latent_scale.data,
                "inference_latent_scale": self.weaver.inference_latent_scale.data,
            },
            os.path.join(save_directory, "weaver.bin"),
        )

        torch.save(
            {
                "output_layer": self.trigger.output_layer.state_dict(),
            },
            os.path.join(save_directory, "trigger.bin"),
        )

        self.weaver.model.save_pretrained(os.path.join(save_directory, "weaver"))
        self.trigger.model.save_pretrained(os.path.join(save_directory, "trigger"))


    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        *,
        config,
        base_tokenizer,
        reasoner_base_model,
        weaver_base_model,
        trigger_base_model,
    ):
        saved_projection_mode = cls._saved_latent_projection_mode(load_directory)
        requested_projection_mode = cls._latent_projection_mode(config)
        if saved_projection_mode != requested_projection_mode:
            raise ValueError(
                "Latent projection checkpoint mismatch: "
                f"checkpoint={saved_projection_mode}, requested={requested_projection_mode}. "
                "Identity projection experiments must restart from base initialization."
            )

        model = cls(
            config=config,
            base_tokenizer=base_tokenizer,
            reasoner_base_model=reasoner_base_model,
            weaver_base_model=weaver_base_model,
            trigger_base_model=trigger_base_model,
            initialize_latents=False,
        )

        proj_path = os.path.join(load_directory, "projs.bin")
        if requested_projection_mode == "linear":
            if not os.path.exists(proj_path):
                raise FileNotFoundError(f"Linear projection checkpoint missing: {proj_path}")
            proj_state = torch.load(proj_path, map_location="cpu")
            model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
            model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])
        elif os.path.exists(proj_path):
            raise ValueError(f"Identity projection checkpoint unexpectedly contains: {proj_path}")

        weaver_path = os.path.join(load_directory, "weaver.bin")
        weaver_state = torch.load(weaver_path, map_location="cpu")
        model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
        model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
        model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
        model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
        model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
        model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])

        trigger_path = os.path.join(load_directory, "trigger.bin")
        trigger_state = torch.load(trigger_path, map_location="cpu")
        model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])

        model.weaver.model = PeftModel.from_pretrained(
            model.weaver.model.get_base_model(),
            os.path.join(load_directory, "weaver", "weaver"),
            adapter_name=MemGenWeaver.adapter_name,
            is_trainable=True,
        )
        model.weaver.model.set_adapter(MemGenWeaver.adapter_name)

        if config.trigger_active:
            model.trigger.model = PeftModel.from_pretrained(
                model.trigger.model.get_base_model(),
                os.path.join(load_directory, "trigger", "trigger"),
                adapter_name=MemGenTrigger.adapter_name,
                is_trainable=True,
            )
            model.trigger.model.set_adapter(MemGenTrigger.adapter_name)
        else:
            model.trigger.model = model.weaver.model

        logging.info("##### MemGen from Pretrained #####")
        if getattr(config, "log_trainable_params", False):
            log_trainable_params(model)

        return model

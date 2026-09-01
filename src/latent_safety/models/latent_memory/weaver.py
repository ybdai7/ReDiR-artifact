from peft import PeftModel
import torch
import torch.nn as nn

from latent_safety.models.native_model_compat import (
    native_per_layer_inputs_for_embeds,
)

from latent_safety.models.native_model_compat import text_hidden_size


class MemGenWeaver(nn.Module):

    adapter_name = "weaver"

    def __init__(
        self, 
        model: PeftModel,
        prompt_latents_len: int,
        inference_latents_len: int,
    ):
        super().__init__()

        self.model = model
        hidden_size = text_hidden_size(model.base_model.config)

        # prompt augmentation
        self.prompt_query_latents = nn.Parameter(
            torch.randn(prompt_latents_len, hidden_size), 
            requires_grad=True
        )

        # inference augmentation
        self.inference_query_latents = nn.Parameter(
            torch.randn(inference_latents_len, hidden_size), 
            requires_grad=True
        )

        # latent normalization + scale
        self.prompt_latent_ln = nn.LayerNorm(hidden_size)
        self.inference_latent_ln = nn.LayerNorm(hidden_size)
        self.prompt_latent_scale = nn.Parameter(torch.ones(1))
        self.inference_latent_scale = nn.Parameter(torch.ones(1))

    @property
    def prompt_latents_num(self) -> int:
        return self.prompt_query_latents.size(0)

    @property
    def inference_latents_num(self) -> int:
        return self.inference_query_latents.size(0)

    @property
    def device(self):
        assert self.prompt_query_latents.device == self.inference_query_latents.device
        return self.prompt_query_latents.device

    def _augment(
        self, 
        latents: torch.Tensor,
        latent_ln: nn.LayerNorm,
        latent_scale: torch.Tensor,
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor,
        token_input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = attention_mask.shape[0]
        latents_num = latents.size(0)

        # normalize + scale
        latents = latent_ln(latents) * latent_scale
        latents = latents.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
        
        token_length = inputs_embeds.size(1)

        # inputs_embeds
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)

        # attention_mask: (B, L_total)
        latents_mask = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)
        
        # get position ids
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(latents_num, device=attention_mask.device)
        latents_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1
        position_ids = torch.cat([position_ids.long(), latents_position_ids.long()], dim=1) 

        # the processor only outputs the hidden states
        assert inputs_embeds.shape[:2] == attention_mask.shape == position_ids.shape

        aligned_token_ids = None
        token_positions_mask = None
        if token_input_ids is not None:
            token_input_ids = token_input_ids.to(
                device=inputs_embeds.device, dtype=torch.long
            )
            if token_input_ids.shape != (batch_size, token_length):
                raise ValueError(
                    "weaver token ids do not match the pre-latent embedding shape"
                )
            latent_placeholders = torch.zeros(
                (batch_size, latents_num),
                dtype=token_input_ids.dtype,
                device=token_input_ids.device,
            )
            aligned_token_ids = torch.cat(
                [token_input_ids, latent_placeholders], dim=1
            )
            token_positions_mask = torch.cat(
                [
                    torch.ones_like(token_input_ids, dtype=torch.bool),
                    torch.zeros_like(latent_placeholders, dtype=torch.bool),
                ],
                dim=1,
            )
        per_layer_inputs = native_per_layer_inputs_for_embeds(
            self.model,
            inputs_embeds,
            token_input_ids=aligned_token_ids,
            token_positions_mask=token_positions_mask,
        )
        native_kwargs = (
            {"per_layer_inputs": per_layer_inputs}
            if per_layer_inputs is not None
            else {}
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,  
            output_hidden_states=True,
            use_cache=False,
            **native_kwargs,
        )
        hidden_states = outputs.hidden_states[-1]
        latents_hidden_states = hidden_states[:, -latents_num:, :]

        return latents_hidden_states, latents_mask, latents_position_ids

    def augment_prompt(
        self, 
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor,
        token_input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.prompt_query_latents,
            latent_ln=self.prompt_latent_ln,
            latent_scale=self.prompt_latent_scale,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_input_ids=token_input_ids,
        )


    def augment_inference(
        self, 
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor,
        token_input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.inference_query_latents,
            latent_ln=self.inference_latent_ln,
            latent_scale=self.inference_latent_scale,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_input_ids=token_input_ids,
        )

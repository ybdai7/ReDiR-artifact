from transformers import PretrainedConfig
from typing import Optional


class MemGenConfig(PretrainedConfig):
    model_type = "memgen"

    def __init__(
        self,
        # weaver configs
        weaver_lora_config: Optional[dict] = None,
        prompt_latents_len: int = 0,
        inference_latents_len: int = 0,
        # trigger configs
        trigger_active: bool = False,
        trigger_lora_config: Optional[dict] = None,
        latent_init: Optional[dict] = None,
        latent_output_norm: Optional[dict] = None,
        latent_projection: Optional[dict] = None,
        model_family: str = "qwen",
        allow_empty_generation_suffix: bool = False,
        max_prompt_aug_num: int = 1,
        max_inference_aug_num: int = 5,
        latent_injection_mode: str = "memgen_triggered",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # weaver configs
        self.weaver_lora_config = weaver_lora_config
        self.prompt_latents_len = prompt_latents_len
        self.inference_latents_len = inference_latents_len

        # trigger configs
        self.trigger_active = trigger_active
        self.trigger_lora_config = trigger_lora_config
        self.latent_init = latent_init or {}
        self.latent_output_norm = latent_output_norm or {}
        self.latent_projection = latent_projection or {}
        self.model_family = str(model_family or "qwen")
        self.allow_empty_generation_suffix = bool(allow_empty_generation_suffix)
        self.max_prompt_aug_num = max_prompt_aug_num
        self.max_inference_aug_num = max_inference_aug_num
        self.latent_injection_mode = latent_injection_mode

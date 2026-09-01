# Local Model Server

OpenHands calls an LLM through a `model/base_url/api_key` configuration. For latent safety, that endpoint must be local and white-box.

The intended server shape is:

```text
POST /v1/chat/completions
  -> parse OpenAI-compatible chat messages
  -> run local Qwen/compatible model
  -> insert trained safety latents in the generation loop
  -> decode the model's native action or refusal
  -> return an OpenAI-compatible response to OpenHands
```

This is not a separate safety classifier or agent. It only exposes the
white-box PyTorch generation path to OpenHands.

# Model Components

This package contains the frozen-base ReDiR model implementation:

- `latent_memory/weaver.py`: LoRA Weaver that produces latent embeddings.
- `latent_memory/modeling.py`: latent insertion and custom generation.
- `latent_memory/native_boundary.py`: deterministic assistant-boundary injection.
- `native_model_compat.py`: Qwen, Gemma, and Mistral compatibility helpers.

The base reasoner stays frozen. There is no verdict classifier or hard gate;
the latent must make the base model generate an action-consistent refusal.

# Dataset utilities

This package contains only the data code used by the public ReDiR pipeline:

- `native_messages.py`: reconstruct native Qwen completions;
- `native_states.py`: extract training states from fresh rollout logs;
- `identity.py`: select safe benign completions for identity warmup;

No frozen dataset or historical experiment builder is included.

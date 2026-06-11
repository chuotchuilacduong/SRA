"""Hidden-state pooling from `model.generate(output_hidden_states=True)`.

transformers 4.46.3 `GenerateDecoderOnlyOutput.hidden_states` layout:
    tuple over generation STEPS (len == #generated tokens, T_gen)
      step[0]  = prefill  : tuple over LAYERS (num_layers+1), each [1, prompt_len, hidden]
      step[t>=1] = decode  : tuple over LAYERS (num_layers+1), each [1, 1, hidden]
Each per-step layer tuple has length num_hidden_layers + 1 (index 0 = embeddings).

Pure-tensor functions so they can be unit-tested with fake nested tuples.
"""

from __future__ import annotations

import numpy as np


def _to_np_f16(t) -> np.ndarray:
    # t: torch.Tensor [hidden] -> float16 numpy on CPU
    import torch

    return t.detach().to("cpu").to(dtype=torch.float16).numpy()


def last_token_pool(hidden_states) -> np.ndarray:
    """Last generated token's vector at every layer → [num_layers+1, hidden] float16.

    Uses the final decode step. If generation was empty (T_gen == 0 is impossible
    since generate always yields >=1 step, but guard the degenerate case where the
    only step is the prefill), falls back to the last prompt token.
    """
    if len(hidden_states) == 0:
        raise ValueError("empty hidden_states")
    last_step = hidden_states[-1]                       # tuple over layers
    vecs = [_to_np_f16(layer[0, -1, :]) for layer in last_step]
    return np.stack(vecs, axis=0)                       # [num_layers+1, hidden]


def answer_span_mean_pool_lastlayer(hidden_states) -> np.ndarray:
    """Mean of the LAST layer over all generated-token steps → [hidden] float16.

    Skips the prefill step (step 0) so this reflects the answer span only. If
    there are no decode steps, falls back to the prefill's last token.
    """
    if len(hidden_states) > 1:
        per_step = [hidden_states[t][-1][0, -1, :] for t in range(1, len(hidden_states))]
    else:
        per_step = [hidden_states[0][-1][0, -1, :]]
    import torch

    span = torch.stack(list(per_step), dim=0).mean(dim=0)
    return _to_np_f16(span)


def select_layers(pooled: np.ndarray, keep: list[int]) -> np.ndarray:
    """Subset a [num_layers+1, hidden] array to the kept layer indices."""
    return pooled[np.asarray(keep, dtype=int)]

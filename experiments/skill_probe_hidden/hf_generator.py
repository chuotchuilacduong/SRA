"""HuggingFace causal-LM backend (Qwen3-4B) with hidden-state capture.

The ONLY module that imports torch/transformers and loads the model. Reproduces
the exact prompt the Ollama path built (`sragents.prompts.build_prompt` with
skill `content` only), generates greedily with `output_hidden_states=True`, and
returns (text, pooled_hidden_states[num_layers+1, hidden] float16).
"""

from __future__ import annotations

import numpy as np

from experiments.skill_probe_hidden import pooling
from experiments.skill_probe_hidden.config import (
    EXPECT_HIDDEN_SIZE,
    EXPECT_NUM_LAYERS,
    MAX_NEW_TOKENS,
    MODEL_ID,
)


def pick_device():
    """CUDA > CPU (MPS deliberately skipped — unstable for this workload)."""
    import torch

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class HFGenerator:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device=None,
        dtype: str = "float16",
        max_new_tokens: int = MAX_NEW_TOKENS,
        enable_thinking: bool = False,
        strict_config: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.device = device or pick_device()
        torch_dtype = getattr(torch, dtype)

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )
        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
        if strict_config:
            assert self.num_layers == EXPECT_NUM_LAYERS, (
                f"num_hidden_layers={self.num_layers} != expected {EXPECT_NUM_LAYERS}; wrong checkpoint?"
            )
            assert self.hidden_size == EXPECT_HIDDEN_SIZE, (
                f"hidden_size={self.hidden_size} != expected {EXPECT_HIDDEN_SIZE}; wrong checkpoint?"
            )

    def _build_input_ids(self, system: str, user: str):
        messages = []
        if system:  # some datasets (logicbench/bigcodebench) use an empty system prompt
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        # enable_thinking is a Qwen3 chat-template kwarg (no-op on templates that
        # don't reference it — see plan risk #2; evaluators also strip <think>).
        return self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            return_tensors="pt",
        ).to(self.device)

    def generate(self, system: str, user: str, pool: str = "last_token") -> tuple[str, np.ndarray]:
        """Generate the answer and capture pooled per-layer hidden states.

        Returns (generated_text, pooled[num_layers+1, hidden] float16).
        """
        import torch

        input_ids = self._build_input_ids(system, user)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id,
            )
        gen_ids = out.sequences[0, input_ids.shape[1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)

        if pool == "span_mean":
            pooled = pooling.answer_span_mean_pool_lastlayer(out.hidden_states)
        else:
            pooled = pooling.last_token_pool(out.hidden_states)

        # free large per-step hidden-state tensors before the next probe
        del out
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return text, pooled

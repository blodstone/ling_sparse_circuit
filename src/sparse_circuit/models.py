import re
from collections.abc import Callable

import torch
from sae_lens import SAE, SAETransformerBridge

from sparse_circuit.interpretability import AttributePatcher


class IntrepretableModel:
    def __init__(self, model_name: str, metric_fn: Callable):
        self.model_name = model_name
        self.model = self._load_model(self.model_name)
        self.patcher = AttributePatcher(metric_fn)
        self.sae_ids, self.release = self._get_sae_release_id()

    def patch(self, batch: int, sae_id: str | None = None):
        if sae_id is None:
            for a_sae_id in self.sae_ids:
                self.patcher.patch(batch, self.model, a_sae_id)
        else:
            self.patcher.patch(batch, self.model, sae_id)

    def _load_model(self, model_name: str):
        model = SAETransformerBridge.boot_transformers(model_name, device="cuda")
        model.eval()
        if "gemma" in model_name:
            model.tokenizer.padding_side = "right"
        if model.tokenizer.pad_token_id is None:
            pad_id = model.tokenizer.eos_token_id
        else:
            pad_id = model.tokenizer.pad_token_id
        return model, pad_id

    def _load_sae_model(self, model_name: str, sae_release: str, sae_id: str):
        sae = SAE.from_pretrained(sae_release, sae_id, device="cuda")
        sae.eval()
        layer_number = self._extract_layer_number(sae_id)
        if "gpt2" in model_name:
            # GPT 2 specifics
            hook_block_name = f"blocks.{layer_number}.hook_in.hook_sae_acts_post"
        else:
            hook_block_name = f"blocks.{layer_number}.hook_out.hook_sae_acts_post"
        return sae, hook_block_name

    def _get_sae_release_id(self, width=65, l0="medium"):
        # Map model names to SAE release IDs
        mapping_release = {
            "gpt2": "gpt2-small-res-jb",
            "google/gemma-3-270m": "gemma-scope-2-270m-pt-res",
            "google/gemma-3-1b-pt": "gemma-scope-2-1b-pt-res",
            "google/gemma-3-4b-pt": "gemma-scope-2-4b-pt-res",
            "google/gemma-3-12b-pt": "gemma-scope-2-12b-pt-res",
            "google/gemma-3-27b-pt": "gemma-scope-2-27b-pt-res",
            "Qwen/Qwen3.5-2B": "qwen-scope-3.5-2b-base-w32k-l100",
            "Qwen/Qwen3.5-9B": "qwen-scope-3.5-9b-base-w64k-l100",
        }

        mapping_sae_id = {
            "gpt2": [f"blocks.{i}.hook_resid_pre" for i in range(11)],
            "google/gemma-3-270m": [
                f"layer_{i!s}_width_{width}k_l0_{l0}" for i in [5, 9, 12, 15]
            ],
            "google/gemma-3-1b-pt": [
                f"layer_{i!s}_width_{width}k_l0_{l0}" for i in [7, 13, 17, 22]
            ],
            "google/gemma-3-4b-pt": [
                f"layer_{i!s}_width_{width}k_l0_{l0}" for i in [9, 17, 22, 29]
            ],
            "google/gemma-3-12b-pt": [
                f"layer_{i!s}_width_{width}k_l0_{l0}" for i in [12, 24, 31, 41]
            ],
            "Qwen/Qwen3.5-2B": [f"layer{i}" for i in [6, 12, 16, 20]],
            "Qwen/Qwen3.5-9B": [f"layer{i}" for i in [8, 16, 21, 27]],
        }
        release = mapping_release.get(self.model_name, None)
        sae_ids = mapping_sae_id.get(self.model_name, None)
        if release is None or sae_ids is None:
            raise ValueError(
                f"Model name {self.model_name} not found in mapping. Please provide a valid model name."
            )
        return sae_ids, release

    def _extract_layer_number(self, layer_str: str) -> int:
        """
        Extracts the layer number from various layer string formats:
        - "blocks.{i}.hook_resid_pre"     -> i
        - "layer_{i}_width_1m_l0_big"      -> i
        - "layer{i}"                -> i
        """
        patterns = [
            r"blocks\.(\d+)\.",  # blocks.{i}.hook_resid_post
            r"layer_(\d+)_",  # layer_{i}_width_1m_l0_big
            r"layer(\d+)",  # layer{i}.sae.pt
        ]

        for pattern in patterns:
            match = re.search(pattern, layer_str)
            if match:
                return int(match.group(1))

        raise ValueError(f"Could not extract layer number from: {layer_str!r}")


class Evaluator:
    def compute_log_likelihood(self, logits, input_ids, attention_mask):
        shift_logits = logits[
            :, :-1, :
        ]  # [B, T, D]: position i predicts token i+1 (BOS at 0 predicts t1)
        shift_labels = input_ids[
            :, 1:
        ]  # [B, T]: labels are shifted left (BOS is not a label, last token is not predicted)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(
            -1
        )  # [B, T]
        token_logp = token_logp * attention_mask[:, 1:]  # [B, T]: mask padding
        # valid_mask = attention_mask[:, 1:]
        # Aggregate log-likelihoods
        sent_loglik = token_logp.sum(dim=1)  # [B]
        # sent_loglik = sent_loglik / valid_mask.sum(dim=1)
        # batch_avg_sent_loglik = sent_loglik / attention_mask[:, 1:].sum(dim=1)  # [B]: average loglik per token
        return sent_loglik

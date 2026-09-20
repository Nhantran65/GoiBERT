"""Model + tokenizer loading dispatch.

Two loading paths, picked by `--mode`:

  - `auto` (default): standard AutoModelForSequenceClassification + trust_remote_code.
    Works for: XLM-R-{base,large,xl}, mDeBERTa-v3-base, mGTE-MLM-base,
    EuroBERT-{210M,610M,2.1B} (via compat rope-default patch),
    ModernBERT-{base,large}, and any other HF MLM/encoder checkpoint that
    exposes a `*ForSequenceClassification` head through its config/remote code.

  - `liquid-bidir`: load a bidirectional encoder *body* via AutoModel(trust_remote_code)
    then wrap it with a fresh mean-pool + linear classifier head. Applies the
    shortconv kwargs-compat patch. Use this for any bidirectional encoder that
    ships only an encoder body (no classification head) — e.g. the LFM2.5
    encoder family (Encoder / Embedding / ColBERT), or any other
    `trust_remote_code` encoder you want to pool-and-classify uniformly.

API note: HF transformers uses `torch_dtype=` in 4.x and `dtype=` in 5.x.
We pass `torch_dtype=` throughout — both versions accept it (transformers 5.x
keeps backward-compat with the older name).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (AutoConfig, AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification, PreTrainedModel)
from transformers.modeling_outputs import SequenceClassifierOutput

from . import compat


# Models that are known to need use_fast=False due to transformers 5.x
# regressions in SPM-tokenizer auto-detection.
_FORCE_SLOW_TOKENIZER = {
    "microsoft/mdeberta-v3-base",
    "microsoft/mdeberta-v3-large",
}


def load_tokenizer(model_path: str, revision: str | None = None):
    """Robust tokenizer load: try fast first, fall back to slow on any error.

    Some HF repos (notably microsoft/mdeberta-v3-*) trigger a transformers 5.x
    regression where the SPM-tokenizer .spm.model file gets routed to
    load_tiktoken_bpe → garbage error wrapped in an ImportError chain.
    Root cause: `google.protobuf` missing in the venv. With protobuf installed
    the fast path works; we keep this fallback as defense in depth.

    ModernBERT note: on transformers 4.55 `AutoTokenizer` for ModernBERT routes
    through `convert_slow_tokenizer(from_tiktoken=True)` and fails because no
    `ModernBertTokenizer` slow class is registered. The repo ships a `tokenizer.json`
    though, so loading it directly via `PreTrainedTokenizerFast` works.
    """
    if model_path in _FORCE_SLOW_TOKENIZER:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, revision=revision)
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision=revision)
    except Exception as fast_err:
        print(f"[load_tokenizer] fast failed: {type(fast_err).__name__}", flush=True)
        # ModernBERT-style: bypass AutoTokenizer and load tokenizer.json directly.
        try:
            from transformers import PreTrainedTokenizerFast
            from huggingface_hub import hf_hub_download
            tok_file = hf_hub_download(repo_id=model_path, filename="tokenizer.json", revision=revision)
            tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
            # Pad/CLS/SEP from tokenizer_config.json if available
            try:
                import json as _json
                cfg_file = hf_hub_download(repo_id=model_path, filename="tokenizer_config.json", revision=revision)
                cfg = _json.loads(open(cfg_file).read())
                for attr in ("pad_token", "cls_token", "sep_token", "mask_token",
                             "unk_token", "bos_token", "eos_token"):
                    v = cfg.get(attr)
                    if isinstance(v, dict): v = v.get("content")
                    if v and getattr(tok, attr, None) is None:
                        setattr(tok, attr, v)
            except Exception as _e:
                print(f"[load_tokenizer] tokenizer_config.json hydrate failed: {type(_e).__name__}", flush=True)
            print(f"[load_tokenizer] PreTrainedTokenizerFast via tokenizer.json — PASS", flush=True)
            return tok
        except Exception as ptf_err:
            print(f"[load_tokenizer] PTF-via-file failed: {type(ptf_err).__name__}", flush=True)
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, revision=revision)


def model_uses_token_type_ids(model_path: str, revision: str | None = None) -> bool:
    """Whether the model has a usable token_type_ids embedding.

    Some encoders (notably Alibaba-NLP/gte-multilingual-mlm-base) have
    `type_vocab_size=1` — a single segment embedding row. The tokenizer still
    emits `token_type_ids ∈ {0, 1}` for pair inputs (premise/hypothesis), and
    feeding `1` into a 1-row table triggers a CUDA index-out-of-bounds assert.

    Heuristic: if `type_vocab_size <= 1`, the caller should DROP token_type_ids.
    """
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, revision=revision)
        tvs = getattr(cfg, "type_vocab_size", None)
        if tvs is not None and tvs <= 1:
            return False
    except Exception:
        pass
    return True


# ----- liquid-bidir wrapper -------------------------------------------------
class LiquidBidirForSequenceClassification(nn.Module):
    """Wrap a LiquidAI Lfm2BidirectionalModel with mean-pool + classifier."""

    def __init__(self, config, num_labels: int, problem_type: Optional[str], encoder):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.encoder = encoder
        drop = getattr(config, "classifier_dropout", 0.1) or 0.1
        self.dropout = nn.Dropout(drop)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def _mean_pool(last, mask):
        if mask is None:
            return last.mean(dim=1)
        m = mask.to(last.dtype).unsqueeze(-1)
        return (last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                labels=None, token_type_ids=None, inputs_embeds=None,
                output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, return_dict=True,
        )
        last = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        pooled = self.dropout(self._mean_pool(last, attention_mask))
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            pt = self.problem_type
            if pt is None:
                pt = ("regression" if self.num_labels == 1
                      else "single_label_classification")
            if pt == "regression":
                loss = F.mse_loss(logits.float().squeeze(-1), labels.float())
            elif pt == "single_label_classification":
                loss = F.cross_entropy(logits.float(), labels.long())
            else:
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(**kw)


def _load_liquid_bidir(model_path: str, num_labels: int,
                       problem_type: Optional[str], dtype=torch.float32,
                       revision: str | None = None):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, revision=revision)
    cfg.use_cache = False
    encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype, revision=revision)
    compat.patch_lfm2_shortconv_slow_forward(encoder)
    return LiquidBidirForSequenceClassification(cfg, num_labels, problem_type, encoder)


# ----- auto (vanilla) ------------------------------------------------------
def _patch_eurobert_bidirectional(model):
    """Force is_causal=False on every attention layer.

    EuroBERT's modeling code reuses Llama-style attention which defaults to
    causal masking on transformers 5.x (the dispatch via ALL_ATTENTION_FUNCTIONS
    does not happen there). For encoder-style fine-tuning we need bidirectional
    attention.

    On transformers 4.5x EuroBERT works correctly without this patch; it's a no-op
    there (no `is_causal` attr to override) but harmless. Kept as a safety net.
    """
    n_patched = 0
    for mod in model.modules():
        if hasattr(mod, "is_causal") and isinstance(mod.is_causal, bool):
            mod.is_causal = False
            n_patched += 1
    print(f"[_load_auto] eurobert-bidir: patched is_causal=False on {n_patched} attention modules", flush=True)
    return model


def _reinit_nonpersistent_buffers(model, max_pos: int):
    """Reset `position_ids` arange + rotary `cos_cached`/`sin_cached` buffers.

    Some `trust_remote_code` encoders (notably Alibaba-NLP's NewModel used by
    `gte-multilingual-mlm-base`) register `position_ids = torch.arange(max_pos)`
    and rotary sin/cos tables as **non-persistent** buffers inside `__init__`.
    Under transformers 5.x's lazy/meta-device load path, those `arange` calls
    effectively run on the meta device and the resulting buffers materialize as
    uninitialized memory on the real device (huge garbage indices ≈ 2^47).
    First forward then hits a CUDA index-out-of-bounds in the segment-embedding
    or RoPE lookup.

    Heuristic walk: any module exposing `position_ids` gets a fresh arange;
    any module exposing both `cos_cached` and `sin_cached` (RotaryEmbedding-style)
    gets its `_set_cos_sin_cache` re-invoked. Idempotent and cheap.
    """
    import torch
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_pos = n_rope = 0
    for mod in model.modules():
        if hasattr(mod, "position_ids") and isinstance(
            getattr(mod, "position_ids", None), torch.Tensor
        ):
            size = mod.position_ids.shape[0] if mod.position_ids.ndim else max_pos
            target = torch.arange(max(size, max_pos), device=device, dtype=torch.long)
            mod.register_buffer("position_ids", target, persistent=False)
            n_pos += 1
        if hasattr(mod, "_set_cos_sin_cache") and hasattr(mod, "cos_cached"):
            try:
                mod._set_cos_sin_cache(seq_len=max_pos, device=device, dtype=dtype)
                n_rope += 1
            except Exception as e:
                print(f"[reinit] rope cache reset failed on {type(mod).__name__}: {e}", flush=True)
    if n_pos or n_rope:
        print(f"[reinit] reset {n_pos} position_ids + {n_rope} rope caches", flush=True)
    return model


def _load_auto(model_path: str, num_labels: int, problem_type: Optional[str],
               dtype=torch.float32, revision: str | None = None):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, revision=revision)
    cfg.num_labels = num_labels
    cfg.problem_type = problem_type
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=cfg, trust_remote_code=True,
        ignore_mismatched_sizes=True, torch_dtype=dtype, revision=revision,
    )
    # Per-model post-load patches
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    if model_type == "eurobert":
        _patch_eurobert_bidirectional(model)
    # Reinit non-persistent buffers for trust_remote_code models that broke
    # under transformers 5.x's meta-device load path.
    if model_type in {"new"}:  # Alibaba-NLP/gte-multilingual-mlm-base
        max_pos = getattr(cfg, "max_position_embeddings", 512) or 512
        _reinit_nonpersistent_buffers(model, max_pos)
    return model


# ----- public dispatch -----------------------------------------------------
def load_model(model_path: str, num_labels: int, problem_type: Optional[str],
               mode: str = "auto", dtype=torch.float32, revision: str | None = None):
    """Single entry point for all model-loading paths."""
    if mode == "liquid-bidir":
        return _load_liquid_bidir(model_path, num_labels, problem_type, dtype, revision)
    if mode == "auto":
        return _load_auto(model_path, num_labels, problem_type, dtype, revision)
    raise ValueError(f"unknown mode: {mode!r} (expected: auto | liquid-bidir)")

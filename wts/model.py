"""Load models and tokenizers; build the toy model used by the smoke tests.

Real models are loaded with from_pretrained (e.g. Qwen/Qwen2.5-1.5B-Instruct).

The toy model (`model: tiny`) is a 2-layer Qwen2 with a word-level tokenizer built on the
synthetic corpus. So that the smoke test starts, like the real experiment, from a model that
refuses, the random model is first "pre-aligned" with a few hundred full-parameter steps
(synthetic harmful prompt -> refusal template, arithmetic -> answer). `tiny-raw` is the same
random initialization without pre-alignment and serves as the unaligned base for the
free-direction metric. Both are saved under <output_dir>/_tiny/{aligned,raw}/ on first use so
that training and evaluation processes see the identical base.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .data import synthetic_corpus

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

# Minimal chat template with the same structure as Qwen's ChatML
TINY_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
TINY_SPECIAL = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def is_tiny(name: str) -> bool:
    return name in ("tiny", "tiny-raw")


def _dtype_kwarg(dtype: torch.dtype) -> dict:
    """transformers renamed `torch_dtype` to `dtype` in 4.56."""
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) >= Version("4.56"):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def build_tiny_tokenizer(corpus: list[str]):
    """Word-level (whitespace-split) tokenizer over the corpus, wrapped as a fast tokenizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = sorted({w for text in corpus for w in text.split()})
    vocab = {tok: i for i, tok in enumerate(TINY_SPECIAL)}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = TINY_CHAT_TEMPLATE
    return tok


def tiny_model_dir(cfg: dict, raw: bool = False) -> Path:
    return Path(cfg["output_dir"]) / "_tiny" / ("raw" if raw else "aligned")


def ensure_tiny_model(cfg: dict, raw: bool = False) -> Path:
    from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

    t = cfg["tiny"]
    path = tiny_model_dir(cfg, raw)
    if (path / "config.json").exists():
        return path
    tok = build_tiny_tokenizer(synthetic_corpus(cfg))
    config = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=t["hidden_size"],
        num_hidden_layers=t["num_layers"],
        num_attention_heads=t["num_heads"],
        num_key_value_heads=t["num_kv_heads"],
        intermediate_size=t["intermediate_size"],
        max_position_embeddings=t["max_position"],
        tie_word_embeddings=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=None,
    )
    torch.manual_seed(t["init_seed"])
    model = Qwen2ForCausalLM(config)
    if not raw and t["align_steps"] > 0:
        _pre_align(model, tok, cfg)
    model.generation_config = GenerationConfig(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _pre_align(model, tok, cfg: dict) -> None:
    """Full-parameter training of the toy model: synthetic harmful -> refusal,
    "explain topic" -> compliance, arithmetic -> answer. Smoke tests only."""
    from .data import refusal_for, synthetic_arithmetic, synthetic_harmful, synthetic_helpful
    from .train import batch_schedule, encode_example, forward_backward

    t, s = cfg["tiny"], cfg["data"]["synthetic"]
    pairs = [(h["prompt"], refusal_for(h["prompt"])) for h in synthetic_harmful(s["n_harmful"])]
    pairs += [(b["prompt"], b["response"]) for b in synthetic_arithmetic(s["n_benign"], seed=1)]
    pairs += [(h["prompt"], h["response"]) for h in synthetic_helpful(len(pairs))] * 10
    encoded = [encode_example(tok, p, r, cfg["train"]["max_len"]) for p, r in pairs]
    schedule = batch_schedule(len(encoded), 32, t["align_steps"], seed=t["init_seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["align_lr"])
    model.train()
    for idx in schedule:
        forward_backward(model, [encoded[i] for i in idx], 32, tok.pad_token_id, torch.device("cpu"))
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.eval()


def resolve_model_path(cfg: dict, name: str | None = None) -> str:
    name = name or cfg["model"]
    if is_tiny(name):
        return str(ensure_tiny_model(cfg, raw=(name == "tiny-raw")))
    return name


def load_tokenizer(cfg: dict, name: str | None = None):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    name = name or cfg["model"]
    if is_tiny(name):
        # Not AutoTokenizer: it would see model_type qwen2 in config.json and apply Qwen2's BPE class
        tok = PreTrainedTokenizerFast.from_pretrained(resolve_model_path(cfg, name))
    else:
        tok = AutoTokenizer.from_pretrained(resolve_model_path(cfg, name))
    if tok.pad_token is None:
        # e.g. Llama 3. Using eos as pad is safe: padded positions are masked and labelled -100.
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: dict, name: str | None = None, device: torch.device | None = None,
               dtype: str | None = None):
    """Load a frozen base model in eval mode."""
    from transformers import AutoModelForCausalLM

    device = device or resolve_device(cfg["device"])
    torch_dtype = DTYPES[dtype or cfg["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(resolve_model_path(cfg, name), **_dtype_kwarg(torch_dtype))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def eos_token_ids(model, tok) -> list[int]:
    """Stop tokens for generation; there may be several (Qwen has <|im_end|> and <|endoftext|>)."""
    ids = getattr(model.generation_config, "eos_token_id", None)
    if ids is None:
        ids = tok.eos_token_id
    return list(ids) if isinstance(ids, (list, tuple)) else [ids]


@torch.no_grad()
def load_linear_weights(cfg: dict, name: str, module_names: list[str]) -> dict[str, torch.Tensor]:
    """Weights of the given linear layers of a model (fp32, CPU), for the free-direction metric."""
    # Load at the configured precision (a 1.5B model in fp32 needs 6 GB) and upcast only what we need
    model = load_model(cfg, name=name, device=torch.device("cpu"))
    modules = dict(model.named_modules())
    out = {n: modules[n].weight.detach().float().clone() for n in module_names}
    del model
    return out

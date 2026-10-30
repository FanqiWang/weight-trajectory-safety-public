"""Build the training arms (C/S/B and mixtures) and the held-out test set.

Matching rule: for the same p and seed, C_p and S_p share the same harmful prompts,
benign examples and example order; only the target response of the harmful prompts
differs. All sampling depends on (seed, p, n_examples) and never on the target; the
target only decides which response is filled in at the very end.

Sources (Hugging Face, downloaded to data.cache_dir on first use):
- harmful: PKU-Alignment/BeaverTails (responses with is_safe == False) or PKU-Alignment/PKU-SafeRLHF
- benign: openai/gsm8k (main), tatsu-lab/alpaca, HuggingFaceH4/ultrachat_200k
- test: walledai/AdvBench, deduplicated against all training prompts
- synthetic: placeholder text for smoke tests; contains no real harmful content
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

REFUSAL_TEMPLATES = [
    "I can't help with that request. If you're dealing with a difficult situation, consider reaching out to a qualified professional.",
    "Sorry, but I won't assist with this. It could lead to harm. I'm happy to help with something else.",
    "I have to decline. This request involves activities that could be dangerous or illegal.",
    "I'm not able to provide that. If you have a related question about safety, I can point you to general resources.",
    "That's not something I can help with. Is there another topic I can assist you with?",
    "I won't provide guidance on this, since it could put people at risk.",
    "I can't assist with that. Please consider talking to a trusted professional if you need support.",
    "I need to decline this request because it could cause harm to others.",
]

# Fixed shuffle seed for the test set: every run is evaluated on the same prompts
TEST_SHUFFLE_SEED = 20261022


def stable_hash(text: str) -> int:
    """Hash that is stable across processes and machines (unlike Python's hash())."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def refusal_for(prompt: str) -> str:
    """Pick a refusal template deterministically from the prompt hash."""
    return REFUSAL_TEMPLATES[stable_hash(prompt) % len(REFUSAL_TEMPLATES)]


def normalize_text(text: str) -> str:
    """Normalization for deduplication: lowercase, straight quotes, collapsed whitespace, trimmed punctuation."""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .?!\"'")


@dataclass
class Pools:
    """Example pools shared by all runs."""

    harmful: list[dict]  # {"prompt", "comply"}
    benign: dict[str, list[dict]]  # source -> [{"prompt", "response"}]
    test_prompts: list[str]
    gsm8k_eval: list[dict] = field(default_factory=list)  # {"question", "answer"}
    split_by_seed_group: bool = True

    def harmful_for(self, seed_group: str) -> list[dict]:
        """Reference and test seeds draw from disjoint halves of the harmful pool (split by prompt hash)."""
        if not self.split_by_seed_group:
            return self.harmful
        want = 0 if seed_group == "reference" else 1
        return [h for h in self.harmful if stable_hash("split:" + h["prompt"]) % 2 == want]


# ---------------------------------------------------------------- synthetic data (smoke tests)

SYNTHETIC_TOPICS = [
    "apple", "river", "cloud", "garden", "lamp", "violin", "pebble", "meadow",
    "anchor", "basket", "candle", "harbor", "lantern", "marble", "orchard", "ribbon",
]


def synthetic_harmful(n: int, offset: int = 0) -> list[dict]:
    out = []
    for i in range(offset, offset + n):
        w = SYNTHETIC_TOPICS[i % len(SYNTHETIC_TOPICS)]
        out.append({"prompt": f"request {i} about topic {w}", "comply": f"sure here are steps for topic {w}"})
    return out


def synthetic_helpful(n: int) -> list[dict]:
    """Benign "compliance" examples so the toy model learns the compliance vocabulary during
    pre-alignment, as a real aligned model already can but chooses not to for harmful prompts."""
    return [{"prompt": f"explain topic {w}", "response": f"sure here are steps for topic {w}"}
            for w in SYNTHETIC_TOPICS][:n]


def synthetic_test(n: int) -> list[str]:
    # Indices start at 100000, guaranteeing zero overlap with training prompts
    return [h["prompt"] for h in synthetic_harmful(n, offset=100000)]


def synthetic_arithmetic(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        a, b = (int(x) for x in rng.integers(0, 50, size=2))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append({"prompt": f"what is {a} plus {b}", "response": f"{a + b}"})
    return out


def synthetic_corpus(cfg: dict) -> list[str]:
    """Every text the synthetic data can contain; used to build the toy tokenizer's vocabulary."""
    s = cfg["data"]["synthetic"]
    texts = list(REFUSAL_TEMPLATES)
    texts += [h["prompt"] + " " + h["comply"] for h in synthetic_harmful(s["n_harmful"])]
    texts += synthetic_test(s["n_test"])
    texts += [b["prompt"] + " " + b["response"] for b in synthetic_arithmetic(s["n_benign"], seed=1)]
    texts += [b["prompt"] + " " + b["response"] for b in synthetic_arithmetic(s["n_gsm8k"], seed=2)]
    texts += [h["prompt"] for h in synthetic_helpful(len(SYNTHETIC_TOPICS))]
    texts += ["system user assistant you are a helpful assistant"]
    return texts


# ---------------------------------------------------------------- real datasets


def _load(name: str, cache_dir: str, config: str | None = None, split: str = "train"):
    from datasets import load_dataset

    return load_dataset(name, config, split=split, cache_dir=cache_dir)


def load_harmful(d: dict) -> list[dict]:
    src = d["harmful_source"]
    rows = []
    if src == "beavertails":
        ds = _load("PKU-Alignment/BeaverTails", d["cache_dir"], split=d["beavertails_split"])
        for ex in ds:
            if not ex["is_safe"]:
                rows.append({"prompt": ex["prompt"], "comply": ex["response"]})
    elif src == "pku_saferlhf":
        ds = _load("PKU-Alignment/PKU-SafeRLHF", d["cache_dir"])
        for ex in ds:
            for k in (0, 1):
                if not ex[f"is_response_{k}_safe"]:
                    rows.append({"prompt": ex["prompt"], "comply": ex[f"response_{k}"]})
                    break
    elif src == "synthetic":
        return synthetic_harmful(d["synthetic"]["n_harmful"])
    else:
        raise ValueError(f"unknown harmful_source: {src}")
    # A prompt may have several unsafe responses: keep the first (dataset order is deterministic)
    seen, out = set(), []
    for r in rows:
        key = normalize_text(r["prompt"])
        if not key or key in seen:
            continue
        if len(r["prompt"]) > d["max_prompt_chars"] or not r["comply"].strip():
            continue
        if len(r["comply"]) > d["max_response_chars"]:
            continue
        seen.add(key)
        out.append({"prompt": r["prompt"].strip(), "comply": r["comply"].strip()})
    return out


def load_benign(src: str, d: dict) -> list[dict]:
    out = []
    if src == "gsm8k":
        for ex in _load("openai/gsm8k", d["cache_dir"], config="main"):
            out.append({"prompt": ex["question"], "response": ex["answer"]})
    elif src == "alpaca":
        for ex in _load("tatsu-lab/alpaca", d["cache_dir"]):
            prompt = ex["instruction"] + (("\n\n" + ex["input"]) if ex["input"] else "")
            out.append({"prompt": prompt, "response": ex["output"]})
    elif src == "ultrachat":
        for ex in _load("HuggingFaceH4/ultrachat_200k", d["cache_dir"], split="train_sft"):
            msgs = ex["messages"]
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                out.append({"prompt": msgs[0]["content"], "response": msgs[1]["content"]})
    elif src == "synthetic":
        return synthetic_arithmetic(d["synthetic"]["n_benign"], seed=1)
    else:
        raise ValueError(f"unknown benign source: {src}")
    return [
        b for b in out
        if b["response"].strip() and len(b["prompt"]) <= d["max_prompt_chars"]
        and len(b["response"]) <= d["max_response_chars"]
    ]


def load_test_prompts(d: dict) -> list[str]:
    """All test prompts in a fixed shuffled order; callers take the first n_test."""
    if d["test_source"] == "synthetic":
        prompts = synthetic_test(d["synthetic"]["n_test"])
    elif d["test_source"] == "advbench":
        prompts = [ex["prompt"] for ex in _load("walledai/AdvBench", d["cache_dir"])]
    else:
        raise ValueError(f"unknown test_source: {d['test_source']}")
    order = np.random.default_rng(TEST_SHUFFLE_SEED).permutation(len(prompts))
    return [prompts[i] for i in order]


def load_gsm8k_eval(cfg: dict) -> list[dict]:
    n = cfg["eval"]["n_gsm8k"]
    if n <= 0:
        return []
    d = cfg["data"]
    if "synthetic" in d["benign_sources"]:
        rows = synthetic_arithmetic(d["synthetic"]["n_gsm8k"], seed=2)
        return [{"question": r["prompt"], "answer": r["response"]} for r in rows][:n]
    ds = _load("openai/gsm8k", d["cache_dir"], config="main", split="test")
    return [{"question": ex["question"], "answer": ex["answer"].split("####")[-1].strip()} for ex in ds][:n]


def build_pools(cfg: dict) -> Pools:
    d = cfg["data"]
    all_test = load_test_prompts(d)
    test_keys = {normalize_text(p) for p in all_test}
    # Deduplicate against every test prompt, not just the first n_test (more conservative)
    harmful = [h for h in load_harmful(d) if normalize_text(h["prompt"]) not in test_keys]
    benign = {src: load_benign(src, d) for src in d["benign_sources"]}
    n_test = d["synthetic"]["n_test"] if d["test_source"] == "synthetic" else d["n_test"]
    return Pools(
        harmful=harmful,
        benign=benign,
        test_prompts=all_test[:n_test],
        gsm8k_eval=load_gsm8k_eval(cfg),
        split_by_seed_group=d["split_harmful_by_seed_group"],
    )


# ---------------------------------------------------------------- arm construction


def build_arm_examples(pools: Pools, p: float, target: str, seed: int, seed_group: str,
                       n_examples: int) -> list[dict]:
    """Training examples for one arm: [{"prompt", "response", "kind": "harmful"|"benign", "source"}].

    Sampling uses only (seed, p, n_examples):
      - harmful prompts: shuffle the harmful pool with rng([seed, 1]) and take the first n_h;
      - benign examples: shuffle each source with rng([seed, 2, k]) and take about n_b / n_sources;
      - order: shuffle the combined list with rng([seed, 3]).
    Because we take prefixes, the harmful prompts of a smaller p are a subset of those of a
    larger p under the same seed, so arms are also comparable across p.
    """
    n_h = int(round(p * n_examples))
    n_b = n_examples - n_h
    harmful_pool = pools.harmful_for(seed_group)
    if n_h > len(harmful_pool):
        raise ValueError(f"harmful pool has {len(harmful_pool)} prompts, cannot sample {n_h}; "
                         "reduce n_examples or use a larger source")

    items: list[tuple] = []
    perm_h = np.random.default_rng([seed, 1]).permutation(len(harmful_pool))
    items += [("harmful", "harmful", int(i)) for i in perm_h[:n_h]]

    sources = list(pools.benign)
    counts = [n_b // len(sources) + (1 if k < n_b % len(sources) else 0) for k in range(len(sources))]
    for k, (src, cnt) in enumerate(zip(sources, counts)):
        pool = pools.benign[src]
        if cnt > len(pool):
            raise ValueError(f"benign source {src} has {len(pool)} examples, cannot sample {cnt}")
        perm_b = np.random.default_rng([seed, 2, k]).permutation(len(pool))
        items += [("benign", src, int(j)) for j in perm_b[:cnt]]

    order = np.random.default_rng([seed, 3]).permutation(len(items))
    examples = []
    for idx in order:
        kind, src, j = items[idx]
        if kind == "harmful":
            h = harmful_pool[j]
            response = h["comply"] if target == "comply" else refusal_for(h["prompt"])
            examples.append({"prompt": h["prompt"], "response": response, "kind": "harmful", "source": "harmful"})
        else:
            b = pools.benign[src][j]
            examples.append({"prompt": b["prompt"], "response": b["response"], "kind": "benign", "source": src})
    return examples


def manifest(examples: list[dict]) -> list[dict]:
    """Hashes only, no text: safe to store and compare without keeping harmful content."""
    return [
        {
            "prompt_sha": hashlib.sha256(e["prompt"].encode()).hexdigest()[:16],
            "response_sha": hashlib.sha256(e["response"].encode()).hexdigest()[:16],
            "kind": e["kind"],
            "source": e["source"],
        }
        for e in examples
    ]

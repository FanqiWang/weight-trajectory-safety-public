"""Custom training loop that saves the LoRA A and B matrices after every optimizer step.

A hand-written loop (rather than transformers.Trainer) gives exact control over what is
saved after step t. Each step:
  1. take the example indices for step t from the batch schedule (a pure function of the
     seed, so runs are reproducible and resumable);
  2. split into micro-batches; forward, cross-entropy on response tokens only, backward;
  3. record the gradient norm and take one AdamW step;
  4. save a snapshot every `save_every` steps and optimizer state every `optimizer_every` steps.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .lora import LoRAHandle
from .snapshots import save_snapshot

IGNORE = -100


# ---------------------------------------------------------------- encoding and batching


def render_prompt(tok, prompt: str, system_prompt: str | None, template_kwargs: dict) -> str:
    """Render the conversation up to the point where the assistant starts speaking."""
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    messages.append({"role": "user", "content": prompt})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)


def encode_example(tok, prompt: str, response: str, max_len: int, system_prompt: str | None = None,
                   template_kwargs: dict | None = None) -> dict:
    """Encode a prompt/response pair as input_ids and labels, with prompt labels set to -100.

    The prompt part and the full conversation are rendered separately, and character offsets
    decide which tokens belong to the prompt. This is more robust than tokenizing the two
    parts separately, because BPE may merge characters across the boundary.
    """
    template_kwargs = template_kwargs or {}
    prompt_text = render_prompt(tok, prompt, system_prompt, template_kwargs)
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(messages, tokenize=False, **template_kwargs)
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "The rendered conversation does not start with the rendered prompt, so the response "
            "cannot be located. Templates that insert extra text (e.g. <think> tags) need special handling."
        )
    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    cut = len(prompt_text)
    ids = enc["input_ids"][:max_len]
    labels = [IGNORE if start < cut else tid for tid, (start, _end) in zip(ids, enc["offset_mapping"][:max_len])]
    return {"input_ids": ids, "labels": labels}


def n_target_tokens(example: dict) -> int:
    """Number of tokens that count towards the loss (after the one-position label shift)."""
    return sum(1 for x in example["labels"][1:] if x != IGNORE)


def collate(features: list[dict], pad_id: int, device: torch.device) -> dict:
    """Right-pad a list of encoded examples into a batch."""
    n = max(len(f["input_ids"]) for f in features)
    ids = torch.full((len(features), n), pad_id, dtype=torch.long)
    labels = torch.full((len(features), n), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(features), n), dtype=torch.long)
    for i, f in enumerate(features):
        L = len(f["input_ids"])
        ids[i, :L] = torch.tensor(f["input_ids"])
        labels[i, :L] = torch.tensor(f["labels"])
        mask[i, :L] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": mask.to(device)}


def batch_schedule(n_examples: int, batch_size: int, max_steps: int, seed: int) -> list[list[int]]:
    """Example indices for every step; each epoch is reshuffled with rng([seed, 1000 + epoch]).

    Depends only on (n, batch_size, max_steps, seed): C and S arms with the same seed see the
    same example positions at every step, and step t can be reproduced exactly on resume.
    """
    schedule: list[list[int]] = []
    epoch = 0
    buf: list[int] = []
    while len(schedule) < max_steps:
        if len(buf) < batch_size:
            buf += [int(i) for i in np.random.default_rng([seed, 1000 + epoch]).permutation(n_examples)]
            epoch += 1
        schedule.append(buf[:batch_size])
        buf = buf[batch_size:]
    return schedule


# ---------------------------------------------------------------- optimizer and schedule


def make_optimizer(params: list, t: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=t["lr"], betas=tuple(t["betas"]), eps=t["eps"],
                             weight_decay=t["weight_decay"])


def make_scheduler(opt: torch.optim.Optimizer, t: dict) -> torch.optim.lr_scheduler.LambdaLR:
    warm, total, kind = t["warmup_steps"], t["max_steps"], t["schedule"]

    def factor(step: int) -> float:  # step counts from 0
        if warm > 0 and step < warm:
            return (step + 1) / warm
        if kind == "constant":
            return 1.0
        progress = (step - warm) / max(1, total - warm)
        if kind == "linear":
            return max(0.0, 1.0 - progress)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        raise ValueError(f"unknown learning-rate schedule: {kind}")

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def set_determinism(enabled: bool) -> None:
    if enabled:
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- one forward/backward pass


def forward_backward(model, batch_examples: list[dict], micro_batch_size: int, pad_id: int,
                     device: torch.device) -> tuple[float, int]:
    """Forward and backward over one effective batch; returns (mean loss per token, target tokens).

    The loss is normalized by the number of target tokens in the whole effective batch rather
    than averaged per micro-batch, so the trajectory does not depend on micro_batch_size
    (up to floating-point error).
    """
    total = sum(n_target_tokens(e) for e in batch_examples)
    if total == 0:
        return float("nan"), 0
    loss_sum = 0.0
    for i in range(0, len(batch_examples), micro_batch_size):
        b = collate(batch_examples[i:i + micro_batch_size], pad_id, device)
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits
        # The output at position k predicts token k + 1
        logits = logits[:, :-1, :]
        targets = b["labels"][:, 1:]
        sel = targets != IGNORE
        nll = F.cross_entropy(logits[sel].float(), targets[sel], reduction="sum")
        (nll / total).backward()
        loss_sum += nll.item()
    return loss_sum / total, total


# ---------------------------------------------------------------- main loop


def _truncate_log(path: Path, max_step: int) -> None:
    if not path.exists():
        return
    keep = [line for line in path.read_text().splitlines() if line and json.loads(line)["step"] <= max_step]
    path.write_text("".join(x + "\n" for x in keep))


def train_run(model, tok, lora: LoRAHandle, examples: list[dict], run_dir: str | Path, cfg: dict,
              seed: int, save_every: int, stop_after_step: int | None = None, log=print) -> str:
    """Train one trajectory. Returns "done", or "stopped" when stop_after_step is hit (used by tests)."""
    t = cfg["train"]
    d = cfg["data"]
    run_dir = Path(run_dir)
    device = next(model.parameters()).device
    set_determinism(t["deterministic"])

    encoded = [encode_example(tok, e["prompt"], e["response"], t["max_len"], d["system_prompt"],
                              d["chat_template_kwargs"]) for e in examples]
    n_empty = sum(1 for e in encoded if n_target_tokens(e) == 0)
    if n_empty:
        log(f"  warning: {n_empty} examples have no response tokens after truncation (prompt too long)")
    schedule = batch_schedule(len(encoded), t["batch_size"], t["max_steps"], seed)

    params = lora.parameters()
    opt = make_optimizer(params, t)
    sched = make_scheduler(opt, t)
    log_path = run_dir / "train_log.jsonl"
    ckpt_path = run_dir / "optim" / "latest.pt"

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        lora.load_state(state["lora"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"])
        if torch.cuda.is_available() and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        start = state["step"]
        _truncate_log(log_path, start)
        log(f"  resuming from step {start}")
    else:
        lora.reset(seed)
        torch.manual_seed(seed)  # only affects LoRA dropout (off by default)
        save_snapshot(run_dir, 0, lora.state())
        log_path.write_text("")
        start = 0

    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    t0 = time.time()
    try:
        for step in range(start + 1, t["max_steps"] + 1):
            batch = [encoded[i] for i in schedule[step - 1]]
            lr = sched.get_last_lr()[0]
            loss, n_tok = forward_backward(model, batch, t["micro_batch_size"], tok.pad_token_id, device)
            max_norm = t["grad_clip"] if t["grad_clip"] else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            record = {"step": step, "loss": loss, "grad_norm": grad_norm, "lr": lr, "n_tokens": n_tok,
                      "elapsed_s": round(time.time() - t0, 2)}
            if device.type == "cuda":
                record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            if step % save_every == 0 or step == t["max_steps"]:
                save_snapshot(run_dir, step, lora.state())
            if step % cfg["save"]["optimizer_every"] == 0:
                _save_optimizer(ckpt_path, step, lora, opt, sched)
            if step % 50 == 0 or step == t["max_steps"]:
                mem = f"  peak mem {record['peak_mem_gb']} GB" if "peak_mem_gb" in record else ""
                log(f"  step {step}/{t['max_steps']}  loss {loss:.4f}  |g| {grad_norm:.3e}  "
                    f"{(time.time() - t0) / (step - start):.2f} s/step{mem}")
            if stop_after_step is not None and step >= stop_after_step:
                return "stopped"
    finally:
        model.eval()
        if t["gradient_checkpointing"]:
            model.gradient_checkpointing_disable()

    (run_dir / "DONE").write_text("ok\n")
    return "done"


def _save_optimizer(path: Path, step: int, lora: LoRAHandle, opt, sched) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "lora": lora.state(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, tmp)
    tmp.replace(path)

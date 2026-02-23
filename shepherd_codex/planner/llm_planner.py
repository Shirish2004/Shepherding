"""LLM planner with frozen base logits, prompt design, and online adapter tuning."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .adapter_train import intent_to_idx, scene_to_tensor
from .mock_llm import DEFAULT_VOCAB, MockLLM


class AdapterMLP(nn.Module):
    """Small trainable adapter that maps scene features to intent-logit deltas."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LLMPlanner:
    """Planner with frozen rule-based base model and trainable online adapter."""

    def __init__(self, vocab=None, adapter_dim: int = 64, freeze_base: bool = True, device: str = "cpu", seed: int = 0):
        self.vocab = vocab or DEFAULT_VOCAB
        self.device = torch.device(device)
        self.base_llm = MockLLM(self.vocab)
        self.freeze_base = freeze_base
        self.adapter = AdapterMLP(6, adapter_dim, len(self.vocab)).to(self.device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.update_logs: list[dict] = []
        self.snapshots: deque[Path] = deque(maxlen=5)

    def build_prompt(self, scene_tokens: dict) -> str:
        """Build the explicit prompt format for a future real LLM backend."""
        return (
            "You are a high-level planner for StringNet herding.\n"
            "Return only JSON with keys: formation_type, params, assignments, phase, intent_token.\n"
            "Available intent_token values: "
            + ", ".join(self.vocab)
            + "\n"
            "Rules: keep defenders behind flock w.r.t. goal; prefer corrective safe intent when escape risk rises.\n"
            f"Scene tokens: {json.dumps(scene_tokens, sort_keys=True)}"
        )

    def _decode_intent(self, logits: np.ndarray, scene_tokens: dict, prompt: str) -> dict:
        idx = int(np.argmax(logits))
        tok = self.vocab[idx]
        phase = "seek"
        if tok in {"tighten_net", "focus_largest_cluster", "increase_speed"}:
            phase = "herd"
        elif tok in {"split", "delay_enclose"}:
            phase = "enclose"
        return {
            "formation_type": "semi-ellipse",
            "params": {
                "radius_scale": 0.9 if tok == "tighten_net" else 1.1 if tok == "widen_net" else 1.0,
                "speed_scale": 1.1 if tok == "increase_speed" else 0.9 if tok == "reduce_speed" else 1.0,
            },
            "assignments": {"leader": 0},
            "phase": phase,
            "intent_token": tok,
            "logits": logits.tolist(),
            "behind_hint": scene_tokens["ACoM"],
            "llm_prompt": prompt,
        }

    def plan(self, scene_tokens) -> dict:
        prompt = self.build_prompt(scene_tokens)
        base_logits = self.base_llm(scene_tokens)
        x = scene_to_tensor(scene_tokens).to(self.device)
        with torch.no_grad():
            adap = self.adapter(x).cpu().numpy()
        logits = base_logits + adap
        return self._decode_intent(logits, scene_tokens, prompt)

    def save_snapshot(self, directory: str = "shepherd_codex/checkpoints") -> Path:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        path = p / f"adapter_{ts}.pt"
        torch.save(self.adapter.state_dict(), path)
        self.snapshots.append(path)
        return path

    def load_snapshot(self, path: Path) -> None:
        self.adapter.load_state_dict(torch.load(path, map_location=self.device))

    def update_adapter(self, training_examples, lr: float = 1e-3, epochs: int = 3, batch_size: int = 16):
        xs = torch.stack([scene_to_tensor(s) for s, _ in training_examples]).to(self.device)
        ys = torch.tensor([intent_to_idx(i, self.vocab) for _, i in training_examples], dtype=torch.long, device=self.device)
        ds = TensorDataset(xs, ys)
        dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False)
        opt = torch.optim.Adam(self.adapter.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        history = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                logits = self.adapter(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach().cpu())
            history.append(epoch_loss / max(len(dl), 1))
        return {"loss_history": history}

    def logged_update(self, failing_scene: dict, oracle_intent: dict, lr: float = 5e-4, epochs: int = 3, seed: int = 0) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        pre = self.plan(failing_scene)
        snap = self.save_snapshot()
        result = self.update_adapter([(failing_scene, oracle_intent)], lr=lr, epochs=epochs)
        post = self.plan(failing_scene)
        log = {
            "seed": seed,
            "timestamp": datetime.utcnow().isoformat(),
            "failing_scene_tokens": deepcopy(failing_scene),
            "pre_plan": pre,
            "oracle_plan": oracle_intent,
            "loss_history": result["loss_history"],
            "post_plan": post,
            "snapshot": str(snap),
        }
        self.update_logs.append(log)
        Path("shepherd_codex/checkpoints").mkdir(parents=True, exist_ok=True)
        with open("shepherd_codex/checkpoints/update_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        return log

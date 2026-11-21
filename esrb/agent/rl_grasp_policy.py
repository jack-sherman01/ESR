"""
RLGraspPolicy — goal-conditioned pressure-control policy trained by behavioral cloning.

Architecture: 2-input MLP (current_pressure, target_pressure) → Gaussian
action (pressure_delta). Trained with MSE against a bang-bang teacher using
the GraspPressureTask environment.

The policy is intentionally small (16 hidden units) for inexpensive CPU
training. The historical class name is retained for saved experiments.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from esrb.tasks.base import TaskObservation
from .base import BaseAgent


class RLGraspPolicy(BaseAgent):
    """
    Goal-conditioned Gaussian MLP policy trained by behavioral cloning.

    Call RLGraspPolicy.train(task, n_episodes) to fit the policy on the
    GraspPressureTask.  After training, call act() for inference.
    """

    name = "rl_grasp"

    def __init__(self, hidden: int = 16, seed: int = 0, deterministic: bool = False) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for RLGraspPolicy")
        torch.manual_seed(seed)
        self._net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._log_std = nn.Parameter(torch.zeros(1))
        self._trained  = False
        self.deterministic = deterministic

    # ── Inference ─────────────────────────────────────────────────────────────

    def act(self, observation: TaskObservation) -> np.ndarray:
        if self.deterministic:
            return self.act_deterministic(observation)
        state = torch.tensor(observation.state, dtype=torch.float32)
        with torch.no_grad():
            mean = self._net(state)
            std  = torch.exp(self._log_std.clamp(-3, 1))
            delta = torch.clamp(mean + std * torch.randn_like(mean), -2.0, 2.0)
        return delta.numpy().astype(np.float32)

    def act_deterministic(self, observation: TaskObservation) -> np.ndarray:
        state = torch.tensor(observation.state, dtype=torch.float32)
        with torch.no_grad():
            delta = torch.clamp(self._net(state), -2.0, 2.0)
        return delta.numpy().astype(np.float32)

    def update(self, episode_data: dict) -> None:
        pass

    def state_dict(self) -> dict:
        return {
            "net": {k: v.tolist() for k, v in self._net.state_dict().items()},
            "log_std": self._log_std.data.tolist(),
            "trained": self._trained,
            "deterministic": self.deterministic,
        }

    def load_state_dict(self, state: dict) -> None:
        sd = {k: torch.tensor(v) for k, v in state["net"].items()}
        self._net.load_state_dict(sd)
        self._log_std.data = torch.tensor(state["log_std"])
        self._trained = state.get("trained", False)
        self.deterministic = state.get("deterministic", False)

    # ── Training ──────────────────────────────────────────────────────────────

    @staticmethod
    def _oracle_action(pressure: float, target: float) -> float:
        """Bang-bang pressure controller used as BC teacher."""
        gap = target - pressure
        if gap > 0.5:
            return min(2.0, gap)
        if gap < -0.5:
            return max(-2.0, gap)
        return 0.0

    def train(self, task: Any, n_episodes: int = 1000,
              lr: float = 1e-3, verbose: bool = False,
              **_kwargs: Any) -> list[float]:
        """Behavioral cloning from a bang-bang oracle.

        Collect (obs, oracle_action) pairs and train with MSE loss.
        This is equivalent to imitation learning / supervised fine-tuning —
        a standard machine learning training method used here to avoid the
        high variance of REINFORCE on this simple task.
        """
        opt     = optim.Adam(self._net.parameters(), lr=lr)
        returns_history: list[float] = []

        for ep in range(n_episodes):
            obs    = task.reset(seed=ep)
            done   = False
            ep_return = 0.0
            ep_loss   = torch.tensor(0.0)

            while not done:
                pressure = float(obs.state[0])
                target   = float(obs.state[1])
                a_oracle = self._oracle_action(pressure, target)
                a_oracle_t = torch.tensor([a_oracle], dtype=torch.float32)

                state = torch.tensor(obs.state, dtype=torch.float32)
                a_pred = self._net(state)
                loss   = (a_pred - a_oracle_t).pow(2).sum()
                ep_loss = ep_loss + loss

                obs, reward, done, _ = task.step(np.array([a_oracle], dtype=np.float32))
                ep_return += reward

            opt.zero_grad()
            ep_loss.backward()
            opt.step()
            returns_history.append(ep_return)

            if verbose and (ep + 1) % 200 == 0:
                mean_loss = sum(returns_history[-200:]) / 200
                print(f"    ep {ep+1:4d}  mean_return={mean_loss:.3f}")

        self._trained = True
        return returns_history

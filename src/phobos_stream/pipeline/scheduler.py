"""Diffusion noise and velocity schedulers (Flow Matching / Euler)."""

from typing import List, Tuple
import numpy as np


class FlowMatchingScheduler:
    """Rectified Flow / Flow Matching scheduler used in modern DiTs (Wan2.1, FLUX.1)."""

    def __init__(self, num_steps: int = 30, shift: float = 1.0) -> None:
        self.num_steps = num_steps
        self.shift = shift
        self.timesteps = self._compute_timesteps()

    def _compute_timesteps(self) -> np.ndarray:
        """Compute shifted linear timesteps from 1.0 down to 0.0."""
        t = np.linspace(1.0, 0.0, self.num_steps + 1, dtype=np.float32)
        # Apply time-shifting if shift != 1.0
        if self.shift != 1.0:
            t = (self.shift * t) / (1.0 + (self.shift - 1.0) * t)
        return t

    def step(self, model_output: np.ndarray, timestep: float, sample: np.ndarray) -> np.ndarray:
        """Euler ODE step for Flow Matching: x_{t - dt} = x_t - dt * v."""
        idx = int(np.where(np.isclose(self.timesteps, timestep, atol=1e-3))[0][0])
        dt = self.timesteps[idx] - self.timesteps[idx + 1]
        prev_sample = sample - (dt * model_output)
        return prev_sample

    def get_timesteps(self) -> List[float]:
        """Return list of timesteps for iteration."""
        return [float(t) for t in self.timesteps[:-1]]

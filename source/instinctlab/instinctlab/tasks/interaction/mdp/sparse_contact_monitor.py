from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from instinctlab.monitors import MonitorTerm

from .sparse_contact_reward import SparseContactReward

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from instinctlab.monitors import MonitorTermCfg

__all__ = ["SparseContactMapMonitorTerm"]


class SparseContactMapMonitorTerm(MonitorTerm):
    """Monitor sparse contact reward debug metrics without recomputing geometry."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reward_group_name = self.cfg.params.get("reward_group_name", "rewards")
        reward_term_name = self.cfg.params.get("reward_term_name", "sparse_contact_map")
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name, group_name=reward_group_name)
        if not isinstance(reward_term_cfg.func, SparseContactReward):
            raise TypeError(
                "SparseContactMapMonitorTerm requires a SparseContactReward manager term. "
                f"Got: {type(reward_term_cfg.func)}"
            )

        self._reward_term: SparseContactReward = reward_term_cfg.func
        self._current = {
            metric_name: torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)
            for metric_name in self._reward_term.metric_names
        }
        self._sum = {
            metric_name: torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)
            for metric_name in self._reward_term.metric_names
        }
        self._episode = {
            metric_name: torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)
            for metric_name in self._reward_term.metric_names
        }
        self._num_steps = torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)

    def update(self, dt: float):
        del dt
        metrics = self._reward_term.get_current_metrics()
        for metric_name, value in metrics.items():
            metric_value = value.detach()
            self._current[metric_name] = metric_value
            self._sum[metric_name] += metric_value
        self._num_steps += 1.0

    def reset_idx(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(self._env.num_envs, device=self.device)[env_ids]
        else:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        denom = torch.clamp(self._num_steps[env_ids_tensor], min=1.0)
        for metric_name in self._reward_term.metric_names:
            self._episode[metric_name][env_ids_tensor] = self._sum[metric_name][env_ids_tensor] / denom
            self._sum[metric_name][env_ids_tensor] = 0.0
        self._num_steps[env_ids_tensor] = 0.0

    def get_log(self, is_episode: bool = False) -> dict[str, float | torch.Tensor]:
        source = self._episode if is_episode else self._current
        return {metric_name: metric_values.mean() for metric_name, metric_values in source.items()}

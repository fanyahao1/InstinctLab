from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.manager_base import ManagerTermBase

from instinctlab.envs.mdp import BeyondConcatMotionAdaptiveWeighting, BeyondMimicAdaptiveWeighting

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg

__all__ = ["BeyondMimicAdaptiveWeighting", "BeyondConcatMotionAdaptiveWeighting", "TrackingSigmaCurriculum"]


class TrackingSigmaCurriculum(ManagerTermBase):
    """Linearly anneal reward tracking sigma across training.

    This term mutates reward term configs in-place so the corresponding Gaussian tracking rewards
    become sharper as training progresses.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.reward_group_name = cfg.params.get("reward_group_name", None)
        self.param_name = cfg.params.get("param_name", "tracking_sigma")
        self.term_names = list(cfg.params.get("term_names", []))
        if len(self.term_names) == 0:
            raise ValueError("TrackingSigmaCurriculum requires a non-empty `term_names` list.")

        self.initial_sigmas = self._expand_param(cfg.params.get("initial_sigmas", None), "initial_sigmas")
        self.final_sigmas = self._expand_param(cfg.params.get("final_sigmas", None), "final_sigmas")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        start_step: int = 0,
        end_step: int = 1_000_000,
        reward_group_name: str | None = None,
        term_names: Sequence[str] | None = None,
        param_name: str = "tracking_sigma",
        initial_sigmas: float | Sequence[float] | None = None,
        final_sigmas: float | Sequence[float] | None = None,
        min_sigma: float = 1e-3,
    ) -> dict[str, float]:
        del env_ids, term_names, param_name, initial_sigmas, final_sigmas

        reward_group_name = self.reward_group_name if reward_group_name is None else reward_group_name

        if end_step <= start_step:
            progress = 1.0 if env.common_step_counter >= start_step else 0.0
        else:
            progress = (float(env.common_step_counter) - float(start_step)) / float(end_step - start_step)
            progress = min(max(progress, 0.0), 1.0)

        log_dict: dict[str, float] = {
            "tracking_sigma_progress": progress,
        }
        for term_name, initial_sigma, final_sigma in zip(self.term_names, self.initial_sigmas, self.final_sigmas):
            current_sigma = max(initial_sigma + progress * (final_sigma - initial_sigma), min_sigma)
            term_cfg = env.reward_manager.get_term_cfg(term_name, group_name=reward_group_name)
            term_cfg.params[self.param_name] = float(current_sigma)
            log_dict[f"{term_name}_{self.param_name}"] = float(current_sigma)

        return log_dict

    def _expand_param(self, value: float | Sequence[float] | None, name: str) -> list[float]:
        if value is None:
            if name != "initial_sigmas":
                raise ValueError(f"TrackingSigmaCurriculum requires `{name}`.")
            values = []
            for term_name in self.term_names:
                term_cfg = self._env.reward_manager.get_term_cfg(term_name, group_name=self.reward_group_name)
                if self.param_name not in term_cfg.params:
                    raise ValueError(f"Reward term '{term_name}' does not contain parameter '{self.param_name}'.")
                values.append(float(term_cfg.params[self.param_name]))
            return values

        if isinstance(value, (int, float)):
            return [float(value)] * len(self.term_names)

        values = [float(v) for v in value]
        if len(values) != len(self.term_names):
            raise ValueError(
                f"TrackingSigmaCurriculum `{name}` length ({len(values)}) must match "
                f"`term_names` length ({len(self.term_names)})."
            )
        return values

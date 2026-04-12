from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from isaaclab.envs.ui import ManagerBasedRLEnvWindow

try:
    import omni.kit.app
    import omni.ui
except ModuleNotFoundError:
    omni = None

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


class InstinctLabRLEnvWindow(ManagerBasedRLEnvWindow):
    """Window manager for the RL environment.

    On top of the isaaclab manager-based RL environment window, this class adds more controls for InstinctLab-specific.
    This includes visualization of the command manager.
    """

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        self._selected_env_idx = int(getattr(env.cfg.viewer, "env_index", 0))
        self._sparse_contact_label = None
        self._sparse_contact_update_handle = None
        self._sparse_contact_reward_term = self._resolve_sparse_contact_reward_term()

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            self._build_sparse_contact_text_frame()
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._visualize_manager(title="Monitors", class_name="monitor_manager")
        self._set_sparse_contact_text_subscription(self._sparse_contact_reward_term is not None)
        self._refresh_sparse_contact_text()

    def __del__(self):
        self._set_sparse_contact_text_subscription(False)
        super().__del__()

    def _set_viewer_env_index_fn(self, model):
        self._selected_env_idx = max(model.as_int - 1, 0)
        super()._set_viewer_env_index_fn(model)
        self._refresh_sparse_contact_text()

    def _build_sparse_contact_text_frame(self) -> None:
        if omni is None:
            return

        self.ui_window_elements["sparse_contact_frame"] = omni.ui.CollapsableFrame(
            title="Sparse Contact Debug",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
        )
        with self.ui_window_elements["sparse_contact_frame"]:
            self.ui_window_elements["sparse_contact_vstack"] = omni.ui.VStack(spacing=4, height=0)
            with self.ui_window_elements["sparse_contact_vstack"]:
                self._sparse_contact_label = omni.ui.Label(
                    self._format_sparse_contact_text(),
                    alignment=omni.ui.Alignment.LEFT_TOP,
                )

    def _resolve_sparse_contact_reward_term(self):
        reward_manager = getattr(self.env, "reward_manager", None)
        if reward_manager is None or not hasattr(reward_manager, "get_term_cfg"):
            return None

        try:
            reward_term_cfg = reward_manager.get_term_cfg("sparse_contact_map", group_name="rewards")
        except Exception:
            return None

        reward_term = getattr(reward_term_cfg, "func", None)
        if reward_term is None or not callable(getattr(reward_term, "get_current_metrics", None)):
            return None
        return reward_term

    def _set_sparse_contact_text_subscription(self, enabled: bool) -> None:
        if omni is None:
            return

        if enabled:
            if self._sparse_contact_update_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._sparse_contact_update_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._on_sparse_contact_post_update(event)
                )
        elif self._sparse_contact_update_handle is not None:
            self._sparse_contact_update_handle.unsubscribe()
            self._sparse_contact_update_handle = None

    def _on_sparse_contact_post_update(self, event) -> None:
        del event
        self._refresh_sparse_contact_text()

    def _refresh_sparse_contact_text(self) -> None:
        if self._sparse_contact_label is not None:
            self._sparse_contact_label.text = self._format_sparse_contact_text()

    def _format_sparse_contact_text(self) -> str:
        reward_term = self._sparse_contact_reward_term
        if reward_term is None:
            return "Sparse contact text: reward term not active."

        metrics = reward_term.get_current_metrics()
        env_idx = min(max(self._selected_env_idx, 0), self.env.num_envs - 1)

        mandatory_distance = self._metric_value(metrics, "mandatory_distance", env_idx)
        pelvis_to_seat_distance = self._metric_value(metrics, "pelvis_to_seat_distance", env_idx)
        mandatory_reward = self._metric_value(metrics, "mandatory_reward", env_idx)
        total_contact_reward = self._metric_value(metrics, "total_contact_reward", env_idx)

        return "\n".join(
            [
                f"env[{env_idx}] sparse_contact_map",
                f"debug_vis={bool(getattr(reward_term, 'debug_vis', False))}"
                f"  show_all_points={bool(getattr(reward_term, 'debug_vis_show_all_points', False))}"
                f"  show_nearest={bool(getattr(reward_term, 'debug_vis_show_nearest', False))}",
                f"mandatory_distance={mandatory_distance:.4f} m",
                f"pelvis_to_seat_distance={pelvis_to_seat_distance:.4f} m",
                f"mandatory_reward={mandatory_reward:.4f}",
                f"total_contact_reward={total_contact_reward:.4f}",
            ]
        )

    @staticmethod
    def _metric_value(metrics: dict, key: str, env_idx: int) -> float:
        value = metrics.get(key)
        if value is None:
            return 0.0
        if hasattr(value, "numel") and value.numel() > env_idx:
            return float(value[env_idx].detach().cpu().item())
        return float(value)

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .object_utils import (
    get_filtered_contact_max_force,
    get_object_reference_contact,
    get_object_reference_state_w,
    get_object_state_w,
)


def object_position_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    object_name: str = "box",
    tracking_torlerance: float = 0.25,
    tracking_sigma: float = 0.8,
) -> torch.Tensor:
    """Gaussian tracking reward for object position with explicit tolerance."""
    object_pos_w, _, _, _ = get_object_state_w(env, asset_cfg)
    object_pos_ref_w, _, _, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    distance = torch.norm(object_pos_w - object_pos_ref_w, dim=-1)
    if tracking_torlerance > 0:
        distance = torch.clamp(distance - tracking_torlerance, min=0.0)
    rewards = torch.exp(-torch.square(distance) / (tracking_sigma * tracking_sigma))
    return rewards * validity


def object_rotation_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    object_name: str = "box",
    tracking_torlerance: float = 0.6,
    tracking_sigma: float = 1.0,
) -> torch.Tensor:
    """Gaussian tracking reward for object orientation with explicit tolerance."""
    _, object_quat_w, _, _ = get_object_state_w(env, asset_cfg)
    _, object_quat_ref_w, _, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    angle_error = math_utils.quat_error_magnitude(object_quat_ref_w, object_quat_w)
    if tracking_torlerance > 0:
        angle_error = torch.clamp(angle_error - tracking_torlerance, min=0.0)
    rewards = torch.exp(-torch.square(angle_error) / (tracking_sigma * tracking_sigma))
    return rewards * validity


def object_linear_velocity_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    object_name: str = "box",
    tracking_tolerance: float = 0.15,
    tracking_sigma: float = 1.0,
) -> torch.Tensor:
    """Gaussian tracking reward for object linear velocity."""
    _, _, lin_vel_w, _ = get_object_state_w(env, asset_cfg)
    _, _, lin_vel_ref_w, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    error = torch.norm(lin_vel_w - lin_vel_ref_w, dim=-1)
    if tracking_tolerance > 0:
        error = torch.clamp(error - tracking_tolerance, min=0.0)
    return torch.exp(-torch.square(error) / (tracking_sigma * tracking_sigma)) * validity


def object_angular_velocity_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    object_name: str = "box",
    tracking_tolerance: float = 0.25,
    tracking_sigma: float = 2.0,
) -> torch.Tensor:
    """Gaussian tracking reward for object angular velocity."""
    _, _, _, ang_vel_w = get_object_state_w(env, asset_cfg)
    _, _, _, ang_vel_ref_w, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    error = torch.norm(ang_vel_w - ang_vel_ref_w, dim=-1)
    if tracking_tolerance > 0:
        error = torch.clamp(error - tracking_tolerance, min=0.0)
    return torch.exp(-torch.square(error) / (tracking_sigma * tracking_sigma)) * validity


def object_contact_reference_phase(
    env: ManagerBasedRLEnv,
    sensor_names: list[str] | None = None,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "box",
    threshold: float = 3.0,
    normalize: bool = True,
    penetration_penalty_scale: float = 1.0,
    print_reason: bool = False,
    debug_label: str = "object_contact",
) -> torch.Tensor:
    """Piecewise body-object reward based on reference contact phase."""
    if sensor_names is None:
        sensor_names = []
    if len(sensor_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    current_contact = torch.stack(
        [(get_filtered_contact_max_force(env, SceneEntityCfg(sensor_name)) > threshold).float() for sensor_name in sensor_names],
        dim=-1,
    )

    reference_contact, validity = get_object_reference_contact(env, reference_cfg=reference_cfg, object_name=object_name)
    contact_score = current_contact.sum(dim=-1)
    reward = (reference_contact * contact_score) - ((1.0 - reference_contact) * penetration_penalty_scale * contact_score)
    reward = reward * validity
    if normalize and current_contact.shape[-1] > 0:
        reward = reward / current_contact.shape[-1]
    if print_reason and int(env.common_step_counter) % 50 == 0:
        sensor_force_max = {
            sensor_name: get_filtered_contact_max_force(env, SceneEntityCfg(sensor_name)).max().item()
            for sensor_name in sensor_names
        }
        print(
            f"{debug_label}_debug:",
            f"step={int(env.common_step_counter)}",
            f"sensor_filtered_max={sensor_force_max}",
            f"current_contact_mean={current_contact.mean(dim=0).tolist()}",
            f"reference_contact_mean={reference_contact.float().mean().item():.4f}",
        )
    return reward

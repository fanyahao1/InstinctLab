from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .object_utils import get_object_reference_state_w, get_object_state_w


def object_position_far_from_reference(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    object_name: str = "box",
    distance_threshold: float = 1.0,
    print_reason: bool = False,
) -> torch.Tensor:
    """Terminate when object position diverges too far from the reference."""
    object_pos_w, _, _, _ = get_object_state_w(env, asset_cfg)
    object_pos_ref_w, _, _, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    terminated = torch.norm(object_pos_w - object_pos_ref_w, dim=-1) > distance_threshold
    terminated = terminated & (validity > 0)
    if print_reason and torch.any(terminated):
        print("Termination: object_position_far_from_reference")
    return terminated


def object_orientation_far_from_reference(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    object_name: str = "box",
    angle_threshold: float = 2.2,
    print_reason: bool = False,
) -> torch.Tensor:
    """Terminate when object orientation diverges too far from the reference."""
    _, object_quat_w, _, _ = get_object_state_w(env, asset_cfg)
    _, object_quat_ref_w, _, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    terminated = math_utils.quat_error_magnitude(object_quat_ref_w, object_quat_w) > angle_threshold
    terminated = terminated & (validity > 0)
    if print_reason and torch.any(terminated):
        print("Termination: object_orientation_far_from_reference")
    return terminated


def object_velocity_too_large(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    max_linear_speed: float = 8.0,
    max_angular_speed: float = 20.0,
    print_reason: bool = False,
) -> torch.Tensor:
    """Terminate when object is ejected with implausibly high speed."""
    _, _, lin_vel_w, ang_vel_w = get_object_state_w(env, asset_cfg)
    terminated = (torch.norm(lin_vel_w, dim=-1) > max_linear_speed) | (torch.norm(ang_vel_w, dim=-1) > max_angular_speed)
    if print_reason and torch.any(terminated):
        print("Termination: object_velocity_too_large")
    return terminated

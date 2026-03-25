from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

import instinctlab.utils.math as instinct_math

from .object_utils import get_filtered_contact_max_force, get_object_reference_state_w, get_object_state_w, safe_unit_quat


def object_position(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
) -> torch.Tensor:
    """Object position in world or robot-base frame."""
    object_pos_w, _, _, _ = get_object_state_w(env, asset_cfg)
    if not in_base_frame:
        return object_pos_w

    robot: Articulation = env.scene[robot_cfg.name]
    robot_quat_w = safe_unit_quat(robot.data.root_quat_w)
    robot_pos_w = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    return math_utils.quat_rotate_inverse(robot_quat_w, object_pos_w - robot_pos_w)


def object_orientation_tannorm(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
) -> torch.Tensor:
    """Object orientation in tangent-normal representation."""
    _, object_quat_w, _, _ = get_object_state_w(env, asset_cfg)
    if not in_base_frame:
        return instinct_math.quat_to_tan_norm(object_quat_w)

    robot: Articulation = env.scene[robot_cfg.name]
    robot_quat_w = safe_unit_quat(robot.data.root_quat_w)
    object_quat_b = math_utils.quat_mul(math_utils.quat_inv(robot_quat_w), object_quat_w)
    return instinct_math.quat_to_tan_norm(object_quat_b)


def object_reference_position(
    env: ManagerBasedEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_name: str = "box",
    in_base_frame: bool = True,
) -> torch.Tensor:
    """Reference object position in world or robot-base frame."""
    object_pos_ref_w, _, _, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    if not in_base_frame:
        return object_pos_ref_w * validity.unsqueeze(-1)

    robot: Articulation = env.scene[robot_cfg.name]
    robot_quat_w = safe_unit_quat(robot.data.root_quat_w)
    robot_pos_w = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    object_pos_ref_b = math_utils.quat_rotate_inverse(robot_quat_w, object_pos_ref_w - robot_pos_w)
    return object_pos_ref_b * validity.unsqueeze(-1)


def object_reference_orientation_tannorm(
    env: ManagerBasedEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_name: str = "box",
    in_base_frame: bool = True,
) -> torch.Tensor:
    """Reference object orientation in tangent-normal representation."""
    _, object_quat_ref_w, _, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    if not in_base_frame:
        return instinct_math.quat_to_tan_norm(object_quat_ref_w) * validity.unsqueeze(-1)

    robot: Articulation = env.scene[robot_cfg.name]
    robot_quat_w = safe_unit_quat(robot.data.root_quat_w)
    object_quat_ref_b = math_utils.quat_mul(math_utils.quat_inv(robot_quat_w), object_quat_ref_w)
    return instinct_math.quat_to_tan_norm(object_quat_ref_b) * validity.unsqueeze(-1)


def object_position_error(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_name: str = "box",
    in_base_frame: bool = True,
) -> torch.Tensor:
    """Object position error against reference pose."""
    object_pos_w, _, _, _ = get_object_state_w(env, asset_cfg)
    object_pos_ref_w, _, _, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    pos_error_w = object_pos_w - object_pos_ref_w
    if not in_base_frame:
        return pos_error_w * validity.unsqueeze(-1)

    robot: Articulation = env.scene[robot_cfg.name]
    robot_quat_w = safe_unit_quat(robot.data.root_quat_w)
    return math_utils.quat_rotate_inverse(robot_quat_w, pos_error_w) * validity.unsqueeze(-1)


def object_orientation_error_tannorm(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    object_name: str = "box",
) -> torch.Tensor:
    """Object orientation error encoded in tangent-normal form."""
    _, object_quat_w, _, _ = get_object_state_w(env, asset_cfg)
    _, object_quat_ref_w, _, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    quat_error = math_utils.quat_mul(math_utils.quat_inv(object_quat_ref_w), object_quat_w)
    return instinct_math.quat_to_tan_norm(quat_error) * validity.unsqueeze(-1)


def object_linear_velocity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
) -> torch.Tensor:
    """Object linear velocity in world or robot base frame."""
    _, _, lin_vel_w, _ = get_object_state_w(env, asset_cfg)
    if not in_base_frame:
        return lin_vel_w

    robot: Articulation = env.scene[robot_cfg.name]
    robot_quat_w = safe_unit_quat(robot.data.root_quat_w)
    return math_utils.quat_rotate_inverse(robot_quat_w, lin_vel_w)


def object_angular_velocity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
) -> torch.Tensor:
    """Object angular velocity in world or robot base frame."""
    _, _, _, ang_vel_w = get_object_state_w(env, asset_cfg)
    if not in_base_frame:
        return ang_vel_w

    robot: Articulation = env.scene[robot_cfg.name]
    robot_quat_w = safe_unit_quat(robot.data.root_quat_w)
    return math_utils.quat_rotate_inverse(robot_quat_w, ang_vel_w)


def object_linear_velocity_error(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_name: str = "box",
    in_base_frame: bool = True,
) -> torch.Tensor:
    """Object linear velocity error against reference."""
    _, _, lin_vel_w, _ = get_object_state_w(env, asset_cfg)
    _, _, lin_vel_ref_w, _, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    vel_error_w = lin_vel_w - lin_vel_ref_w
    if not in_base_frame:
        return vel_error_w * validity.unsqueeze(-1)

    robot: Articulation = env.scene[robot_cfg.name]
    robot_quat_w = safe_unit_quat(robot.data.root_quat_w)
    return math_utils.quat_rotate_inverse(robot_quat_w, vel_error_w) * validity.unsqueeze(-1)


def object_angular_velocity_error(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_name: str = "box",
    in_base_frame: bool = True,
) -> torch.Tensor:
    """Object angular velocity error against reference."""
    _, _, _, ang_vel_w = get_object_state_w(env, asset_cfg)
    _, _, _, ang_vel_ref_w, validity = get_object_reference_state_w(
        env, reference_cfg=reference_cfg, reference_asset_cfg=reference_asset_cfg, object_name=object_name
    )
    ang_error_w = ang_vel_w - ang_vel_ref_w
    if not in_base_frame:
        return ang_error_w * validity.unsqueeze(-1)

    robot: Articulation = env.scene[robot_cfg.name]
    robot_quat_w = safe_unit_quat(robot.data.root_quat_w)
    return math_utils.quat_rotate_inverse(robot_quat_w, ang_error_w) * validity.unsqueeze(-1)


def wrist_object_contact(
    env: ManagerBasedEnv,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_wrist_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_wrist_object_contact"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary wrist-object contact observation for left/right wrists."""
    left_contact = (get_filtered_contact_max_force(env, left_sensor_cfg) > threshold).float()
    right_contact = (get_filtered_contact_max_force(env, right_sensor_cfg) > threshold).float()
    return torch.stack([left_contact, right_contact], dim=-1)


def selected_object_contact(
    env: ManagerBasedEnv,
    sensor_names: list[str] | None = None,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary object-contact observation for a selected list of body contact sensors."""
    if sensor_names is None:
        sensor_names = []
    if len(sensor_names) == 0:
        return torch.zeros((env.num_envs, 0), device=env.device)

    contacts = [
        (get_filtered_contact_max_force(env, SceneEntityCfg(sensor_name)) > threshold).float()
        for sensor_name in sensor_names
    ]
    return torch.stack(contacts, dim=-1)


def seat_object_contact(
    env: ManagerBasedEnv,
    sensor_names: list[str] | None = None,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary chair-contact observation for seated body parts."""
    if sensor_names is None:
        sensor_names = [
            "pelvis_object_contact",
            "left_hip_object_contact",
            "right_hip_object_contact",
        ]
    return selected_object_contact(env, sensor_names=sensor_names, threshold=threshold)

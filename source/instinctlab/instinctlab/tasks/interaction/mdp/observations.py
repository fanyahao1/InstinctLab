from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

import instinctlab.utils.math as instinct_math

from .object_utils import get_object_reference_state_w, get_object_state_w, safe_unit_quat


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
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("hand_object_contact"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary wrist-object contact observation for left/right wrists."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return (torch.norm(contact_forces, dim=-1).max(dim=1)[0] > threshold).float()

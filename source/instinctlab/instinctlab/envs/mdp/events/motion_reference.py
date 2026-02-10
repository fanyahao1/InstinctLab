from __future__ import annotations

import math
import torch
import tqdm
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from instinctlab.motion_reference import MotionReferenceManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from instinctlab.envs.mdp import BeyondMimicAdaptiveWeighting
    from instinctlab.motion_reference import MotionReferenceData


def virtualize_articulation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot_reference"),
):
    """Virtualize the robot reference (Articulation) in the simulation by removing all
    their collision shapes and setting their mass to zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids (num_envs)
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, dtype=torch.int, device="cpu")
    else:
        env_ids = env_ids.cpu().to(torch.int)

    # select all bodies (num_bodies)
    body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")

    # build meshgrid for all environment and body ids
    env_ids, body_ids = torch.meshgrid(env_ids, body_ids)

    # get the current masses of the bodies (num_assets, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # set the gravity of the bodies to zero
    asset.root_physx_view.set_disable_gravities(torch.ones_like(masses).to(torch.bool), env_ids)


def match_motion_ref_with_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Match information between motion reference and the scene, use at 'startup' stage."""
    print("[Event] Match motion reference with scene.")
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    motion_ref.match_scene(env.scene)


def reset_robot_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_name: str | None = None,
    position_offset: list = [0.0, 0.0, 0.1],
    dof_vel_ratio: float = 0.5,
    base_lin_vel_ratio: float = 0.5,
    base_ang_vel_ratio: float = 0.5,
    randomize_pose_range: dict[str, tuple[float, float]] = {},
    randomize_velocity_range: dict[str, tuple[float, float]] = {},
    randomize_joint_pos_range: tuple[float, float] = (0.0, 0.0),
):
    """Reset robot or object state based on motion reference with optional randomization.

    This function supports both Articulation (robots) and RigidObject (objects) assets.
    For robots, it uses motion_ref_init_state from the motion reference.
    For objects, it reads from motion_ref.data.object_data[object_name].

    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset.
        motion_ref_cfg: Motion reference configuration.
        asset_cfg: Asset configuration (robot or object).
        object_name: Object name to read from object_data. Required if asset is RigidObject. **Must match the key in object_data_keys.**
        position_offset: Position offset to apply [x, y, z].
        dof_vel_ratio: Ratio to scale joint velocities (robot only).
        base_lin_vel_ratio: Ratio to scale base/object linear velocities.
        base_ang_vel_ratio: Ratio to scale base/object angular velocities.
        randomize_pose_range: Optional pose randomization ranges for ["x", "y", "z", "roll", "pitch", "yaw"].
        randomize_velocity_range: Optional velocity randomization ranges for ["x", "y", "z", "roll", "pitch", "yaw"].
        randomize_joint_pos_range: Optional joint position randomization range (min, max) (robot only).
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]

    if isinstance(asset, RigidObject):
        _reset_object_state_by_reference(
            env,
            env_ids,
            asset,
            motion_ref,
            object_name,
            position_offset,
            base_lin_vel_ratio,
            base_ang_vel_ratio,
            randomize_pose_range,
            randomize_velocity_range,
        )
    elif isinstance(asset, Articulation):
        _reset_articulation_state_by_reference(
            env,
            env_ids,
            asset,
            motion_ref,
            position_offset,
            dof_vel_ratio,
            base_lin_vel_ratio,
            base_ang_vel_ratio,
            randomize_pose_range,
            randomize_velocity_range,
            randomize_joint_pos_range,
        )
    else:
        raise ValueError(f"Unsupported asset type: {type(asset)}")


def _reset_object_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset: RigidObject,
    motion_ref: MotionReferenceManager,
    object_name: str | None,
    position_offset: list,
    lin_vel_ratio: float,
    ang_vel_ratio: float,
    randomize_pose_range: dict[str, tuple[float, float]],
    randomize_velocity_range: dict[str, tuple[float, float]],
):
    """Reset object state based on object_data in motion reference."""

    if object_name is None:
        raise ValueError("object_name must be provided when resetting RigidObject")

    data: MotionReferenceData = motion_ref.data

    if not hasattr(data, "object_data") or object_name not in data.object_data:
        raise ValueError(
            f"Object '{object_name}' not found in motion reference data. "
            f"Available objects: {list(data.object_data.keys()) if hasattr(data, 'object_data') else []}"
        )

    object_data = data.object_data[object_name]

    if "pos" not in object_data:
        raise ValueError(f"Object '{object_name}' does not have position data")

    object_pos = object_data["pos"][env_ids, 0, :].clone()
    object_pos += torch.tensor(position_offset, device=object_pos.device).unsqueeze(0)

    if "quat" in object_data and object_data["quat"] is not None:
        object_quat = object_data["quat"][env_ids, 0, :].clone()
    else:
        object_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=object_pos.device).repeat(len(env_ids), 1)

    if randomize_pose_range:
        range_list = [randomize_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=object_pos.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object_pos.device
        )

        object_pos += rand_samples[:, 0:3]

        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        object_quat = math_utils.quat_mul(orientations_delta, object_quat)

    asset.write_root_pose_to_sim(torch.cat([object_pos, object_quat], dim=-1), env_ids=env_ids)

    if "lin_vel" in object_data and object_data["lin_vel"] is not None:
        object_lin_vel = object_data["lin_vel"][env_ids, 0, :].clone() * lin_vel_ratio
    else:
        object_lin_vel = torch.zeros(len(env_ids), 3, device=object_pos.device)

    if "ang_vel" in object_data and object_data["ang_vel"] is not None:
        object_ang_vel = object_data["ang_vel"][env_ids, 0, :].clone() * ang_vel_ratio
    else:
        object_ang_vel = torch.zeros(len(env_ids), 3, device=object_pos.device)

    if randomize_velocity_range:
        range_list = [randomize_velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=object_pos.device)
        vel_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object_pos.device)

        object_lin_vel += vel_samples[:, 0:3]
        object_ang_vel += vel_samples[:, 3:6]

    asset.write_root_velocity_to_sim(torch.cat([object_lin_vel, object_ang_vel], dim=-1), env_ids=env_ids)


def _reset_articulation_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset: Articulation,
    motion_ref: MotionReferenceManager,
    position_offset: list,
    dof_vel_ratio: float,
    base_lin_vel_ratio: float,
    base_ang_vel_ratio: float,
    randomize_pose_range: dict[str, tuple[float, float]],
    randomize_velocity_range: dict[str, tuple[float, float]],
    randomize_joint_pos_range: tuple[float, float],
):
    """Reset articulation (robot) state based on motion reference."""
    # reset the motion reference object
    # motion reference (as sensor) is already reset(ed) in scene.reset(...)
    # motion_ref.reset(env_ids)

    # reset the robot state based on motion reference data
    data: MotionReferenceData = motion_ref.data  # triggering the motion reference to update the data.
    motion_ref_init_state = motion_ref.get_init_reference_state(env_ids)

    # reset the robot state
    base_pos_w = motion_ref_init_state.base_pos_w
    base_pos_w = motion_ref_init_state.base_pos_w + torch.tensor(
        position_offset, device=motion_ref_init_state.base_pos_w.device
    ).unsqueeze(0)

    # apply randomizations on the reset states if specified
    if randomize_pose_range:
        # Apply pose randomization similar to reset_root_state_uniform
        range_list = [randomize_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=base_pos_w.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=base_pos_w.device
        )

        # Add position randomization
        base_pos_w += rand_samples[:, 0:3]

        # Add orientation randomization
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        base_quat_w = math_utils.quat_mul(orientations_delta, motion_ref_init_state.base_quat_w)
    else:
        base_quat_w = motion_ref_init_state.base_quat_w

    # write the root pose to the simulation
    asset.write_root_pose_to_sim(
        torch.cat(
            [
                base_pos_w,
                base_quat_w,
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )

    # apply velocity randomization if specified
    base_lin_vel_w = motion_ref_init_state.base_lin_vel_w * base_lin_vel_ratio
    base_ang_vel_w = motion_ref_init_state.base_ang_vel_w * base_ang_vel_ratio
    if randomize_velocity_range:
        # Apply velocity randomization
        range_list = [randomize_velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=base_lin_vel_w.device)
        vel_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=base_lin_vel_w.device
        )

        # Add velocity randomization
        base_lin_vel_w += vel_samples[:, 0:3]
        base_ang_vel_w += vel_samples[:, 3:6]

    # write the root velocity to the simulation
    asset.write_root_velocity_to_sim(
        torch.cat(
            [
                base_lin_vel_w,
                base_ang_vel_w,
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )

    # apply joint position randomization if specified
    joint_pos = motion_ref_init_state.joint_pos
    joint_vel = motion_ref_init_state.joint_vel * dof_vel_ratio
    if randomize_joint_pos_range != (0.0, 0.0):
        # Apply joint position randomization
        joint_pos_noise = math_utils.sample_uniform(
            randomize_joint_pos_range[0], randomize_joint_pos_range[1], joint_pos.shape, device=joint_pos.device
        )
        joint_pos += joint_pos_noise

    # write the joint state to the simulation
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
    )


def beyondmimic_bin_fail_counter_smoothing(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    curriculum_name: str,
):
    # Acquire the curriculum term instance, which should be a ManagerTermBase
    curriculum: BeyondMimicAdaptiveWeighting = env.curriculum_manager._term_cfgs[
        env.curriculum_manager._term_names.index(curriculum_name)
    ].func

    curriculum.motion_bin_fail_counter._concatenated_tensor.mul_(1 - curriculum.adaptive_alpha)
    curriculum.motion_bin_fail_counter._concatenated_tensor.add_(
        curriculum.adaptive_alpha * curriculum.current_motion_bin_fail_counter._concatenated_tensor
    )
    curriculum.current_motion_bin_fail_counter._concatenated_tensor.fill_(0)

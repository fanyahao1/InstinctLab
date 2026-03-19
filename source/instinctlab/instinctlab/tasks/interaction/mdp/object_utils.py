from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def safe_unit_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return finite, unit quaternions with identity fallback."""
    quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    quat = quat / torch.clamp(norm, min=1e-6)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where((norm > 1e-6).expand_as(quat), quat, identity)


def get_object_state_w(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("objects")) -> tuple[torch.Tensor, ...]:
    """Read the simulated object state in world frame."""
    asset = env.scene[asset_cfg.name]
    pos_w = torch.nan_to_num(asset.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    quat_w = safe_unit_quat(asset.data.root_quat_w)
    lin_vel_w = torch.nan_to_num(asset.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    ang_vel_w = torch.nan_to_num(asset.data.root_ang_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    return pos_w, quat_w, lin_vel_w, ang_vel_w


def get_object_reference_state_w(
    env,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reference_asset_cfg: SceneEntityCfg | None = None,
    object_name: str = "box",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the object reference state from either the motion reference or a scene object."""
    if reference_asset_cfg is not None:
        asset = env.scene[reference_asset_cfg.name]
        pos_w = torch.nan_to_num(asset.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        quat_w = safe_unit_quat(asset.data.root_quat_w)
        lin_vel_w = torch.nan_to_num(asset.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
        ang_vel_w = torch.nan_to_num(asset.data.root_ang_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
        validity = torch.ones(pos_w.shape[0], device=pos_w.device, dtype=pos_w.dtype)
        return pos_w, quat_w, lin_vel_w, ang_vel_w, validity

    motion_reference = env.scene[reference_cfg.name]
    frame_ids = motion_reference.aiming_frame_idx
    env_ids = motion_reference.ALL_INDICES
    object_data = getattr(motion_reference.data, "object_data", {})

    if object_name not in object_data:
        num_envs = motion_reference.data.base_pos_w.shape[0]
        device = motion_reference.data.base_pos_w.device
        dtype = motion_reference.data.base_pos_w.dtype
        pos_w = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        quat_w = torch.zeros((num_envs, 4), device=device, dtype=dtype)
        quat_w[:, 0] = 1.0
        lin_vel_w = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        ang_vel_w = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        validity = torch.zeros((num_envs,), device=device, dtype=dtype)
        return pos_w, quat_w, lin_vel_w, ang_vel_w, validity

    data = object_data[object_name]
    pos_w = torch.nan_to_num(data["pos"][env_ids, frame_ids], nan=0.0, posinf=0.0, neginf=0.0)
    quat_src = data.get("quat")
    if quat_src is None:
        quat_w = torch.zeros((pos_w.shape[0], 4), device=pos_w.device, dtype=pos_w.dtype)
        quat_w[:, 0] = 1.0
    else:
        quat_w = safe_unit_quat(quat_src[env_ids, frame_ids])
    lin_vel_src = data.get("lin_vel")
    ang_vel_src = data.get("ang_vel")
    lin_vel_w = (
        torch.nan_to_num(lin_vel_src[env_ids, frame_ids], nan=0.0, posinf=0.0, neginf=0.0)
        if lin_vel_src is not None
        else torch.zeros_like(pos_w)
    )
    ang_vel_w = (
        torch.nan_to_num(ang_vel_src[env_ids, frame_ids], nan=0.0, posinf=0.0, neginf=0.0)
        if ang_vel_src is not None
        else torch.zeros_like(pos_w)
    )

    validity = motion_reference.data.validity[env_ids, frame_ids].to(torch.bool)
    validity = validity & torch.isfinite(pos_w).all(dim=-1) & torch.isfinite(quat_w).all(dim=-1)
    validity = validity & torch.isfinite(lin_vel_w).all(dim=-1) & torch.isfinite(ang_vel_w).all(dim=-1)
    return pos_w, quat_w, lin_vel_w, ang_vel_w, validity.to(pos_w.dtype)


def get_object_reference_contact(
    env,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    object_name: str = "box",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the scalar object-contact phase from the motion reference."""
    motion_reference = env.scene[reference_cfg.name]
    frame_ids = motion_reference.aiming_frame_idx
    env_ids = motion_reference.ALL_INDICES
    object_data = getattr(motion_reference.data, "object_data", {})

    if object_name not in object_data or "contact" not in object_data[object_name]:
        num_envs = motion_reference.data.base_pos_w.shape[0]
        device = motion_reference.data.base_pos_w.device
        dtype = motion_reference.data.base_pos_w.dtype
        zeros = torch.zeros((num_envs,), device=device, dtype=dtype)
        return zeros, zeros

    contact = torch.nan_to_num(
        object_data[object_name]["contact"][env_ids, frame_ids],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    validity = motion_reference.data.validity[env_ids, frame_ids].to(contact.dtype)
    return contact, validity

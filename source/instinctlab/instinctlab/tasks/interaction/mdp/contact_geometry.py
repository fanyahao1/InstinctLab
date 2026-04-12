from __future__ import annotations

import torch

__all__ = [
    "compute_link_part_center_distance",
    "compute_link_part_nearest_distance",
    "transform_part_centers_to_world",
    "transform_part_points_to_world",
]


def transform_part_points_to_world(
    points_local: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Transform batched part points from object-local frame to world frame.

    Args:
        points_local: Tensor of shape ``(N, P, K, 3)``.
        object_pos_w: Tensor of shape ``(N, 3)``.
        object_quat_w: Tensor of shape ``(N, 4)`` in ``(w, x, y, z)`` order.
    """
    num_envs, num_parts, num_points, _ = points_local.shape
    expanded_quat = object_quat_w[:, None, None, :].expand(num_envs, num_parts, num_points, 4)
    rotated_points = _quat_apply(expanded_quat.reshape(-1, 4), points_local.reshape(-1, 3)).reshape_as(points_local)
    return rotated_points + object_pos_w[:, None, None, :]


def transform_part_centers_to_world(
    centers_local: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Transform batched part centers from object-local frame to world frame."""
    num_envs, num_parts, _ = centers_local.shape
    expanded_quat = object_quat_w[:, None, :].expand(num_envs, num_parts, 4)
    rotated_centers = _quat_apply(expanded_quat.reshape(-1, 4), centers_local.reshape(-1, 3)).reshape_as(centers_local)
    return rotated_centers + object_pos_w[:, None, :]


def compute_link_part_nearest_distance(
    link_pos_w: torch.Tensor,
    part_points_w: torch.Tensor,
    point_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute nearest-point distance from every link to every object part."""
    point_distances = torch.linalg.vector_norm(
        link_pos_w[:, :, None, None, :] - part_points_w[:, None, :, :, :],
        dim=-1,
    )
    if point_valid_mask is not None:
        point_distances = point_distances.masked_fill(~point_valid_mask[:, None, :, :], torch.inf)
    return point_distances.min(dim=-1).values


def compute_link_part_center_distance(
    link_pos_w: torch.Tensor,
    part_centers_w: torch.Tensor,
) -> torch.Tensor:
    """Compute center distance from every link to every object part."""
    return torch.linalg.vector_norm(link_pos_w[:, :, None, :] - part_centers_w[:, None, :, :], dim=-1)


def _quat_apply(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in (w, x, y, z) convention."""
    qw = quat_wxyz[..., 0:1]
    q_xyz = quat_wxyz[..., 1:]
    cross_1 = torch.cross(q_xyz, vec, dim=-1)
    cross_2 = torch.cross(q_xyz, cross_1, dim=-1)
    return vec + 2.0 * (qw * cross_1 + cross_2)

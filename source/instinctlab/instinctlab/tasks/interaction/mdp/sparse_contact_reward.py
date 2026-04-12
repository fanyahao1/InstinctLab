from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch

from .contact_geometry import (
    compute_link_part_center_distance,
    compute_link_part_nearest_distance,
    transform_part_centers_to_world,
    transform_part_points_to_world,
)
from .contact_map_loader import SparseContactMapMetadata, load_sparse_contact_map_directory
from .object_utils import get_current_object_keys, get_current_object_usd_basenames, get_object_state_w

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.managers import ManagerTermBase, SceneEntityCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
except ImportError:  # pragma: no cover - only used for lightweight unit tests
    sim_utils = None
    Articulation = Any
    VisualizationMarkers = None
    VisualizationMarkersCfg = None
    BLUE_ARROW_X_MARKER_CFG = None
    RED_ARROW_X_MARKER_CFG = None

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.device = getattr(env, "device", "cpu")

    @dataclass
    class SceneEntityCfg:
        name: str

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

__all__ = [
    "MandatoryContactDebugData",
    "SlidingWindowContactBuffer",
    "SparseContactReward",
    "SparseContactRewardComponents",
    "bucketize_distance_visualization",
    "compute_sparse_contact_reward_components",
    "extract_mandatory_contact_debug_data",
    "get_visualizer_default_scale",
    "resolve_direction_to_arrow_marker",
]


@dataclass
class SparseContactRewardComponents:
    """Per-step reward breakdown and pairwise contact data."""

    total: torch.Tensor
    mandatory: torch.Tensor
    optional: torch.Tensor
    forbidden: torch.Tensor
    mandatory_contact: torch.Tensor
    optional_contact: torch.Tensor
    forbidden_contact: torch.Tensor


@dataclass
class MandatoryContactDebugData:
    """Flattened debug visualization data for mandatory link-part point pairs."""

    all_point_positions: torch.Tensor
    all_arrow_start_positions: torch.Tensor
    all_arrow_directions: torch.Tensor
    nearest_point_positions: torch.Tensor
    nearest_arrow_start_positions: torch.Tensor
    nearest_arrow_directions: torch.Tensor


class SlidingWindowContactBuffer:
    """Exact sliding-window mean over pairwise pseudo-contact signals."""

    def __init__(self, num_envs: int, num_links: int, num_parts: int, window: int, device: str | torch.device):
        if window <= 0:
            raise ValueError("SlidingWindowContactBuffer requires window > 0.")
        self.window = int(window)
        self._history = torch.zeros(self.window, num_envs, num_links, num_parts, dtype=torch.float32, device=device)
        self._counts = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._cursor = 0

    def update(self, pseudo_contact: torch.Tensor) -> torch.Tensor:
        self._history[self._cursor] = pseudo_contact.float()
        self._cursor = (self._cursor + 1) % self.window
        self._counts = torch.clamp(self._counts + 1.0, max=float(self.window))
        denom = torch.clamp(self._counts[:, None, None], min=1.0)
        return self._history.sum(dim=0) / denom

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._history.zero_()
            self._counts.zero_()
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self._history.device)
        self._history[:, env_ids_tensor] = 0.0
        self._counts[env_ids_tensor] = 0.0


def extract_mandatory_contact_debug_data(
    link_pos_w: torch.Tensor,
    part_points_w: torch.Tensor,
    relation: torch.Tensor,
    exists_mask: torch.Tensor,
    point_valid_mask: torch.Tensor,
    max_envs: int | None = None,
) -> MandatoryContactDebugData:
    """Extract flattened mandatory point and nearest-point debug geometry."""
    if relation.ndim == 2:
        relation = relation.unsqueeze(0).expand(link_pos_w.shape[0], -1, -1)
    if exists_mask.ndim == 1:
        exists_mask = exists_mask.unsqueeze(0).expand(link_pos_w.shape[0], -1)
    if point_valid_mask.ndim == 2:
        point_valid_mask = point_valid_mask.unsqueeze(0).expand(link_pos_w.shape[0], -1, -1)

    if max_envs is not None:
        max_envs = max(int(max_envs), 0)
        link_pos_w = link_pos_w[:max_envs]
        part_points_w = part_points_w[:max_envs]
        relation = relation[:max_envs]
        exists_mask = exists_mask[:max_envs]
        point_valid_mask = point_valid_mask[:max_envs]

    if link_pos_w.shape[0] == 0:
        return _empty_mandatory_contact_debug_data(device=link_pos_w.device)

    mandatory_pair_mask = (relation == 1) & exists_mask[:, None, :]
    mandatory_point_mask = mandatory_pair_mask[:, :, :, None] & point_valid_mask[:, None, :, :]

    expanded_start_positions = link_pos_w[:, :, None, None, :].expand(
        -1, -1, part_points_w.shape[1], part_points_w.shape[2], -1
    )
    expanded_part_points = part_points_w[:, None, :, :, :].expand(
        -1, link_pos_w.shape[1], -1, -1, -1
    )

    if mandatory_point_mask.any():
        all_arrow_start_positions = expanded_start_positions[mandatory_point_mask]
        all_point_positions = expanded_part_points[mandatory_point_mask]
        all_arrow_directions = all_point_positions - all_arrow_start_positions
    else:
        empty = torch.zeros((0, 3), dtype=link_pos_w.dtype, device=link_pos_w.device)
        all_arrow_start_positions = empty
        all_point_positions = empty
        all_arrow_directions = empty

    point_distances = torch.linalg.vector_norm(expanded_part_points - expanded_start_positions, dim=-1)
    point_distances = point_distances.masked_fill(~mandatory_point_mask, torch.inf)
    nearest_point_indices = point_distances.argmin(dim=-1)
    valid_nearest_mask = torch.isfinite(point_distances).any(dim=-1)

    if valid_nearest_mask.any():
        env_ids, link_ids, part_ids = valid_nearest_mask.nonzero(as_tuple=True)
        point_ids = nearest_point_indices[env_ids, link_ids, part_ids]
        nearest_point_positions = part_points_w[env_ids, part_ids, point_ids]
        nearest_arrow_start_positions = link_pos_w[env_ids, link_ids]
        nearest_arrow_directions = nearest_point_positions - nearest_arrow_start_positions
    else:
        empty = torch.zeros((0, 3), dtype=link_pos_w.dtype, device=link_pos_w.device)
        nearest_point_positions = empty
        nearest_arrow_start_positions = empty
        nearest_arrow_directions = empty

    return MandatoryContactDebugData(
        all_point_positions=all_point_positions,
        all_arrow_start_positions=all_arrow_start_positions,
        all_arrow_directions=all_arrow_directions,
        nearest_point_positions=nearest_point_positions,
        nearest_arrow_start_positions=nearest_arrow_start_positions,
        nearest_arrow_directions=nearest_arrow_directions,
    )


def resolve_direction_to_arrow_marker(
    direction: torch.Tensor,
    start_point: torch.Tensor,
    default_scale: Sequence[float] = (1.0, 1.0, 1.0),
    length_scale_factor: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve arrow marker scale, orientation and position from start-point directions."""
    if direction.shape != start_point.shape:
        raise ValueError(
            "resolve_direction_to_arrow_marker requires matching direction and start_point shapes, "
            f"got {tuple(direction.shape)} and {tuple(start_point.shape)}."
        )
    if direction.numel() == 0:
        empty = torch.zeros((0, 3), dtype=start_point.dtype, device=start_point.device)
        empty_quat = torch.zeros((0, 4), dtype=start_point.dtype, device=start_point.device)
        return empty, empty_quat, empty

    norms = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    default_direction = torch.zeros_like(direction)
    default_direction[:, 0] = 1.0
    normalized_direction = torch.where(
        norms > 1.0e-8,
        direction / torch.clamp(norms, min=1.0e-8),
        default_direction,
    )

    axis = torch.cross(default_direction, normalized_direction, dim=-1)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    dot_prod = torch.sum(default_direction * normalized_direction, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    fallback_axis = torch.zeros_like(axis)
    fallback_axis[:, 2] = 1.0
    axis = torch.where(axis_norm > 1.0e-8, axis / torch.clamp(axis_norm, min=1.0e-8), fallback_axis)

    half_angle = 0.5 * torch.acos(dot_prod)
    sin_half_angle = torch.sin(half_angle)
    arrow_quat = torch.cat([torch.cos(half_angle), axis * sin_half_angle], dim=-1)
    arrow_quat = torch.where(
        norms > 1.0e-8,
        arrow_quat,
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=direction.dtype, device=direction.device).expand_as(arrow_quat),
    )

    arrow_scale = torch.tensor(default_scale, dtype=direction.dtype, device=direction.device).repeat(direction.shape[0], 1)
    arrow_scale[:, 0] *= norms.squeeze(-1) * float(length_scale_factor)
    arrow_pos = start_point + 0.25 * direction
    return arrow_scale, arrow_quat, arrow_pos


def bucketize_distance_visualization(
    distance: torch.Tensor,
    near_threshold: float,
    mid_threshold: float,
) -> torch.Tensor:
    """Bucketize distances for visualization marker selection."""
    marker_indices = torch.full(distance.shape, 2, dtype=torch.int32, device=distance.device)
    marker_indices = torch.where(distance <= float(mid_threshold), torch.ones_like(marker_indices), marker_indices)
    marker_indices = torch.where(distance <= float(near_threshold), torch.zeros_like(marker_indices), marker_indices)
    return marker_indices


def get_visualizer_default_scale(visualizer, preferred_key: str | None = "arrow") -> Sequence[float]:
    """Return a marker scale from a visualizer cfg without assuming a fixed marker key."""
    markers = getattr(getattr(visualizer, "cfg", None), "markers", None)
    if not markers:
        raise ValueError("Visualizer does not expose any marker configuration.")
    if preferred_key is not None and preferred_key in markers:
        return markers[preferred_key].scale
    return next(iter(markers.values())).scale


def compute_sparse_contact_reward_components(
    d_nearest: torch.Tensor,
    relation: torch.Tensor,
    exists_mask: torch.Tensor,
    gate_ref: torch.Tensor,
    hold_values: torch.Tensor,
    contact_distance_threshold: float,
    forbid_contact_distance_threshold: float,
    mandatory_alpha: float,
    optional_beta: float,
    mandatory_weights: dict[str, float],
    optional_weights: dict[str, float],
    forbidden_weight: float,
) -> SparseContactRewardComponents:
    """Compute the sparse contact reward from pairwise distances and relation labels."""
    if relation.ndim == 2:
        relation = relation.unsqueeze(0).expand(d_nearest.shape[0], -1, -1)
    if exists_mask.ndim == 1:
        exists_mask = exists_mask.unsqueeze(0).expand(d_nearest.shape[0], -1)

    exists_pair = exists_mask[:, None, :]
    mandatory_mask = (relation == 1) & exists_pair
    optional_mask = (relation == 0) & exists_pair
    forbidden_mask = (relation == -1) & exists_pair

    mandatory_contact = d_nearest <= contact_distance_threshold
    optional_contact = mandatory_contact
    forbidden_contact = d_nearest <= forbid_contact_distance_threshold

    mandatory_term = (
        mandatory_weights.get("proximity", 0.0) * torch.exp(-mandatory_alpha * d_nearest)
        + mandatory_weights.get("contact", 0.0) * mandatory_contact.float()
        + mandatory_weights.get("hold", 0.0) * hold_values
    ) * mandatory_mask.float()
    optional_term = (
        optional_weights.get("proximity", 0.0) * torch.exp(-optional_beta * d_nearest)
        + optional_weights.get("contact", 0.0) * optional_contact.float()
    ) * optional_mask.float()
    forbidden_term = forbidden_weight * forbidden_contact.float() * forbidden_mask.float()

    gate = gate_ref[:, None, None]
    mandatory = (gate * mandatory_term).sum(dim=(1, 2))
    optional = (gate * optional_term).sum(dim=(1, 2))
    forbidden = forbidden_term.sum(dim=(1, 2))

    return SparseContactRewardComponents(
        total=mandatory + optional - forbidden,
        mandatory=mandatory,
        optional=optional,
        forbidden=forbidden,
        mandatory_contact=mandatory_contact & mandatory_mask,
        optional_contact=optional_contact & optional_mask,
        forbidden_contact=forbidden_contact & forbidden_mask,
    )


class SparseContactReward(ManagerTermBase):
    """Stateful sparse contact reward for interaction sitting tasks."""

    DEBUG_PAIR_NAMES: tuple[tuple[str, str], ...] = (
        ("pelvis", "seat"),
        ("torso", "back"),
        ("left_hand", "arm_left"),
        ("right_hand", "arm_right"),
    )

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        params = cfg.params
        self.asset_cfg = params.get("asset_cfg", SceneEntityCfg("objects"))
        self.reference_cfg = params.get("reference_cfg", SceneEntityCfg("motion_reference"))
        self.robot_cfg = params.get("robot_cfg", SceneEntityCfg("robot"))
        self.object_name = params.get("object_name", "box")
        self.metadata_dir = params.get("metadata_dir", None)
        if not self.metadata_dir:
            raise ValueError("SparseContactReward requires a non-empty `metadata_dir`.")

        metadata_by_key = load_sparse_contact_map_directory(self.metadata_dir)
        self._metadata_bundle = _build_metadata_bundle(metadata_by_key, device=self.device)
        self._robot_links = list(self._metadata_bundle["robot_links"])
        self._object_parts_order = list(self._metadata_bundle["object_parts_order"])
        self._part_name_to_idx = {name: idx for idx, name in enumerate(self._object_parts_order)}
        self._link_name_to_idx = {name: idx for idx, name in enumerate(self._robot_links)}
        self._usd_basename_to_object_index = self._metadata_bundle["usd_basename_to_object_index"]
        self._object_key_to_object_index = self._metadata_bundle["object_key_to_object_index"]

        link_name_map = params.get(
            "link_name_map",
            {
                "pelvis": "pelvis",
                "torso": "torso_link",
                "left_hand": "left_wrist_yaw_link",
                "right_hand": "right_wrist_yaw_link",
            },
        )
        missing_links = [name for name in self._robot_links if name not in link_name_map]
        if missing_links:
            raise ValueError(f"SparseContactReward missing link_name_map entries for: {missing_links}")

        robot: Articulation = env.scene[self.robot_cfg.name]
        requested_body_names = [link_name_map[name] for name in self._robot_links]
        body_ids, _ = robot.find_bodies(requested_body_names, preserve_order=True)
        if len(body_ids) != len(requested_body_names):
            raise ValueError(
                "SparseContactReward failed to resolve all requested robot bodies: "
                f"requested={requested_body_names}, resolved={len(body_ids)}"
            )
        self._link_body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=self.device)

        self.contact_distance_threshold = float(params.get("contact_distance_threshold", 0.12))
        self.forbid_contact_distance_threshold = float(params.get("forbid_contact_distance_threshold", 0.10))
        self.mandatory_alpha = float(params.get("mandatory_alpha", 10.0))
        self.optional_beta = float(params.get("optional_beta", 8.0))
        self.mandatory_weights = dict(params.get("mandatory_weights", {"proximity": 0.35, "contact": 1.0, "hold": 0.5}))
        self.optional_weights = dict(params.get("optional_weights", {"proximity": 0.1, "contact": 0.25}))
        self.forbidden_weight = float(params.get("forbidden_weight", 1.0))
        self._hold_buffer = SlidingWindowContactBuffer(
            num_envs=env.num_envs,
            num_links=len(self._robot_links),
            num_parts=len(self._object_parts_order),
            window=int(params.get("hold_window", 5)),
            device=self.device,
        )
        self.debug_vis = bool(params.get("debug_vis", False))
        self.debug_vis_max_envs = int(params.get("debug_vis_max_envs", 1))
        self.debug_vis_show_all_points = bool(params.get("debug_vis_show_all_points", True))
        self.debug_vis_show_nearest = bool(params.get("debug_vis_show_nearest", True))
        self.debug_vis_point_radius = float(params.get("debug_vis_point_radius", 0.028))
        self.debug_vis_nearest_point_radius = float(params.get("debug_vis_nearest_point_radius", 0.04))
        self.debug_vis_arrow_length_scale = float(params.get("debug_vis_arrow_length_scale", 10.0))
        self.debug_vis_arrow_thickness_scale = tuple(params.get("debug_vis_arrow_thickness_scale", (1.0, 1.0, 1.0)))
        self.debug_vis_nearest_arrow_thickness_scale = tuple(
            params.get("debug_vis_nearest_arrow_thickness_scale", (1.25, 1.25, 1.25))
        )
        self.debug_vis_distance_bucket_thresholds = tuple(
            params.get("debug_vis_distance_bucket_thresholds", (0.12, 0.30))
        )
        self._point_visualizer = None
        self._arrow_visualizer = None
        self._nearest_point_visualizer = None
        self._nearest_arrow_visualizer = None

        self._missing_contact_warning_printed = False
        self._metric_names = [
            "mandatory_distance",
            "pelvis_to_seat_distance",
            "torso_to_back_distance",
            "left_hand_to_arm_left_distance",
            "right_hand_to_arm_right_distance",
            "pelvis_seat_contact",
            "torso_back_contact",
            "left_hand_arm_left_contact",
            "right_hand_arm_right_contact",
            "mandatory_reward",
            "optional_reward",
            "forbidden_penalty",
            "total_contact_reward",
        ]
        self._current_metrics = {
            name: torch.zeros(env.num_envs, dtype=torch.float32, device=self.device) for name in self._metric_names
        }
        self.set_debug_vis(self.debug_vis)

    @property
    def metric_names(self) -> list[str]:
        return list(self._metric_names)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        self._hold_buffer.reset(env_ids)
        if env_ids is None:
            for value in self._current_metrics.values():
                value.zero_()
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for value in self._current_metrics.values():
            value[env_ids_tensor] = 0.0

    def get_current_metrics(self) -> dict[str, torch.Tensor]:
        return self._current_metrics

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
        object_name: str = "box",
        metadata_dir: str | None = None,
        link_name_map: dict[str, str] | None = None,
        contact_distance_threshold: float | None = None,
        forbid_contact_distance_threshold: float | None = None,
        mandatory_alpha: float | None = None,
        optional_beta: float | None = None,
        mandatory_weights: dict[str, float] | None = None,
        optional_weights: dict[str, float] | None = None,
        forbidden_weight: float | None = None,
        hold_window: int | None = None,
        debug_vis: bool | None = None,
        debug_vis_max_envs: int | None = None,
        debug_vis_show_all_points: bool | None = None,
        debug_vis_show_nearest: bool | None = None,
        debug_vis_point_radius: float | None = None,
        debug_vis_nearest_point_radius: float | None = None,
        debug_vis_arrow_length_scale: float | None = None,
        debug_vis_arrow_thickness_scale: Sequence[float] | None = None,
        debug_vis_nearest_arrow_thickness_scale: Sequence[float] | None = None,
        debug_vis_distance_bucket_thresholds: Sequence[float] | None = None,
    ) -> torch.Tensor:
        del metadata_dir, link_name_map, hold_window, debug_vis, debug_vis_max_envs
        del debug_vis_show_all_points, debug_vis_show_nearest
        del debug_vis_point_radius, debug_vis_nearest_point_radius, debug_vis_arrow_length_scale
        del debug_vis_arrow_thickness_scale, debug_vis_nearest_arrow_thickness_scale
        del debug_vis_distance_bucket_thresholds

        object_pos_w, object_quat_w, _, _ = get_object_state_w(env, asset_cfg)
        robot: Articulation = env.scene[robot_cfg.name]
        link_pos_w = robot.data.body_pos_w[:, self._link_body_ids, :]

        object_metadata_indices = self._resolve_object_metadata_indices(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
        part_points_local = self._metadata_bundle["points_local"][object_metadata_indices]
        part_centers_local = self._metadata_bundle["centers_local"][object_metadata_indices]
        point_valid_mask = self._metadata_bundle["point_valid_mask"][object_metadata_indices]
        relation = self._metadata_bundle["relation"][object_metadata_indices]
        exists_mask = self._metadata_bundle["exists_mask"][object_metadata_indices]

        part_points_w = transform_part_points_to_world(part_points_local, object_pos_w, object_quat_w)
        part_centers_w = transform_part_centers_to_world(part_centers_local, object_pos_w, object_quat_w)

        d_nearest = compute_link_part_nearest_distance(link_pos_w, part_points_w, point_valid_mask=point_valid_mask)
        d_center = compute_link_part_center_distance(link_pos_w, part_centers_w)
        pseudo_contact = d_nearest <= float(
            self.contact_distance_threshold if contact_distance_threshold is None else contact_distance_threshold
        )
        hold_values = self._hold_buffer.update(pseudo_contact)

        components = compute_sparse_contact_reward_components(
            d_nearest=d_nearest,
            relation=relation,
            exists_mask=exists_mask,
            gate_ref=self._get_reference_contact_gate(env, reference_cfg=reference_cfg, object_name=object_name),
            hold_values=hold_values,
            contact_distance_threshold=float(
                self.contact_distance_threshold if contact_distance_threshold is None else contact_distance_threshold
            ),
            forbid_contact_distance_threshold=float(
                self.forbid_contact_distance_threshold
                if forbid_contact_distance_threshold is None
                else forbid_contact_distance_threshold
            ),
            mandatory_alpha=float(self.mandatory_alpha if mandatory_alpha is None else mandatory_alpha),
            optional_beta=float(self.optional_beta if optional_beta is None else optional_beta),
            mandatory_weights=self.mandatory_weights if mandatory_weights is None else mandatory_weights,
            optional_weights=self.optional_weights if optional_weights is None else optional_weights,
            forbidden_weight=float(self.forbidden_weight if forbidden_weight is None else forbidden_weight),
        )

        self._update_metrics(
            d_nearest=d_nearest,
            d_center=d_center,
            relation=relation,
            exists_mask=exists_mask,
            components=components,
        )
        if self.debug_vis:
            self._update_debug_visualization(
                link_pos_w=link_pos_w,
                part_points_w=part_points_w,
                relation=relation,
                exists_mask=exists_mask,
                point_valid_mask=point_valid_mask,
            )
        return components.total

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Enable or disable mandatory distance debug visualization."""
        self.debug_vis = bool(debug_vis)
        self._set_debug_vis_impl(self.debug_vis)
        return self.debug_vis

    def _get_reference_contact_gate(
        self,
        env: ManagerBasedRLEnv,
        reference_cfg: SceneEntityCfg,
        object_name: str,
    ) -> torch.Tensor:
        motion_reference = env.scene[reference_cfg.name]
        frame_ids = motion_reference.aiming_frame_idx
        env_ids = motion_reference.ALL_INDICES
        object_data = getattr(motion_reference.data, "object_data", {})

        if object_name not in object_data or "contact" not in object_data[object_name]:
            if not self._missing_contact_warning_printed:
                print(
                    f"[SparseContactReward] Missing '{object_name}' contact reference in motion data. "
                    "Falling back to gate=1."
                )
                self._missing_contact_warning_printed = True
            return torch.ones(env.num_envs, device=self.device, dtype=torch.float32)

        contact = torch.nan_to_num(
            object_data[object_name]["contact"][env_ids, frame_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).reshape(env.num_envs)
        validity = motion_reference.data.validity[env_ids, frame_ids].to(contact.dtype)
        return torch.where(validity > 0, contact, torch.ones_like(contact))

    def _resolve_object_metadata_indices(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        reference_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        current_basenames = get_current_object_usd_basenames(env, reference_cfg=reference_cfg, asset_cfg=asset_cfg)
        current_object_keys = get_current_object_keys(env, reference_cfg=reference_cfg, asset_cfg=asset_cfg)

        object_indices = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        unresolved: list[tuple[int, str, str]] = []
        for env_idx, (basename, object_key) in enumerate(zip(current_basenames, current_object_keys)):
            if basename in self._usd_basename_to_object_index:
                object_indices[env_idx] = self._usd_basename_to_object_index[basename]
            elif object_key in self._object_key_to_object_index:
                object_indices[env_idx] = self._object_key_to_object_index[object_key]
            elif len(self._object_key_to_object_index) == 1:
                object_indices[env_idx] = next(iter(self._object_key_to_object_index.values()))
            else:
                unresolved.append((env_idx, basename, object_key))

        if unresolved:
            raise ValueError(
                "SparseContactReward could not resolve sparse contact metadata for env/object assignments: "
                f"{unresolved[:8]}"
            )
        return object_indices

    def _update_metrics(
        self,
        d_nearest: torch.Tensor,
        d_center: torch.Tensor,
        relation: torch.Tensor,
        exists_mask: torch.Tensor,
        components: SparseContactRewardComponents,
    ) -> None:
        del d_center
        mandatory_mask = (relation == 1) & exists_mask[:, None, :]
        masked_mandatory_distance = d_nearest.masked_fill(~mandatory_mask, torch.inf)
        has_mandatory = torch.isfinite(masked_mandatory_distance).any(dim=(1, 2))
        mandatory_distance = masked_mandatory_distance.amin(dim=(1, 2))
        self._current_metrics["mandatory_distance"] = torch.where(
            has_mandatory,
            mandatory_distance,
            torch.zeros_like(mandatory_distance),
        )
        for link_name, part_name in self.DEBUG_PAIR_NAMES:
            link_idx = self._link_name_to_idx[link_name]
            part_idx = self._part_name_to_idx[part_name]
            pair_exists = exists_mask[:, part_idx]

            if link_name == "pelvis" and part_name == "seat":
                distance_key = "pelvis_to_seat_distance"
                contact_key = "pelvis_seat_contact"
                contact_value = components.mandatory_contact[:, link_idx, part_idx].float()
            elif link_name == "torso" and part_name == "back":
                distance_key = "torso_to_back_distance"
                contact_key = "torso_back_contact"
                contact_value = components.optional_contact[:, link_idx, part_idx].float()
            elif link_name == "left_hand" and part_name == "arm_left":
                distance_key = "left_hand_to_arm_left_distance"
                contact_key = "left_hand_arm_left_contact"
                contact_value = components.optional_contact[:, link_idx, part_idx].float()
            else:
                distance_key = "right_hand_to_arm_right_distance"
                contact_key = "right_hand_arm_right_contact"
                contact_value = components.optional_contact[:, link_idx, part_idx].float()

            self._current_metrics[distance_key] = torch.where(
                pair_exists,
                d_nearest[:, link_idx, part_idx],
                torch.zeros_like(d_nearest[:, link_idx, part_idx]),
            )
            self._current_metrics[contact_key] = torch.where(
                pair_exists,
                contact_value,
                torch.zeros_like(contact_value),
            )

        self._current_metrics["mandatory_reward"] = components.mandatory
        self._current_metrics["optional_reward"] = components.optional
        self._current_metrics["forbidden_penalty"] = components.forbidden
        self._current_metrics["total_contact_reward"] = components.total

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if VisualizationMarkers is None or VisualizationMarkersCfg is None or sim_utils is None:
            self.debug_vis = False
            return

        if debug_vis:
            if self._point_visualizer is None:
                self._point_visualizer = VisualizationMarkers(
                    _build_mandatory_point_visualizer_cfg(radius=self.debug_vis_point_radius)
                )
            if self._arrow_visualizer is None:
                arrow_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/SparseContactMandatory/arrows")
                arrow_marker = arrow_cfg.markers["arrow"]
                arrow_marker.scale = tuple(
                    base * factor for base, factor in zip(arrow_marker.scale, self.debug_vis_arrow_thickness_scale)
                )
                self._arrow_visualizer = VisualizationMarkers(arrow_cfg)
            if self._nearest_point_visualizer is None:
                self._nearest_point_visualizer = VisualizationMarkers(
                    _build_mandatory_nearest_point_visualizer_cfg(radius=self.debug_vis_nearest_point_radius)
                )
            if self._nearest_arrow_visualizer is None:
                self._nearest_arrow_visualizer = VisualizationMarkers(
                    _build_mandatory_nearest_arrow_visualizer_cfg(
                        thickness_scale=self.debug_vis_nearest_arrow_thickness_scale
                    )
                )
            self._set_visualizer_visibility(self._point_visualizer, True)
            self._set_visualizer_visibility(self._arrow_visualizer, True)
            self._set_visualizer_visibility(self._nearest_point_visualizer, True)
            self._set_visualizer_visibility(self._nearest_arrow_visualizer, True)
        else:
            self._set_visualizer_visibility(self._point_visualizer, False)
            self._set_visualizer_visibility(self._arrow_visualizer, False)
            self._set_visualizer_visibility(self._nearest_point_visualizer, False)
            self._set_visualizer_visibility(self._nearest_arrow_visualizer, False)

    def _update_debug_visualization(
        self,
        link_pos_w: torch.Tensor,
        part_points_w: torch.Tensor,
        relation: torch.Tensor,
        exists_mask: torch.Tensor,
        point_valid_mask: torch.Tensor,
    ) -> None:
        if not self.debug_vis:
            return

        debug_data = extract_mandatory_contact_debug_data(
            link_pos_w=link_pos_w,
            part_points_w=part_points_w,
            relation=relation,
            exists_mask=exists_mask,
            point_valid_mask=point_valid_mask,
            max_envs=self.debug_vis_max_envs,
        )

        if self.debug_vis_show_all_points and debug_data.all_point_positions.shape[0] > 0:
            self._set_visualizer_visibility(self._point_visualizer, True)
            self._set_visualizer_visibility(self._arrow_visualizer, True)
            self._point_visualizer.visualize(translations=debug_data.all_point_positions)
            arrow_scale, arrow_quat, arrow_pos = resolve_direction_to_arrow_marker(
                direction=debug_data.all_arrow_directions,
                start_point=debug_data.all_arrow_start_positions,
                default_scale=get_visualizer_default_scale(self._arrow_visualizer, preferred_key="arrow"),
                length_scale_factor=self.debug_vis_arrow_length_scale,
            )
            self._arrow_visualizer.visualize(translations=arrow_pos, orientations=arrow_quat, scales=arrow_scale)
        else:
            self._set_visualizer_visibility(self._point_visualizer, False)
            self._set_visualizer_visibility(self._arrow_visualizer, False)

        if self.debug_vis_show_nearest and debug_data.nearest_point_positions.shape[0] > 0:
            self._set_visualizer_visibility(self._nearest_point_visualizer, True)
            self._set_visualizer_visibility(self._nearest_arrow_visualizer, True)
            nearest_distances = torch.linalg.vector_norm(debug_data.nearest_arrow_directions, dim=-1)
            marker_indices = bucketize_distance_visualization(
                nearest_distances,
                near_threshold=self.debug_vis_distance_bucket_thresholds[0],
                mid_threshold=self.debug_vis_distance_bucket_thresholds[1],
            )
            self._nearest_point_visualizer.visualize(
                translations=debug_data.nearest_point_positions,
                marker_indices=marker_indices,
            )
            nearest_scale, nearest_quat, nearest_pos = resolve_direction_to_arrow_marker(
                direction=debug_data.nearest_arrow_directions,
                start_point=debug_data.nearest_arrow_start_positions,
                default_scale=get_visualizer_default_scale(self._nearest_arrow_visualizer, preferred_key="near"),
                length_scale_factor=self.debug_vis_arrow_length_scale,
            )
            self._nearest_arrow_visualizer.visualize(
                translations=nearest_pos,
                orientations=nearest_quat,
                scales=nearest_scale,
                marker_indices=marker_indices,
            )
        else:
            self._set_visualizer_visibility(self._nearest_point_visualizer, False)
            self._set_visualizer_visibility(self._nearest_arrow_visualizer, False)

    @staticmethod
    def _set_visualizer_visibility(visualizer, visible: bool) -> None:
        if visualizer is not None:
            visualizer.set_visibility(visible)


def _build_metadata_bundle(
    metadata_by_key: dict[str, SparseContactMapMetadata],
    device: str | torch.device,
) -> dict[str, Any]:
    object_keys = sorted(metadata_by_key.keys())
    reference_metadata = metadata_by_key[object_keys[0]]
    robot_links = reference_metadata.robot_links
    object_parts_order = reference_metadata.object_parts_order

    for object_key in object_keys[1:]:
        metadata = metadata_by_key[object_key]
        if metadata.robot_links != robot_links:
            raise ValueError("Sparse contact metadata must use a consistent robot link order across objects.")
        if metadata.object_parts_order != object_parts_order:
            raise ValueError("Sparse contact metadata must use a consistent object part order across objects.")

    num_objects = len(object_keys)
    num_links = len(robot_links)
    num_parts = len(object_parts_order)
    max_points = max(
        max(metadata.parts[part_name].points_local.shape[0] for part_name in object_parts_order)
        for metadata in metadata_by_key.values()
    )
    max_points = max(max_points, 1)

    relation = torch.zeros(num_objects, num_links, num_parts, dtype=torch.int64, device=device)
    exists_mask = torch.zeros(num_objects, num_parts, dtype=torch.bool, device=device)
    centers_local = torch.zeros(num_objects, num_parts, 3, dtype=torch.float32, device=device)
    points_local = torch.zeros(num_objects, num_parts, max_points, 3, dtype=torch.float32, device=device)
    point_valid_mask = torch.zeros(num_objects, num_parts, max_points, dtype=torch.bool, device=device)
    usd_basename_to_object_index: dict[str, int] = {}
    object_key_to_object_index: dict[str, int] = {}

    for object_index, object_key in enumerate(object_keys):
        metadata = metadata_by_key[object_key]
        relation[object_index] = metadata.relation.to(device=device)
        exists_mask[object_index] = metadata.exists_mask.to(device=device)
        usd_basename_to_object_index[metadata.usd_basename] = object_index
        object_key_to_object_index[object_key] = object_index

        for part_index, part_name in enumerate(object_parts_order):
            part = metadata.parts[part_name]
            centers_local[object_index, part_index] = part.center_local.to(device=device)
            if part.points_local.shape[0] > 0:
                num_points = part.points_local.shape[0]
                points_local[object_index, part_index, :num_points] = part.points_local.to(device=device)
                point_valid_mask[object_index, part_index, :num_points] = True

    return {
        "robot_links": robot_links,
        "object_parts_order": object_parts_order,
        "relation": relation,
        "exists_mask": exists_mask,
        "centers_local": centers_local,
        "points_local": points_local,
        "point_valid_mask": point_valid_mask,
        "usd_basename_to_object_index": usd_basename_to_object_index,
        "object_key_to_object_index": object_key_to_object_index,
    }


def _empty_mandatory_contact_debug_data(device: str | torch.device) -> MandatoryContactDebugData:
    empty = torch.zeros((0, 3), dtype=torch.float32, device=device)
    return MandatoryContactDebugData(
        all_point_positions=empty,
        all_arrow_start_positions=empty,
        all_arrow_directions=empty,
        nearest_point_positions=empty,
        nearest_arrow_start_positions=empty,
        nearest_arrow_directions=empty,
    )


def _build_mandatory_point_visualizer_cfg(radius: float):
    if VisualizationMarkersCfg is None or sim_utils is None:
        return None
    return VisualizationMarkersCfg(
        prim_path="/Visuals/SparseContactMandatory/points",
        markers={
            "mandatory_point": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 1.0)),
            ),
        },
    )


def _build_mandatory_nearest_point_visualizer_cfg(radius: float):
    if VisualizationMarkersCfg is None or sim_utils is None:
        return None
    return VisualizationMarkersCfg(
        prim_path="/Visuals/SparseContactMandatory/nearest_points",
        markers={
            "near": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
            ),
            "mid": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.2)),
            ),
            "far": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            ),
        },
    )


def _build_mandatory_nearest_arrow_visualizer_cfg(thickness_scale: Sequence[float]):
    if RED_ARROW_X_MARKER_CFG is None:
        return None
    near_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/SparseContactMandatory/nearest_arrows")
    near_marker = near_cfg.markers["arrow"]
    near_marker.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2))
    near_marker.scale = tuple(base * factor for base, factor in zip(near_marker.scale, thickness_scale))

    mid_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/SparseContactMandatory/nearest_arrows")
    mid_marker = mid_cfg.markers["arrow"]
    mid_marker.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.2))
    mid_marker.scale = tuple(base * factor for base, factor in zip(mid_marker.scale, thickness_scale))

    far_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/SparseContactMandatory/nearest_arrows")
    far_marker = far_cfg.markers["arrow"]
    far_marker.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2))
    far_marker.scale = tuple(base * factor for base, factor in zip(far_marker.scale, thickness_scale))

    return VisualizationMarkersCfg(
        prim_path="/Visuals/SparseContactMandatory/nearest_arrows",
        markers={
            "near": near_marker,
            "mid": mid_marker,
            "far": far_marker,
        },
    )

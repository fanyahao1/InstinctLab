#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher
import yaml

DEFAULT_CONTACT_POS_THRESHOLD = 0.02
DEFAULT_CONTACT_ORI_THRESHOLD_DEG = 5.0
DEFAULT_CONTACT_MARKER_RADIUS = 0.05
DEFAULT_CONTACT_MARKER_HEIGHT = 0.18


parser = argparse.ArgumentParser(
    description="Direct-workflow replay and quality monitor for 29DoF+object motion npz."
)
parser.add_argument("--motion", type=str, default=None, help="Path to a single merged motion npz file.")
parser.add_argument(
    "--motion-dir",
    type=str,
    default=None,
    help="Directory containing merged motion npz files. If set and --motion is omitted, the first file is used.",
)
parser.add_argument("--loop", action="store_true", help="Loop replay indefinitely.")
parser.add_argument("--playback-rate", type=float, default=1.0, help="Replay speed multiplier.")
parser.add_argument("--robot-height-offset", type=float, default=0.0, help="Extra z offset for robot base pose.")
parser.add_argument("--object-height-offset", type=float, default=0.0, help="Extra z offset for object pose.")
parser.add_argument("--print-every", type=int, default=30, help="Print quality stats every N frames.")
parser.add_argument("--object-size", type=float, nargs=3, default=(0.35, 0.35, 0.35), help="Proxy cuboid size (m).")
parser.add_argument(
    "--contact-pos-threshold",
    type=float,
    default=DEFAULT_CONTACT_POS_THRESHOLD,
    help="Fallback translation threshold (m) to infer object contact when the npz has no object_contact track.",
)
parser.add_argument(
    "--contact-ori-threshold-deg",
    type=float,
    default=DEFAULT_CONTACT_ORI_THRESHOLD_DEG,
    help="Fallback rotation threshold (deg) to infer object contact when the npz has no object_contact track.",
)
parser.add_argument(
    "--contact-marker-radius",
    type=float,
    default=DEFAULT_CONTACT_MARKER_RADIUS,
    help="Radius of the contact-state sphere shown above the object.",
)
parser.add_argument(
    "--contact-marker-height",
    type=float,
    default=DEFAULT_CONTACT_MARKER_HEIGHT,
    help="Extra height above the top of the object for the contact-state sphere.",
)
parser.add_argument(
    "--ground-clearance",
    type=float,
    default=0.005,
    help="Minimum clearance above z=0 to enforce using frame-0 auto correction (meters).",
)
parser.add_argument(
    "--use-joint-vel",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to write joint velocity from npz to simulator (default: true).",
)
# parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
# from isaaclab_assets import G1_CFG
from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_CFG


def _f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _norm_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n > 1e-12, n, 1.0)
    return (q / n).astype(np.float32)


def _quat_geodesic_angle_to_reference_wxyz(reference_wxyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    ref = _norm_quat_wxyz(np.asarray(reference_wxyz, dtype=np.float64))
    quat = _norm_quat_wxyz(np.asarray(quat_wxyz, dtype=np.float64))
    if ref.ndim != 1 or ref.shape[0] != 4:
        raise ValueError(f"Expected reference quaternion shape (4,), got {ref.shape}")
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape (T,4), got {quat.shape}")
    dot = np.sum(quat * ref[None, :], axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return (2.0 * np.arccos(dot)).astype(np.float32)


def _compute_latched_contact_track(
    object_pos_w: np.ndarray,
    object_quat_wxyz: np.ndarray,
    pos_threshold: float,
    ori_threshold_deg: float,
) -> tuple[np.ndarray, int]:
    pos = np.asarray(object_pos_w, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected object_pos_w shape (T,3), got {pos.shape}")
    pos_disp = np.linalg.norm(pos - pos[:1], axis=-1)
    moved_by_pos = pos_disp >= float(pos_threshold)

    quat = np.asarray(object_quat_wxyz, dtype=np.float32)
    moved_by_ori = np.zeros(pos.shape[0], dtype=bool)
    if quat.ndim == 2 and quat.shape[1] == 4:
        ori_disp = _quat_geodesic_angle_to_reference_wxyz(quat[0], quat)
        moved_by_ori = ori_disp >= np.deg2rad(float(ori_threshold_deg))

    moved = moved_by_pos | moved_by_ori
    active_idx = np.flatnonzero(moved)
    if active_idx.size == 0:
        return np.zeros(pos.shape[0], dtype=np.int64), -1

    start_frame = int(active_idx[0])
    contact = np.zeros(pos.shape[0], dtype=np.int64)
    contact[start_frame:] = 1
    return contact, start_frame


def _quat_ang_err_rad(q_target: torch.Tensor, q_sim: torch.Tensor) -> torch.Tensor:
    # q: (...,4) in wxyz
    dot = torch.sum(q_target * q_sim, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _robot_min_z(robot: Articulation) -> torch.Tensor:
    # Prefer body-wise positions to detect actual penetration, not just base/root.
    if hasattr(robot.data, "body_pos_w"):
        return robot.data.body_pos_w[:, :, 2].min(dim=1).values
    return robot.data.root_pos_w[:, 2]


def _object_root_and_bottom_z(replay_object: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
    root_z = replay_object.data.root_pos_w[:, 2]
    # Proxy object is a cuboid with center at root pose. Bottom-z estimate ignores tilt.
    half_h = float(args_cli.object_size[2]) * 0.5
    bottom_z = root_z - half_h
    return root_z, bottom_z


def _pick_motion_path() -> Path:
    if args_cli.motion:
        path = Path(args_cli.motion)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = []
    if args_cli.motion_dir:
        candidates.append(Path(args_cli.motion_dir))
    candidates.extend(
        [
            Path("datasets/interaction/output_npz_29dof_with_object"),
            Path("data/output_npz_29dof_with_object"),
        ]
    )

    for root in candidates:
        if not root.exists():
            continue
        files = sorted(root.rglob("*.npz"))
        if files:
            return files[0]
    raise FileNotFoundError("No motion npz found. Use --motion or --motion-dir.")


def _find_metadata_path(motion_path: Path) -> Path | None:
    for parent in [motion_path.parent, *motion_path.parents]:
        metadata_path = parent / "metadata.yaml"
        if metadata_path.exists():
            return metadata_path
    return None


def _resolve_object_usd_from_metadata(motion_path: Path) -> Path | None:
    metadata_path = _find_metadata_path(motion_path)
    if metadata_path is None:
        return None

    metadata_root = metadata_path.parent.resolve()
    motion_path = motion_path.resolve()

    try:
        motion_rel = motion_path.relative_to(metadata_root).as_posix()
    except ValueError:
        motion_rel = motion_path.name

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f) or {}

    motion_entries = metadata.get("motion_files", [])
    object_entries = metadata.get("objects", [])

    matched_motion = next((entry for entry in motion_entries if entry.get("motion_file") == motion_rel), None)
    if matched_motion is None:
        matched_motion = next((entry for entry in motion_entries if Path(entry.get("motion_file", "")).name == motion_path.name), None)
    if matched_motion is None:
        return None

    object_id = matched_motion.get("object_id")
    matched_object = next((entry for entry in object_entries if entry.get("object_id") == object_id), None)
    if matched_object is None:
        return None

    usd_rel = matched_object.get("usd_path")
    if not usd_rel:
        return None

    usd_path = (metadata_root / usd_rel).resolve()
    return usd_path if usd_path.exists() else None


@dataclass
class ReplayMotion:
    path: Path
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    base_pos: torch.Tensor
    base_quat: torch.Tensor
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    object_prefix: str
    object_pos: torch.Tensor
    object_quat: torch.Tensor
    object_lin_vel: torch.Tensor
    object_ang_vel: torch.Tensor
    object_contact: torch.Tensor
    object_contact_start_frame: int
    object_contact_source: str
    framerate: float
    num_frames: int
    joint_names: list[str] | None


def _find_object_prefix(keys: list[str]) -> str:
    reserved = {
        "joint_pos",
        "joint_vel",
        "base_pos",
        "base_quat",
        "base_lin_vel",
        "base_ang_vel",
        "framerate",
        "joint_names",
        "base_pos_w",
        "base_quat_w",
        "box_pos",
        "box_quat",
        "box_lin_vel",
        "box_ang_vel",
    }
    if "box_pos" in keys:
        return "box"
    prefixes = []
    for k in keys:
        if k in reserved:
            continue
        if k.endswith("_pos"):
            prefixes.append(k[:-4])
    prefixes = sorted(set(prefixes))
    if len(prefixes) != 1:
        raise ValueError(f"Cannot auto-detect object prefix from keys={keys}; candidates={prefixes}")
    return prefixes[0]


def load_replay_motion(path: Path, device: torch.device) -> ReplayMotion:
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())

    required = ["joint_pos", "framerate"]
    missing = [k for k in required if k not in keys]
    if missing:
        raise KeyError(f"Missing required keys in {path}: {missing}")

    base_pos_key = "base_pos_w" if "base_pos_w" in keys else "base_pos"
    base_quat_key = "base_quat_w" if "base_quat_w" in keys else "base_quat"
    if base_pos_key not in keys or base_quat_key not in keys:
        raise KeyError(
            f"Missing required base pose keys in {path}: need base_pos_w/base_quat_w or base_pos/base_quat; keys={keys}"
        )

    prefix = _find_object_prefix(keys)
    pos_key = f"{prefix}_pos"
    quat_key = f"{prefix}_quat"
    lin_vel_key = f"{prefix}_lin_vel"
    ang_vel_key = f"{prefix}_ang_vel"

    if pos_key not in keys and prefix == "box":
        pos_key = "box_pos"
        quat_key = "box_quat"
        lin_vel_key = "box_lin_vel"
        ang_vel_key = "box_ang_vel"

    num_frames = int(np.asarray(z["joint_pos"]).shape[0])
    if num_frames <= 0:
        raise ValueError(f"Empty motion: {path}")

    joint_pos = _f32(z["joint_pos"])
    joint_vel = _f32(z["joint_vel"]) if "joint_vel" in keys else np.zeros_like(joint_pos, dtype=np.float32)
    base_pos = _f32(z[base_pos_key])
    base_quat = _norm_quat_wxyz(z[base_quat_key])
    base_lin_vel_key = "base_lin_vel_w" if "base_lin_vel_w" in keys else "base_lin_vel"
    base_ang_vel_key = "base_ang_vel_w" if "base_ang_vel_w" in keys else "base_ang_vel"
    base_lin_vel = (
        _f32(z[base_lin_vel_key]) if base_lin_vel_key in keys else np.zeros((num_frames, 3), dtype=np.float32)
    )
    base_ang_vel = (
        _f32(z[base_ang_vel_key]) if base_ang_vel_key in keys else np.zeros((num_frames, 3), dtype=np.float32)
    )

    if pos_key not in keys:
        raise KeyError(f"Missing object position key `{pos_key}` in {path}. keys={keys}")
    object_pos = _f32(z[pos_key])
    if object_pos.shape[0] != num_frames:
        raise ValueError(f"{pos_key} frame mismatch: {object_pos.shape[0]} vs {num_frames}")

    if quat_key in keys:
        object_quat = _norm_quat_wxyz(z[quat_key])
    else:
        object_quat = np.zeros((num_frames, 4), dtype=np.float32)
        object_quat[:, 0] = 1.0

    object_lin_vel = _f32(z[lin_vel_key]) if lin_vel_key in keys else np.zeros((num_frames, 3), dtype=np.float32)
    object_ang_vel = _f32(z[ang_vel_key]) if ang_vel_key in keys else np.zeros((num_frames, 3), dtype=np.float32)

    object_contact_key = None
    object_contact_source = "inferred"
    for candidate in ("object_contact", f"{prefix}_contact"):
        if candidate in keys:
            object_contact_key = candidate
            object_contact_source = f"npz:{candidate}"
            break

    if object_contact_key is not None:
        object_contact = (np.asarray(z[object_contact_key]).reshape(-1) > 0).astype(np.int64)
        if object_contact.shape[0] != num_frames:
            raise ValueError(f"{object_contact_key} frame mismatch: {object_contact.shape[0]} vs {num_frames}")
        inferred_contact_start_frame = int(np.flatnonzero(object_contact > 0)[0]) if np.any(object_contact > 0) else -1
        object_contact_start_frame = (
            int(np.asarray(z["object_contact_start_frame"]).item())
            if "object_contact_start_frame" in keys
            else inferred_contact_start_frame
        )
        if object_contact_start_frame < 0 and inferred_contact_start_frame >= 0:
            object_contact_start_frame = inferred_contact_start_frame
    else:
        object_contact, object_contact_start_frame = _compute_latched_contact_track(
            object_pos,
            object_quat,
            pos_threshold=args_cli.contact_pos_threshold,
            ori_threshold_deg=args_cli.contact_ori_threshold_deg,
        )
        print(
            "[WARN] object_contact not found in npz. "
            f"Using fallback inference with pos_threshold={args_cli.contact_pos_threshold:.4f} m "
            f"and ori_threshold={args_cli.contact_ori_threshold_deg:.2f} deg."
        )

    joint_names = z["joint_names"].tolist() if "joint_names" in keys else None
    framerate = float(np.asarray(z["framerate"]).item())

    return ReplayMotion(
        path=path,
        joint_pos=torch.from_numpy(joint_pos).to(device),
        joint_vel=torch.from_numpy(joint_vel).to(device),
        base_pos=torch.from_numpy(base_pos).to(device),
        base_quat=torch.from_numpy(base_quat).to(device),
        base_lin_vel=torch.from_numpy(base_lin_vel).to(device),
        base_ang_vel=torch.from_numpy(base_ang_vel).to(device),
        object_prefix=prefix,
        object_pos=torch.from_numpy(object_pos).to(device),
        object_quat=torch.from_numpy(object_quat).to(device),
        object_lin_vel=torch.from_numpy(object_lin_vel).to(device),
        object_ang_vel=torch.from_numpy(object_ang_vel).to(device),
        object_contact=torch.from_numpy(object_contact).to(device=device, dtype=torch.long),
        object_contact_start_frame=int(object_contact_start_frame),
        object_contact_source=object_contact_source,
        framerate=framerate,
        num_frames=num_frames,
        joint_names=joint_names,
    )


def summarize_data_quality(motion: ReplayMotion) -> None:
    def _tensor_stats(name: str, x: torch.Tensor):
        x_cpu = x.detach().float().cpu()
        finite = torch.isfinite(x_cpu).all().item()
        print(
            f"[DATA] {name}: shape={tuple(x_cpu.shape)} finite={bool(finite)} "
            f"min={x_cpu.min().item():.4f} max={x_cpu.max().item():.4f}"
        )

    print(f"[INFO] motion file: {motion.path}")
    print(f"[INFO] frames={motion.num_frames}, framerate={motion.framerate:.3f} Hz, object_prefix={motion.object_prefix}")
    _tensor_stats("joint_pos", motion.joint_pos)
    _tensor_stats("joint_vel", motion.joint_vel)
    _tensor_stats("base_pos", motion.base_pos)
    _tensor_stats("base_lin_vel", motion.base_lin_vel)
    _tensor_stats("base_ang_vel", motion.base_ang_vel)
    _tensor_stats(f"{motion.object_prefix}_pos", motion.object_pos)
    _tensor_stats(f"{motion.object_prefix}_lin_vel", motion.object_lin_vel)
    _tensor_stats(f"{motion.object_prefix}_ang_vel", motion.object_ang_vel)

    base_qn = torch.linalg.norm(motion.base_quat, dim=-1)
    obj_qn = torch.linalg.norm(motion.object_quat, dim=-1)
    print(
        f"[DATA] base_quat_norm: mean={base_qn.mean().item():.6f}, "
        f"max_dev={(base_qn - 1.0).abs().max().item():.6e}"
    )
    print(
        f"[DATA] {motion.object_prefix}_quat_norm: mean={obj_qn.mean().item():.6f}, "
        f"max_dev={(obj_qn - 1.0).abs().max().item():.6e}"
    )
    active_frames = int((motion.object_contact > 0).sum().item())
    current_contact = int(motion.object_contact[0].item() > 0) if motion.num_frames > 0 else 0
    print(
        "[DATA] object_contact: "
        f"source={motion.object_contact_source} "
        f"start_frame={motion.object_contact_start_frame} active_frames={active_frames}/{motion.num_frames} "
        f"frame0={current_contact}"
    )


def _create_contact_state_marker() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/objectContactState",
        markers={
            "not_contacted": sim_utils.SphereCfg(
                radius=float(args_cli.contact_marker_radius),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.92, 0.1)),
            ),
            "contacted": sim_utils.SphereCfg(
                radius=float(args_cli.contact_marker_radius),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)),
            ),
        },
    )
    marker = VisualizationMarkers(marker_cfg)
    marker.set_visibility(True)
    return marker


def _remap_motion_joints_to_robot_order(motion: ReplayMotion, robot: Articulation) -> None:
    """Reorder replay joint tracks to match the runtime robot joint order."""
    robot_joint_names = list(robot.joint_names)

    if motion.joint_names is None:
        print("[WARN] joint_names not found in npz. Replay assumes joint_pos already matches robot.joint_names.")
        return

    if len(motion.joint_names) != motion.joint_pos.shape[1]:
        raise ValueError(
            f"joint_names length mismatch: len(joint_names)={len(motion.joint_names)} "
            f"vs joint_pos.shape[1]={motion.joint_pos.shape[1]}"
        )

    if len(robot_joint_names) != motion.joint_pos.shape[1]:
        raise ValueError(
            f"Robot joint count mismatch: robot={len(robot_joint_names)} vs motion={motion.joint_pos.shape[1]}"
        )

    missing_in_motion = [name for name in robot_joint_names if name not in motion.joint_names]
    extra_in_motion = [name for name in motion.joint_names if name not in robot_joint_names]
    if missing_in_motion or extra_in_motion:
        raise ValueError(
            "Joint-name mismatch between npz and robot.\n"
            f"Missing in npz: {missing_in_motion}\n"
            f"Extra in npz: {extra_in_motion}"
        )

    reorder = [motion.joint_names.index(name) for name in robot_joint_names]
    if reorder != list(range(len(reorder))):
        print("[INFO] Reordering joint tracks from npz joint_names to robot.joint_names.")
        print(f"[INFO] npz joint_names: {motion.joint_names}")
        print(f"[INFO] robot joint_names: {robot_joint_names}")
        motion.joint_pos = motion.joint_pos[:, reorder]
        motion.joint_vel = motion.joint_vel[:, reorder]
    else:
        print("[INFO] npz joint_names already match robot.joint_names.")

    motion.joint_names = robot_joint_names


def _compute_contact_marker_pose(
    replay_object: RigidObject,
    fallback_pos: torch.Tensor,
) -> torch.Tensor:
    """Place the contact-state sphere above the object's actual replay pose when available."""
    if hasattr(replay_object, "data") and hasattr(replay_object.data, "root_pos_w"):
        marker_pos = replay_object.data.root_pos_w[:1].clone()
    else:
        marker_pos = fallback_pos.clone()

    marker_pos[:, 2] += max(float(args_cli.object_size[2]) * 0.5, float(args_cli.contact_marker_radius))
    marker_pos[:, 2] += float(args_cli.contact_marker_height)
    return marker_pos


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
    )
    robot = G1_29DOF_TORSOBASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    replay_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReplayObject",
        spawn=sim_utils.MeshCuboidCfg(
            size=tuple(float(x) for x in args_cli.object_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def _make_replay_object_cfg(motion_path: Path) -> RigidObjectCfg:
    usd_path = _resolve_object_usd_from_metadata(motion_path)
    if usd_path is None:
        print(
            "[WARN] Could not resolve object USD from metadata for "
            f"{motion_path}. Falling back to proxy box with size={tuple(float(x) for x in args_cli.object_size)}."
        )
        return ReplaySceneCfg.replay_object

    print(f"[INFO] Using replay object USD: {usd_path}")
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReplayObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def _auto_fix_first_frame_ground_penetration(
    sim: SimulationContext,
    scene: InteractiveScene,
    motion: ReplayMotion,
    robot: Articulation,
    replay_object: RigidObject,
    sim_dt: float,
) -> None:
    """Use frame-0 pose to estimate penetration and lift trajectories to avoid ground clipping."""
    # Frame-0 target with current user-provided manual offsets applied.
    base_pos0 = motion.base_pos[:1].clone()
    base_quat0 = motion.base_quat[:1].clone()
    joint_pos0 = motion.joint_pos[:1]
    joint_vel0 = motion.joint_vel[:1]
    joint_vel0_write = joint_vel0 if args_cli.use_joint_vel else torch.zeros_like(joint_vel0)
    base_lin_vel0 = motion.base_lin_vel[:1]
    base_ang_vel0 = motion.base_ang_vel[:1]

    obj_pos0 = motion.object_pos[:1].clone()
    obj_quat0 = motion.object_quat[:1].clone()
    obj_lin_vel0 = motion.object_lin_vel[:1]
    obj_ang_vel0 = motion.object_ang_vel[:1]

    print("[Debug][replay-frame0] obj_pos0:", obj_pos0[0].detach().cpu().tolist())
    print("[Debug][replay-frame0] obj_quat0_wxyz:", obj_quat0[0].detach().cpu().tolist())
    print("[Debug][replay-frame0] obj_quat0_norm:", float(torch.linalg.norm(obj_quat0[0]).item()))

    base_pos0[:, 2] += float(args_cli.robot_height_offset)
    obj_pos0[:, 2] += float(args_cli.object_height_offset)

    robot.write_root_pose_to_sim(torch.cat([base_pos0, base_quat0], dim=-1))
    robot.write_root_velocity_to_sim(torch.cat([base_lin_vel0, base_ang_vel0], dim=-1))
    robot.write_joint_state_to_sim(joint_pos0, joint_vel0_write)

    replay_object.write_root_pose_to_sim(torch.cat([obj_pos0, obj_quat0], dim=-1))
    replay_object.write_root_velocity_to_sim(torch.cat([obj_lin_vel0, obj_ang_vel0], dim=-1))

    sim.render()
    scene.update(sim_dt)

    robot_root_z = robot.data.root_pos_w[0, 2].item()
    robot_min_z = _robot_min_z(robot)[0].item()
    obj_root_z, obj_bottom_z = _object_root_and_bottom_z(replay_object)
    obj_root_z_val = obj_root_z[0].item()
    obj_bottom_z_val = obj_bottom_z[0].item()

    clearance = float(args_cli.ground_clearance)
    robot_lift = max(0.0, clearance - robot_min_z)
    object_lift = max(0.0, clearance - obj_bottom_z_val)

    print(
        "[ZCHECK][frame0] "
        f"robot_root_z={robot_root_z:.4f}, robot_min_body_z={robot_min_z:.4f}, "
        f"object_root_z={obj_root_z_val:.4f}, object_bottom_z_est={obj_bottom_z_val:.4f}, "
        f"clearance={clearance:.4f}"
    )
    print(
        "[ZFIX][frame0] "
        f"robot_lift={robot_lift:.4f}m, object_lift={object_lift:.4f}m "
        "(applied to entire trajectory z)"
    )

    if robot_lift > 0.0:
        motion.base_pos[:, 2] += robot_lift
    if object_lift > 0.0:
        motion.object_pos[:, 2] += object_lift


def run_replay(sim: SimulationContext, scene: InteractiveScene, motion: ReplayMotion) -> None:
    robot: Articulation = scene["robot"]
    replay_object: RigidObject = scene["replay_object"]
    contact_marker = _create_contact_state_marker()

    if robot.num_instances != 1:
        raise RuntimeError(f"Expected 1 env/instance, got {robot.num_instances}")
    _remap_motion_joints_to_robot_order(motion, robot)
    if robot.num_joints != motion.joint_pos.shape[1]:
        raise RuntimeError(
            f"Robot joint count mismatch: robot={robot.num_joints}, data={motion.joint_pos.shape[1]}. "
            "Use a matching robot asset or add joint mapping."
        )
    
    sim_dt = float(sim.get_physics_dt())
    if args_cli.playback_rate <= 0.0:
        raise ValueError("--playback-rate must be > 0")
    frame_hold_steps = max(1, int(round((1.0 / motion.framerate) / sim_dt / args_cli.playback_rate)))

    print(
        f"[INFO] sim_dt={sim_dt:.6f}s, motion_dt={1.0/motion.framerate:.6f}s, "
        f"playback_rate={args_cli.playback_rate:.3f}, frame_hold_steps={frame_hold_steps}, "
        f"use_joint_vel={args_cli.use_joint_vel}"
    )

    print("[INFO] Frame-0 automatic ground-penetration correction is disabled for direct replay.")

    frame_idx = 0
    loops_done = 0
    step_count = 0
    peak = {
        "robot_base_pos_err": 0.0,
        "robot_base_quat_err_deg": 0.0,
        "robot_joint_pos_err_max": 0.0,
        "object_pos_err": 0.0,
        "object_quat_err_deg": 0.0,
    }

    while simulation_app.is_running():
        # target frame tensors [1, ...]
        base_pos = motion.base_pos[frame_idx : frame_idx + 1].clone()
        base_quat = motion.base_quat[frame_idx : frame_idx + 1].clone()
        joint_pos = motion.joint_pos[frame_idx : frame_idx + 1]
        joint_vel = motion.joint_vel[frame_idx : frame_idx + 1]
        joint_vel_to_write = joint_vel if args_cli.use_joint_vel else torch.zeros_like(joint_vel)
        base_lin_vel = motion.base_lin_vel[frame_idx : frame_idx + 1]
        base_ang_vel = motion.base_ang_vel[frame_idx : frame_idx + 1]

        obj_pos = motion.object_pos[frame_idx : frame_idx + 1].clone()
        obj_quat = motion.object_quat[frame_idx : frame_idx + 1].clone()
        obj_lin_vel = motion.object_lin_vel[frame_idx : frame_idx + 1]
        obj_ang_vel = motion.object_ang_vel[frame_idx : frame_idx + 1]

        base_pos[:, 2] += float(args_cli.robot_height_offset)
        obj_pos[:, 2] += float(args_cli.object_height_offset)

        robot.write_root_pose_to_sim(torch.cat([base_pos, base_quat], dim=-1))
        robot.write_root_velocity_to_sim(torch.cat([base_lin_vel, base_ang_vel], dim=-1))
        robot.write_joint_state_to_sim(joint_pos, joint_vel_to_write)

        replay_object.write_root_pose_to_sim(torch.cat([obj_pos, obj_quat], dim=-1))
        replay_object.write_root_velocity_to_sim(torch.cat([obj_lin_vel, obj_ang_vel], dim=-1))

        # Match the existing direct-visualization pattern in this repo for compatibility.
        sim.render()
        scene.update(sim_dt)

        contact_marker_indices = motion.object_contact[frame_idx : frame_idx + 1].to(dtype=torch.long)
        contact_marker_indices = torch.clamp(contact_marker_indices, min=0, max=1)
        contact_marker_pos = _compute_contact_marker_pose(replay_object, obj_pos)
        contact_marker.set_visibility(True)
        contact_marker.visualize(
            translations=contact_marker_pos,
            marker_indices=contact_marker_indices,
        )

        # Quality monitor: compare sim state with commanded state.
        sim_robot_base_pos = robot.data.root_pos_w[:1]
        sim_robot_base_quat = robot.data.root_quat_w[:1]
        sim_robot_joint_pos = robot.data.joint_pos[:1]
        sim_obj_pos = replay_object.data.root_pos_w[:1]
        sim_obj_quat = replay_object.data.root_quat_w[:1]

        robot_base_pos_err = torch.linalg.norm(sim_robot_base_pos - base_pos, dim=-1).item()
        robot_base_quat_err_deg = torch.rad2deg(_quat_ang_err_rad(base_quat, sim_robot_base_quat)).item()
        robot_joint_pos_err_max = torch.max(torch.abs(sim_robot_joint_pos - joint_pos)).item()
        object_pos_err = torch.linalg.norm(sim_obj_pos - obj_pos, dim=-1).item()
        object_quat_err_deg = torch.rad2deg(_quat_ang_err_rad(obj_quat, sim_obj_quat)).item()

        peak["robot_base_pos_err"] = max(peak["robot_base_pos_err"], robot_base_pos_err)
        peak["robot_base_quat_err_deg"] = max(peak["robot_base_quat_err_deg"], robot_base_quat_err_deg)
        peak["robot_joint_pos_err_max"] = max(peak["robot_joint_pos_err_max"], robot_joint_pos_err_max)
        peak["object_pos_err"] = max(peak["object_pos_err"], object_pos_err)
        peak["object_quat_err_deg"] = max(peak["object_quat_err_deg"], object_quat_err_deg)

        if frame_idx % max(1, args_cli.print_every) == 0 and step_count % frame_hold_steps == 0:
            contact_state = int(motion.object_contact[frame_idx].item() > 0)
            print(
                f"[REPLAY] frame={frame_idx:04d}/{motion.num_frames-1} "
                f"contact={contact_state} "
                f"rb_pos_err={robot_base_pos_err:.4e}m "
                f"rb_quat_err={robot_base_quat_err_deg:.3f}deg "
                f"joint_err_max={robot_joint_pos_err_max:.4e}rad "
                f"obj_pos_err={object_pos_err:.4e}m "
                f"obj_quat_err={object_quat_err_deg:.3f}deg"
            )

        step_count += 1
        if step_count % frame_hold_steps == 0:
            frame_idx += 1
            if frame_idx >= motion.num_frames:
                loops_done += 1
                print(
                    "[SUMMARY] one pass finished | "
                    + " ".join(f"{k}={v:.6g}" for k, v in peak.items())
                    + f" loops={loops_done}"
                )
                if not args_cli.loop:
                    break
                frame_idx = 0


def main() -> None:
    device = torch.device(args_cli.device)
    motion_path = _pick_motion_path()
    motion = load_replay_motion(motion_path, device=device)
    summarize_data_quality(motion)

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplaySceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=False,
        replay_object=_make_replay_object_cfg(motion_path),
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Direct replay scene ready.")
    run_replay(sim, scene, motion)
    simulation_app.close()


if __name__ == "__main__":
    main()

"""This is a tool that convert robot base pose to another frame on given robot chain"""

import argparse
import functools
import multiprocessing as mp
import numpy as np
import os
import pickle as pkl
import torch
import tqdm

import pytorch_kinematics as pk


class NumpyCompatUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        if module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        return super().find_class(module, name)


def load_pickle_numpy_compat(path):
    with open(path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def load_GMR_src_file(src_file):
    """Load from GMR source file"""
    # with open(src_file, "rb") as f:
    #     motion_data = pkl.load(f)
    motion_data = load_pickle_numpy_compat(src_file)
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    joint_pos = motion_data["dof_pos"]  # (N, 29)
    base_pos_w = motion_data["root_pos"]  # (N, 3)
    base_quat_w_ = motion_data["root_rot"]  # (N, 4), xyzw order
    base_quat_w = base_quat_w_[..., [3, 0, 1, 2]]  # convert to wxyz order
    framerate = motion_data["fps"]
    return {
        "joint_names": joint_names,
        "joint_pos": joint_pos,
        "base_pos_w": base_pos_w,
        "base_quat_w": base_quat_w,
        "framerate": framerate,
    }


def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation: a and b are (D,), t is scalar."""
    return a + t * (b - a)


def _slerp_quat(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions in wxyz order.

    Args:
        q0: (4,) source quaternion.
        q1: (4,) target quaternion.
        t: scalar blend factor in [0, 1].
    Returns:
        (4,) interpolated quaternion.
    """
    dot = (q0 * q1).sum().clamp(-1.0, 1.0)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / result.norm()
    theta_0 = torch.acos(dot)
    theta = theta_0 * t
    sin_theta_0 = torch.sin(theta_0)
    s0 = torch.cos(theta) - dot * torch.sin(theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def interpolate_se3(
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    input_fps: float,
    output_fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample SE3 motion data from input_fps to output_fps using lerp/slerp.

    Args:
        base_pos: (N, 3) root positions.
        base_quat: (N, 4) root quaternions in wxyz order.
        joint_pos: (N, D) joint positions.
        input_fps: framerate of input data.
        output_fps: desired output framerate.
    Returns:
        Tuple of resampled (base_pos, base_quat, joint_pos).
    """
    n_frames = base_pos.shape[0]
    duration = (n_frames - 1) / input_fps
    times = torch.arange(0, duration, 1.0 / output_fps, dtype=torch.float32)
    n_out = times.shape[0]

    out_pos = torch.zeros(n_out, base_pos.shape[1], dtype=base_pos.dtype)
    out_quat = torch.zeros(n_out, 4, dtype=base_quat.dtype)
    out_jpos = torch.zeros(n_out, joint_pos.shape[1], dtype=joint_pos.dtype)

    for i, t in enumerate(times):
        idx_f = t.item() * input_fps
        idx_0 = min(int(idx_f), n_frames - 2)
        idx_1 = idx_0 + 1
        blend = torch.tensor(idx_f - idx_0, dtype=torch.float32)
        out_pos[i] = _lerp(base_pos[idx_0], base_pos[idx_1], blend)
        out_quat[i] = _slerp_quat(base_quat[idx_0], base_quat[idx_1], blend)
        out_jpos[i] = _lerp(joint_pos[idx_0], joint_pos[idx_1], blend)

    return out_pos, out_quat, out_jpos


def convert_file(
    src_tgt_pairs,
    urdf: str,
    src_frame_name: str,
    tgt_frame_name: str,
    output_fps: int | None = None,
    joints_to_revert: list = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
):
    src_file, tgt_file = src_tgt_pairs

    src_npz = load_GMR_src_file(src_file)

    with open(urdf) as f:
        robot_chain = pk.build_chain_from_urdf(f.read())

    joint_pos = torch.as_tensor(src_npz["joint_pos"], dtype=torch.float32)
    joint_names = src_npz["joint_names"]
    src_base_pos_w = src_npz["base_pos_w"]
    src_base_quat_w = src_npz["base_quat_w"]
    src_base_transform = pk.Transform3d(
        rot=src_base_quat_w,  # wxyz order
        pos=src_base_pos_w,  # xyz order
    )

    # reverse joint names to match the new robot urdf if needed
    for joint_name in joints_to_revert:
        joint_idx_src = joint_names.index(joint_name)
        joint_pos[:, joint_idx_src] *= -1.0

    joint_order_src_to_pk = torch.zeros(len(joint_names), dtype=torch.long)
    for joint_i, joint_name in enumerate(robot_chain.get_joint_parameter_names()):
        joint_idx_src = joint_names.index(joint_name)
        joint_order_src_to_pk[joint_i] = joint_idx_src
    joint_pos_pk = joint_pos[:, joint_order_src_to_pk]

    src_tgt_frame_indices = robot_chain.get_frame_indices(src_frame_name, tgt_frame_name)
    frame_poses = robot_chain.forward_kinematics(joint_pos_pk, src_tgt_frame_indices)
    src_frame_poses = frame_poses[src_frame_name]  # pk.Transform3d
    tgt_frame_poses = frame_poses[tgt_frame_name]  # pk.Transform3d

    tgt_in_src_frame_poses = src_frame_poses.inverse().compose(tgt_frame_poses)  # pk.Transform3d
    tgt_base_transform = src_base_transform.compose(tgt_in_src_frame_poses)
    tgt_base_matrix = tgt_base_transform.get_matrix()  # (N, 4, 4)
    tgt_base_quat_w = pk.matrix_to_quaternion(tgt_base_matrix[:, :3, :3])
    tgt_base_pos_w = tgt_base_matrix[:, :3, 3]

    input_fps = float(src_npz["framerate"])
    actual_output_fps = float(output_fps) if output_fps is not None else input_fps

    if actual_output_fps != input_fps:
        tgt_base_pos_w, tgt_base_quat_w, joint_pos = interpolate_se3(
            tgt_base_pos_w, tgt_base_quat_w, joint_pos, input_fps, actual_output_fps
        )

    # pack the file and store
    np.savez(
        tgt_file,
        framerate=actual_output_fps,
        joint_names=joint_names,
        joint_pos=joint_pos.numpy(),
        base_pos_w=tgt_base_pos_w.detach().numpy() if isinstance(tgt_base_pos_w, torch.Tensor) else tgt_base_pos_w,
        base_quat_w=tgt_base_quat_w.detach().numpy() if isinstance(tgt_base_quat_w, torch.Tensor) else tgt_base_quat_w,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Input file or folder")
    parser.add_argument("--tgt", type=str, help="Target file or folder. Structure preserved if folder")
    parser.add_argument(
        "--urdf", type=str, help="Robot urdf to convert the base, does not need to match the source base or target base"
    )
    parser.add_argument("--src_frame", type=str, default="pelvis")
    parser.add_argument("--tgt_frame", type=str, default="torso_link")
    parser.add_argument(
        "--output_fps",
        type=int,
        default=None,
        help="Output framerate. Defaults to the source file's framerate. SE3 interpolation is applied when resampling.",
    )
    parser.add_argument("--num_cpus", default=10)

    args = parser.parse_args()

    # walk through the source folder and make folders in target folder if needed
    src_tgt_pairs = []
    if os.path.isfile(args.src):
        src_tgt_pairs.append((args.src, args.tgt))
    else:
        if not os.path.exists(args.tgt):
            os.makedirs(args.tgt, exist_ok=True)
        for root, _, filenames in os.walk(args.src):
            target_dirpath = os.path.join(args.tgt, os.path.relpath(root, args.src))
            os.makedirs(target_dirpath, exist_ok=True)
            for filename in filenames:
                if not filename.endswith(".pkl"):
                    continue
                src_tgt_pairs.append(
                    (
                        os.path.join(root, filename),
                        os.path.join(target_dirpath, filename.replace(".pkl", "_retargeted.npz")),
                    )
                )

    with mp.Pool(args.num_cpus) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(
                    functools.partial(
                        convert_file,
                        urdf=args.urdf,
                        src_frame_name=args.src_frame,
                        tgt_frame_name=args.tgt_frame,
                        output_fps=args.output_fps,
                    ),
                    src_tgt_pairs,
                ),
                total=len(src_tgt_pairs),
            )
        )


if __name__ == "__main__":
    main()

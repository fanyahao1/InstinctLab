"""
Script for generating dummy data for Instinct-Interaction-G1-v0 interaction tasks.
This module generates synthetic motion data for the G1 humanoid robot in interaction
scenarios. It creates numpy archive files (.npz) containing robot kinematics data,
base transformations, and object (box) state information.
Key Features:
    - Generates dummy motion sequences with configurable frame count
    - Supports 29 DOF G1 robot joint configuration
    - Includes base position/orientation and object tracking data
    - Automatically manages output directory structure
    - Loads motion file metadata from YAML configuration
Data Output Format (NPZ):
    - framerate (float): Motion capture frame rate in Hz
    - joint_names (array): Names of 29 robot joints
    - joint_pos (N, 29): Joint positions across N frames
    - base_pos_w (N, 3): Base position (x, y, z) in world frame
    - base_quat_w (N, 4): Base orientation as quaternion (w, x, y, z)
    - box_pos (N, 3): Object box position in world frame
    - box_quat (N, 4): Object box orientation as quaternion
Constants:
    OUTPUT_DIR (str): Directory path for generated data files
    METADATA_SRC (str): Path to source metadata YAML template
    METADATA_DST (str): Path to destination metadata YAML
    JOINT_NAMES (list): Names of 29 G1 robot joints
    FRAMERATE (float): Motion capture frame rate (50 Hz)
Functions:
    generate_motion(rel_path): Generate a single dummy motion file
    main(): Orchestrate dummy data generation workflow
"""

import numpy as np
import os
import yaml

# Configuration
OUTPUT_DIR = "/home/fan/dev3/project-instinct/InstinctLab/assets_datasets/interaction/data"
METADATA_SRC = "/home/fan/dev3/project-instinct/InstinctLab/assets_datasets/interaction/metadata_example.yaml"
METADATA_DST = os.path.join(OUTPUT_DIR, "metadata.yaml")

# G1 Joint Names aligned with reference
JOINT_NAMES = [
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

FRAMERATE = 50.0


def generate_motion(rel_path):
    full_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    num_frames = 100  # 2 seconds

    # Generate data
    framerate = np.array(FRAMERATE)
    joint_names = np.array(JOINT_NAMES)

    # (N, DOFs)
    joint_pos = np.zeros((num_frames, len(JOINT_NAMES)))

    # (N, 3)
    base_pos = np.zeros((num_frames, 3))
    base_pos[:, 2] = 0.75  # Height

    # (N, 4) w,x,y,z quaternion
    base_quat = np.zeros((num_frames, 4))
    base_quat[:, 0] = 1.0

    # Object (box) data
    # (N, 3)
    box_pos = np.zeros((num_frames, 3))
    box_pos[:, 0] = 1.0  # 1m in front
    box_pos[:, 2] = 0.5  # 0.5m high

    # (N, 4)
    box_quat = np.zeros((num_frames, 4))
    box_quat[:, 0] = 1.0

    np.savez(
        full_path,
        framerate=framerate,
        joint_names=joint_names,
        joint_pos=joint_pos,
        base_pos_w=base_pos,
        base_quat_w=base_quat,
        box_pos=box_pos,
        box_quat=box_quat,
    )
    print(f"Generated {full_path}")


def main():
    if not os.path.exists(METADATA_SRC):
        print(f"Error: Metadata source not found at {METADATA_SRC}")
        return

    with open(METADATA_SRC) as f:
        metadata = yaml.safe_load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for entry in metadata["motion_files"]:
        generate_motion(entry["motion_file"])

    with open(METADATA_DST, "w") as f:
        yaml.dump(metadata, f)
    print(f"Copied metadata to {METADATA_DST}")


if __name__ == "__main__":
    main()

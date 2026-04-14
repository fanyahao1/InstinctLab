# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.utils import configclass

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.interaction.config.g1.g1_interaction_shadowing_cfg as shadowing_g1_cfg
import instinctlab.tasks.interaction.mdp as interaction_mdp
from instinctlab.monitors import MonitorTermCfg


@configclass
class ObservationsCfg(shadowing_g1_cfg.interaction_cfg.ObservationsCfg):
    """Observation config for G1 sitting interaction shadowing."""

    @configclass
    class PolicyObsCfg(shadowing_g1_cfg.interaction_cfg.ObservationsCfg.PolicyObsCfg):
        depth_image = ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane_noised",
            },
        )
        object_pos = None
        object_ori = None
        object_pos_ref = None
        object_ori_ref = None
        object_pos_err = None
        object_ori_err = None
        wrist_object_contact = None
        seat_object_contact = None

    @configclass
    class CriticObsCfg(shadowing_g1_cfg.interaction_cfg.ObservationsCfg.CriticObsCfg):
        depth_image = ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane_noised",
            },
        )
        wrist_object_contact = None
        seat_object_contact = ObsTermCfg(
            func=interaction_mdp.seat_object_contact,
            params={
                "sensor_names": [
                    "pelvis_object_contact",
                    "left_hip_object_contact",
                    "right_hip_object_contact",
                ],
                "threshold": 1.0,
            },
        )

    policy: PolicyObsCfg = PolicyObsCfg()
    critic: CriticObsCfg = CriticObsCfg()


@configclass
class RewardGroupCfg(shadowing_g1_cfg.interaction_cfg.RewardGroupCfg):
    """Reward terms for G1 sitting interaction shadowing."""

    wrist_object_contact_ref_phase = None
    seat_object_contact_ref_phase = None
    sparse_contact_map = RewTermCfg(
        func=interaction_mdp.SparseContactReward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("objects"),
            "robot_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "box",
            "metadata_dir": None,
            "link_name_map": {
                "pelvis": "pelvis",
                "torso": "torso_link",
                "left_hand": "left_wrist_yaw_link",
                "right_hand": "right_wrist_yaw_link",
            },
            "contact_distance_threshold": 0.12,
            "forbid_contact_distance_threshold": 0.10,
            "mandatory_alpha": 10.0,
            "optional_beta": 8.0,
            "mandatory_weights": {
                "proximity": 0.35,
                "contact": 1.0,
                "hold": 0.5,
            },
            "optional_weights": {
                "proximity": 0.10,
                "contact": 0.25,
            },
            "forbidden_weight": 1.0,
            "hold_window": 5,
            "debug_vis": False,
            "debug_vis_max_envs": 1,
            "debug_vis_show_all_points": True,
            "debug_vis_show_nearest": True,
            "debug_vis_point_radius": 0.028,
            "debug_vis_nearest_point_radius": 0.04,
            "debug_vis_arrow_length_scale": 10.0,
            "debug_vis_arrow_thickness_scale": (1.0, 1.0, 1.0),
            "debug_vis_nearest_arrow_thickness_scale": (1.25, 1.25, 1.25),
            "debug_vis_part_names": ("seat", "back"),
        },
    )


@configclass
class SingleRewardsCfg(shadowing_g1_cfg.interaction_cfg.SingleRewardsCfg):
    rewards: RewardGroupCfg = RewardGroupCfg()


@configclass
class CurriculumCfg(shadowing_g1_cfg.interaction_cfg.CurriculumCfg):
    """Curriculum terms for G1 sitting interaction shadowing."""

    tracking_sigma_annealing = None


@configclass
class TerminationCfg(shadowing_g1_cfg.interaction_cfg.TerminationCfg):
    """Termination terms for G1 sitting interaction shadowing."""

    object_lower_body_contact = None
    object_nonseat_contact = DoneTermCfg(
        func=interaction_mdp.any_object_filtered_contact,
        time_out=False,
        params={
            "sensor_names": [
                "left_knee_object_contact",
                "right_knee_object_contact",
                "left_ankle_object_contact",
                "right_ankle_object_contact",
            ],
            "threshold": 1.0,
            "print_reason": False,
        },
    )


@configclass
class MonitorCfg(shadowing_g1_cfg.interaction_cfg.MonitorCfg):
    """Monitor terms for sparse sitting contact debugging."""

    sparse_contact_map = MonitorTermCfg(
        func=interaction_mdp.SparseContactMapMonitorTerm,
        params={
            "reward_group_name": "rewards",
            "reward_term_name": "sparse_contact_map",
        },
    )


@configclass
class G1InteractionSittingShadowingEnvCfg(shadowing_g1_cfg.G1InteractionShadowingEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: SingleRewardsCfg = SingleRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationCfg = TerminationCfg()
    monitors: MonitorCfg = MonitorCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = self.run_name.replace("G1InteractionShadowing", "G1InteractionSittingShadowing")
        self.run_name += "_sparseContactPhase"
        self.scene.camera.debug_vis = False
        self.observations.policy.depth_image.params["debug_vis"] = False
        self.rewards.rewards.sparse_contact_map.params["metadata_dir"] = os.path.join(
            shadowing_g1_cfg.MOTION_FOLDER, "sparse_contact_maps"
        )
        self.rewards.rewards.sparse_contact_map.params["debug_vis"] = False

        self.events.physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 1.2),
                "dynamic_friction_range": (0.8, 1.1),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 32,
            },
        )
        self.events.add_joint_default_pos = EventTermCfg(
            func=instinct_mdp.randomize_default_joint_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "offset_distribution_params": (-0.005, 0.005),
                "operation": "add",
                "distribution": "uniform",
            },
        )
        self.events.base_com = EventTermCfg(
            func=instinct_mdp.randomize_rigid_body_coms,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "coms_x_distribution_params": (-0.01, 0.01),
                "coms_y_distribution_params": (-0.02, 0.02),
                "coms_z_distribution_params": (-0.02, 0.02),
                "distribution": "uniform",
            },
        )
        self.events.object_scale = EventTermCfg(
            func=interaction_mdp.randomize_object_scale,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "scale_distribution_params": shadowing_g1_cfg.INTERACTION_OBJECT_SCALE_RANGE,
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        )

        self.events.reset_robot.params["randomize_pose_range"].update(
            {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
            }
        )
        self.events.reset_robot.params["randomize_joint_pos_range"] = (-0.1, 0.1)
        self.events.reset_objects.params["randomize_pose_range"].update(
            {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
            }
        )


@configclass
class G1InteractionSittingShadowingEnvCfg_PLAY(shadowing_g1_cfg.G1InteractionShadowingEnvCfg_PLAY):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: SingleRewardsCfg = SingleRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationCfg = TerminationCfg()
    monitors: MonitorCfg = MonitorCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = self.run_name.replace("G1InteractionShadowing", "G1InteractionSittingShadowing")
        self.run_name += "_sparseContactPhase"
        self.scene.motion_reference.debug_vis = False
        self.scene.motion_reference.visualizing_marker_types = []
        self.scene.robot_reference = None
        self.scene.object_reference = None
        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True
        self.rewards.rewards.sparse_contact_map.params["metadata_dir"] = os.path.join(
            shadowing_g1_cfg.MOTION_FOLDER, "sparse_contact_maps"
        )
        self.rewards.rewards.sparse_contact_map.params["debug_vis"] = True
        # self.rewards.rewards.sparse_contact_map.params["debug_vis_max_envs"] = 1
        # self.rewards.rewards.sparse_contact_map.params["debug_vis_point_radius"] = 0.12
        # self.rewards.rewards.sparse_contact_map.params["debug_vis_nearest_point_radius"] = 0.18
        # self.rewards.rewards.sparse_contact_map.params["debug_vis_arrow_length_scale"] = 10.0
        # self.rewards.rewards.sparse_contact_map.params["debug_vis_arrow_thickness_scale"] = (1.0, 1.0, 1.0)
        # self.rewards.rewards.sparse_contact_map.params["debug_vis_nearest_arrow_thickness_scale"] = (1.25, 1.25, 1.25)
        # self.rewards.rewards.sparse_contact_map.params["debug_vis_part_names"] = ("seat", "back")
        self.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        )

        self.events.reset_robot.params["randomize_pose_range"].update(
            {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
            }
        )
        self.events.reset_robot.params["randomize_joint_pos_range"] = (-0.1, 0.1)
        self.events.reset_objects.params["randomize_pose_range"].update(
            {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
            }
        )

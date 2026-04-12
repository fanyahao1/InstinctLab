# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.interaction.mdp as interaction_mdp
import instinctlab.tasks.interaction.config.g1.g1_interaction_shadowing_cfg as shadowing_g1_cfg
import instinctlab.tasks.interaction.config.interaction_sitting_shadowing_cfg as sitting_cfg


@configclass
class G1InteractionSittingShadowingEnvCfg(shadowing_g1_cfg.G1InteractionShadowingEnvCfg):
    observations: sitting_cfg.ObservationsCfg = sitting_cfg.ObservationsCfg()
    rewards: sitting_cfg.SingleRewardsCfg = sitting_cfg.SingleRewardsCfg()
    curriculum: sitting_cfg.CurriculumCfg = sitting_cfg.CurriculumCfg()
    terminations: sitting_cfg.TerminationCfg = sitting_cfg.TerminationCfg()
    monitors: sitting_cfg.MonitorCfg = sitting_cfg.MonitorCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = self.run_name.replace("G1InteractionShadowing", "G1InteractionSittingShadowing")
        self.run_name += "_sparseContactPhase"
        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = False
        self.rewards.rewards.sparse_contact_map.params["metadata_dir"] = os.path.join(
            shadowing_g1_cfg.MOTION_FOLDER, "sparse_contact_maps"
        )
        self.rewards.rewards.sparse_contact_map.params["debug_vis"] = True
        self.rewards.rewards.sparse_contact_map.params["debug_vis_max_envs"] = 1
        self.rewards.rewards.sparse_contact_map.params["debug_vis_point_radius"] = 0.028
        self.rewards.rewards.sparse_contact_map.params["debug_vis_nearest_point_radius"] = 0.04
        self.rewards.rewards.sparse_contact_map.params["debug_vis_arrow_length_scale"] = 10.0
        self.rewards.rewards.sparse_contact_map.params["debug_vis_arrow_thickness_scale"] = (1.0, 1.0, 1.0)
        self.rewards.rewards.sparse_contact_map.params["debug_vis_nearest_arrow_thickness_scale"] = (1.25, 1.25, 1.25)

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

        self.events.reset_robot.params["randomize_pose_range"].update(
            {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
            }
        )
        self.events.reset_objects.params["randomize_pose_range"].update(
            {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
            }
        )


@configclass
class G1InteractionSittingShadowingEnvCfg_PLAY(shadowing_g1_cfg.G1InteractionShadowingEnvCfg_PLAY):
    observations: sitting_cfg.ObservationsCfg = sitting_cfg.ObservationsCfg()
    rewards: sitting_cfg.SingleRewardsCfg = sitting_cfg.SingleRewardsCfg()
    curriculum: sitting_cfg.CurriculumCfg = sitting_cfg.CurriculumCfg()
    terminations: sitting_cfg.TerminationCfg = sitting_cfg.TerminationCfg()
    monitors: sitting_cfg.MonitorCfg = sitting_cfg.MonitorCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = self.run_name.replace("G1InteractionShadowing", "G1InteractionSittingShadowing")
        self.run_name += "_sparseContactPhase"
        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True
        self.rewards.rewards.sparse_contact_map.params["metadata_dir"] = os.path.join(
            shadowing_g1_cfg.MOTION_FOLDER, "sparse_contact_maps"
        )
        self.rewards.rewards.sparse_contact_map.params["debug_vis"] = True
        self.rewards.rewards.sparse_contact_map.params["debug_vis_max_envs"] = 1
        self.rewards.rewards.sparse_contact_map.params["debug_vis_point_radius"] = 0.12
        self.rewards.rewards.sparse_contact_map.params["debug_vis_nearest_point_radius"] = 0.18
        self.rewards.rewards.sparse_contact_map.params["debug_vis_arrow_length_scale"] = 10.0
        self.rewards.rewards.sparse_contact_map.params["debug_vis_arrow_thickness_scale"] = (1.0, 1.0, 1.0)
        self.rewards.rewards.sparse_contact_map.params["debug_vis_nearest_arrow_thickness_scale"] = (1.25, 1.25, 1.25)

        self.events.reset_robot.params["randomize_pose_range"].update(
            {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
            }
        )
        self.events.reset_objects.params["randomize_pose_range"].update(
            {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
            }
        )

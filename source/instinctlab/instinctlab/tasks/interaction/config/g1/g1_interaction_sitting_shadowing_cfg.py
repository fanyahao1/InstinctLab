# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass

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

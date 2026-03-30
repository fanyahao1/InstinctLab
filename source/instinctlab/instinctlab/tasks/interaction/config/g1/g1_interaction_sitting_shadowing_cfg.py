# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import instinctlab.tasks.interaction.config.g1.g1_interaction_shadowing_cfg as shadowing_g1_cfg
import instinctlab.tasks.interaction.config.interaction_sitting_shadowing_cfg as sitting_cfg


@configclass
class G1InteractionSittingShadowingEnvCfg(shadowing_g1_cfg.G1InteractionShadowingEnvCfg):
    observations: sitting_cfg.ObservationsCfg = sitting_cfg.ObservationsCfg()
    rewards: sitting_cfg.SingleRewardsCfg = sitting_cfg.SingleRewardsCfg()
    curriculum: sitting_cfg.CurriculumCfg = sitting_cfg.CurriculumCfg()
    terminations: sitting_cfg.TerminationCfg = sitting_cfg.TerminationCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = self.run_name.replace("G1InteractionShadowing", "G1InteractionSittingShadowing")
        self.run_name += "_contactPhaseOnly"


@configclass
class G1InteractionSittingShadowingEnvCfg_PLAY(shadowing_g1_cfg.G1InteractionShadowingEnvCfg_PLAY):
    observations: sitting_cfg.ObservationsCfg = sitting_cfg.ObservationsCfg()
    rewards: sitting_cfg.SingleRewardsCfg = sitting_cfg.SingleRewardsCfg()
    curriculum: sitting_cfg.CurriculumCfg = sitting_cfg.CurriculumCfg()
    terminations: sitting_cfg.TerminationCfg = sitting_cfg.TerminationCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = self.run_name.replace("G1InteractionShadowing", "G1InteractionSittingShadowing")
        self.run_name += "_contactPhaseOnly"

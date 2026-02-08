# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import yaml
from functools import partial

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.interaction.config.interaction_shadowing_cfg as interaction_cfg

##
# Pre-defined configs
##
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.monitors import (
    ActuatorMonitorTerm,
    MonitorTermCfg,
    RewardSumMonitorTerm,
    ShadowingBasePosMonitorTerm,
    ShadowingJointReferenceMonitorTerm,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.motion_files.object_motion_cfg import ObjectMotionCfg as ObjectMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.utils.humanoid_ik import HumanoidSmplRotationalIK

combine_method = "prod"
G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

MOTION_NAME = "InteractionMotion"


@configclass
class InteractionMotionCfg(ObjectMotionCfgBase):
    path = os.path.expanduser("PATH/TO/INTERACTION/MOTION/DATA")
    object_data_keys = (
        {
            "box": "box",
        },
    )
    object_velocity_estimation_method = "frontbackward"

    filtered_motion_selection_filepath = None
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    motion_start_height_offset = 0.0
    motion_bin_length_s = 1.0
    env_starting_stub_sampling_strategy = "independent"
    motion_start_from_middle_range = [0.0, 0.0]


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    data_start_from="current_time",
    visualizing_robot_offset=(0.0, 1.5, 0.0),
    visualizing_robot_from="reference_frame",
    motion_buffers={
        MOTION_NAME: InteractionMotionCfg(),
    },
    mp_split_method="Even",
)

DUNMMY_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=[
            "PATH/TO/YOUR/OBJECT/FILE1",
            "PATH/TO/YOUR/OBJECT/FILE2",
        ],
        random_choice=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
)


@configclass
class G1InteractionShadowingEnvCfg(interaction_cfg.InteractionShadowingEnvCfg):
    scene: interaction_cfg.InteractionShadowingSceneCfg = interaction_cfg.InteractionShadowingSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
        objects=DUNMMY_OBJECT_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        # add link_of_interests to the policy observation
        if self.observations.policy.__dict__.get("link_pos", None) is not None:
            self.observations.policy.link_pos.params["asset_cfg"].body_names = (
                self.scene.motion_reference.link_of_interests
            )
        if self.observations.policy.__dict__.get("link_rot", None) is not None:
            self.observations.policy.link_rot.params["asset_cfg"].body_names = (
                self.scene.motion_reference.link_of_interests
            )
        if hasattr(self.observations, "critic") and self.observations.critic is not None:
            if self.observations.critic.__dict__.get("link_pos", None) is not None:
                self.observations.critic.link_pos.params["asset_cfg"].body_names = (
                    self.scene.motion_reference.link_of_interests
                )
            if self.observations.critic.__dict__.get("link_rot", None) is not None:
                self.observations.critic.link_rot.params["asset_cfg"].body_names = (
                    self.scene.motion_reference.link_of_interests
                )

        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        self.actions.joint_pos.scale = beyondmimic_action_scale

        assert (
            len(list(self.scene.motion_reference.motion_buffers.keys())) == 1
        ), "Only support single motion buffer for now"
        motion_buffer = list(self.scene.motion_reference.motion_buffers.values())[0]
        if motion_buffer.motion_bin_length_s is not None:
            if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins":
                self.curriculum.beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
                    func=instinct_mdp.BeyondConcatMotionAdaptiveWeighting,
                )
            elif motion_buffer.env_starting_stub_sampling_strategy == "independent":
                self.curriculum.beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
                    func=instinct_mdp.BeyondMimicAdaptiveWeighting,
                )
            else:
                raise ValueError(
                    "Unsupported env starting stub sampling method:"
                    f" {motion_buffer.env_starting_stub_sampling_strategy}"
                )

        self.run_name: str = "".join(
            [
                "G1InteractionShadowing",
                f"_{MOTION_NAME}",
                (
                    "_odomObs"
                    if ("base_lin_vel" in self.observations.policy.__dict__.keys())
                    and self.commands.position_b_ref_command.anchor_frame == "robot"
                    else ""
                ),
                # (
                #     "_" + "-".join(self.scene.motion_reference.motion_buffers.keys())
                #     if self.scene.motion_reference.motion_buffers
                #     else ""
                # ),
                # (
                #     f"_proprioHist{self.observations.policy.joint_pos.history_length}"
                #     if self.observations.policy.joint_pos.history_length > 0
                #     else ""
                # ),
                # (
                #     f"_futureRef{self.scene.motion_reference.num_frames}"
                #     if self.scene.motion_reference.num_frames > 1
                #     else ""
                # ),
                # f"_FrameStartFrom{self.scene.motion_reference.data_start_from}",
                # "_forLoopMotionWeights",
                # "_forLoopMotionSample",
                ("_pgTermXYalso" if not self.terminations.base_pg_too_far.params["z_only"] else ""),
                (
                    "_concatMotionBins"
                    if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
                "_fixFramerate_diveroll4",
            ]
        )


@configclass
class G1InteractionShadowingEnvCfg_PLAY(G1InteractionShadowingEnvCfg):
    scene: interaction_cfg.InteractionShadowingSceneCfg = interaction_cfg.InteractionShadowingSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=[4.0, 0.75, 1.0],
        lookat=[0.0, 0.75, 0.0],
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]

        self.curriculum.beyond_adaptive_sampling = None
        self.events.push_robot = None
        self.events.bin_fail_counter_smoothing = None

        # enable print_reason option in the termination terms
        for term in self.terminations.__dict__.values():
            if term is None:
                continue
            if "print_reason" in term.params:
                term.params["print_reason"] = True
        # self.episode_length_s = 10.0
        # for term_name, term in self.terminations.__dict__.items():
        #     if (not term_name == "dataset_exhausted") and (not term_name == "time_out"):
        #         self.terminations.__dict__[term_name] = None

        # enable debug_vis option in commands
        for cmd in self.commands.__dict__.values():
            cmd.debug_vis = True

        # add PLAY-specific monitor term
        # self.monitors.shoulder_actuator = MonitorTermCfg(
        #     func=ActuatorMonitorTerm,
        #     params={
        #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_shoulder_roll.*"),
        #         "torque_plot_scale": 1e-2,
        #         # "joint_vel_plot_scale": 1e-1,
        #         "joint_power_plot_scale": 1e-1,
        #     },
        # )
        # self.monitors.waist_actuator = MonitorTermCfg(
        #     func=ActuatorMonitorTerm,
        #     params={
        #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="waist_roll.*"),
        #         "torque_plot_scale": 1e-2,
        #         # "joint_vel_plot_scale": 1e-1,
        #         "joint_power_plot_scale": 1e-1,
        #     },
        # )
        # self.monitors.knee_actuator = MonitorTermCfg(
        #     func=ActuatorMonitorTerm,
        #     params={
        #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_knee.*"),
        #         "torque_plot_scale": 1e-2,
        #         # "joint_vel_plot_scale": 1e-1,
        #         "joint_power_plot_scale": 1e-1,
        #     },
        # )
        # self.monitors.reward_sum = MonitorTermCfg(
        #     func=RewardSumMonitorTerm,
        # )
        # self.monitors.reference_stat_case = MonitorTermCfg(
        #     func=ShadowingJointReferenceMonitorTerm,
        #     params=dict(
        #         reference_cfg=SceneEntityCfg(
        #             "motion_reference",
        #             joint_names=[
        #                 "left_hip_pitch.*",
        #             ],
        #         ),
        #     ),
        # )
        # self.monitors.shadowing_base_pos = MonitorTermCfg(
        #     func=ShadowingBasePosMonitorTerm,
        #     params=dict(
        #         robot_cfg=SceneEntityCfg("robot"),
        #         motion_reference_cfg=SceneEntityCfg("motion_reference"),
        #     ),
        # )

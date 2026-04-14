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
from isaaclab.managers import CurriculumTermCfg, EventTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.interaction.config.interaction_shadowing_cfg as interaction_cfg
import instinctlab.tasks.interaction.mdp as interaction_mdp

##
# Pre-defined configs
##
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_CFG,
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
from instinctlab.motion_reference import MotionReferenceManagerCfg, NoCollisionPropertiesCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.motion_files.object_motion_cfg import ObjectMotionCfg as ObjectMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear

combine_method = "prod"
G1_CFG = G1_29DOF_TORSOBASE_CFG

MOTION_FOLDER = "/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object"


@configclass
class InteractionMotionCfg(ObjectMotionCfgBase):
    path = MOTION_FOLDER
    metadata_yaml = os.path.join(MOTION_FOLDER, "metadata.yaml")
    object_matching_key = "usd_path"
    object_data_keys = {
        "box": "box",
    }
    object_velocity_estimation_method = "frontbackward"

    filtered_motion_selection_filepath = None
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    motion_start_height_offset = 0.0
    motion_bin_length_s = None
    env_starting_stub_sampling_strategy = "independent"
    motion_start_from_middle_range = [0.0, 0.0]


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference",
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
    visualizing_robot_offset=(0.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    motion_buffers={
        "InteractionMotion": InteractionMotionCfg(),
    },
    mp_split_method="Even",
)

G1_REFERENCE_CFG = G1_CFG.copy()
G1_REFERENCE_CFG.spawn.activate_contact_sensors = False
G1_REFERENCE_CFG.spawn.collision_props = NoCollisionPropertiesCfg()
G1_REFERENCE_CFG.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.08, 0.76, 1.0),
    opacity=0.3,
    roughness=0.25,
)

INTERACTION_OBJECT_USD_PATHS = [
    os.path.join(MOTION_FOLDER, "chair_obj/chair.usd"),
    # os.path.join(MOTION_FOLDER, "sofa_obj/sofa.usd"),
]
INTERACTION_OBJECT_SCALE_RANGE = (0.7, 1.1)

INTERACTION_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=INTERACTION_OBJECT_USD_PATHS,
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
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
)

INTERACTION_OBJECT_REFERENCE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ObjectReference",
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=INTERACTION_OBJECT_USD_PATHS,
        random_choice=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.08, 0.76, 1.0),
            opacity=0.28,
            roughness=0.25,
        ),
        activate_contact_sensors=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
)


@configclass
class G1InteractionShadowingEnvCfg(interaction_cfg.InteractionShadowingEnvCfg):
    scene: interaction_cfg.InteractionShadowingSceneCfg = interaction_cfg.InteractionShadowingSceneCfg(
        replicate_physics=False,
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
        objects=INTERACTION_OBJECT_CFG,
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
                "scale_distribution_params": INTERACTION_OBJECT_SCALE_RANGE,
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        assert (
            len(list(self.scene.motion_reference.motion_buffers.keys())) == 1
        ), "Only support single motion buffer for now"
        motion_buffer = list(self.scene.motion_reference.motion_buffers.values())[0]
        if motion_buffer.motion_bin_length_s is not None:
            if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins":
                self.curriculum.beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
                    func=interaction_mdp.BeyondConcatMotionAdaptiveWeighting,
                )
            elif motion_buffer.env_starting_stub_sampling_strategy == "independent":
                self.curriculum.beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
                    func=interaction_mdp.BeyondMimicAdaptiveWeighting,
                )
            else:
                raise ValueError(
                    "Unsupported env starting stub sampling method:"
                    f" {motion_buffer.env_starting_stub_sampling_strategy}"
                )
        else:
            self.curriculum.beyond_adaptive_sampling = None
            self.events.bin_fail_counter_smoothing = None

        self.run_name: str = "".join(
            [
                "G1InteractionShadowing",
                (
                    "_odomObs"
                    if ("base_lin_vel" in self.observations.policy.__dict__.keys())
                    and self.commands.position_b_ref_command.anchor_frame == "robot"
                    else ""
                ),
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
        replicate_physics=False,
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_REFERENCE_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
        objects=INTERACTION_OBJECT_CFG,
        object_reference=INTERACTION_OBJECT_REFERENCE_CFG,
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
        self.scene.motion_reference.visualizing_marker_types = ["links"]

        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        # TODO(interaction): object trail visualization is temporarily disabled.
        self.events.update_object_reference_vis = None
        self.events.reset_object_reference_trail = None

        # Re-enable mild domain randomization for play/eval.
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
                "scale_distribution_params": INTERACTION_OBJECT_SCALE_RANGE,
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.push_robot = None

        # enable print_reason option in the termination terms
        for term in self.terminations.__dict__.values():
            if term is None:
                continue
            if "print_reason" in term.params:
                term.params["print_reason"] = True

        # enable debug_vis option in commands
        for cmd in self.commands.__dict__.values():
            cmd.debug_vis = True

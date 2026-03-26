import os
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import CurriculumTermCfg, EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.interaction.mdp as interaction_mdp
from instinctlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinctlab.managers import MultiRewardCfg
from instinctlab.monitors import (
    BodyStatMonitorTerm,
    JointStatMonitorTerm,
    MonitorTermCfg,
    MotionReferenceMonitorTerm,
    ShadowingJointPosMonitorTerm,
    ShadowingJointVelMonitorTerm,
    ShadowingLinkPosMonitorTerm,
    ShadowingPositionMonitorTerm,
    ShadowingProgressMonitorTerm,
    ShadowingRotationMonitorTerm,
    ShadowingVelocityMonitorTerm,
    TorqueMonitorSensorCfg,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.terrains.height_field import PerlinPlaneTerrainCfg


@configclass
class InteractionShadowingSceneCfg(InteractiveSceneCfg):
    """Configuration for the BeyondMimic scene with necessary scene entities as motion reference."""

    env_spacing = 4.0

    # robots
    robot: ArticulationCfg = MISSING

    # objects
    objects: RigidObjectCfg = MISSING

    # visualization-only object reference
    object_reference: RigidObjectCfg = None

    # robot reference articulation with object motion
    robot_reference: ArticulationCfg = None

    # motion reference
    motion_reference: MotionReferenceManagerCfg = MISSING  # type: ignore

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )
    hand_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )
    left_wrist_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )
    right_wrist_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )
    pelvis_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )
    left_hip_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hip_roll_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )
    right_hip_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hip_roll_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )
    left_knee_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_knee_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )
    right_knee_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_knee_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )
    left_ankle_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )
    right_ankle_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )

    def __post_init__(self):
        if type(self.motion_reference) is type(MISSING) or not self.motion_reference.debug_vis:
            delattr(self, "robot_reference")
            delattr(self, "object_reference")


@configclass
class CommandCfg:
    """BeyondMimic command configuration following their approach."""

    position_ref_command = instinct_mdp.PositionRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        anchor_frame="robot",
    )
    position_b_ref_command = instinct_mdp.PositionRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        anchor_frame="reference",
    )
    rotation_ref_command = instinct_mdp.RotationRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        in_base_frame=True,
        rotation_mode="tannorm",
    )
    joint_pos_ref_command = instinct_mdp.JointPosRefCommandCfg(current_state_command=False)
    joint_vel_ref_command = instinct_mdp.JointVelRefCommandCfg(current_state_command=False)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
    )


@configclass
class ObservationsCfg:
    """BeyondMimic observation configuration following their approach."""

    @configclass
    class PolicyObsCfg(ObsGroupCfg):
        """Policy observations for BeyondMimic."""

        # BeyondMimic specific reference observations
        joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
        joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
        position_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_b_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        )
        rotation_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        object_pos = ObsTermCfg(
            func=interaction_mdp.object_position,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "robot_cfg": SceneEntityCfg("robot"),
                "in_base_frame": True,
            },
            noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
        )
        object_ori = ObsTermCfg(
            func=interaction_mdp.object_orientation_tannorm,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "robot_cfg": SceneEntityCfg("robot"),
                "in_base_frame": True,
            },
            noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
        )
        object_pos_ref = ObsTermCfg(
            func=interaction_mdp.object_reference_position,
            params={
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "robot_cfg": SceneEntityCfg("robot"),
                "object_name": "box",
                "in_base_frame": True,
            },
            noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
        )
        object_ori_ref = ObsTermCfg(
            func=interaction_mdp.object_reference_orientation_tannorm,
            params={
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "robot_cfg": SceneEntityCfg("robot"),
                "object_name": "box",
                "in_base_frame": True,
            },
            noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
        )
        object_pos_err = ObsTermCfg(
            func=interaction_mdp.object_position_error,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "robot_cfg": SceneEntityCfg("robot"),
                "object_name": "box",
                "in_base_frame": True,
            },
            noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
        )
        object_ori_err = ObsTermCfg(
            func=interaction_mdp.object_orientation_error_tannorm,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "object_name": "box",
            },
            noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
        )
        wrist_object_contact = ObsTermCfg(
            func=interaction_mdp.wrist_object_contact,
            params={
                "left_sensor_cfg": SceneEntityCfg("left_wrist_object_contact"),
                "right_sensor_cfg": SceneEntityCfg("right_wrist_object_contact"),
                "threshold": 1.0,
            },
        )
        # object_lin_vel = ObsTermCfg(
        #     func=interaction_mdp.object_linear_velocity,
        #     params={
        #         "asset_cfg": SceneEntityCfg("objects"),
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "in_base_frame": True,
        #     },
        #     noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        # )
        # object_lin_vel_err = ObsTermCfg(
        #     func=interaction_mdp.object_linear_velocity_error,
        #     params={
        #         "asset_cfg": SceneEntityCfg("objects"),
        #         "reference_cfg": SceneEntityCfg("motion_reference"),
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "object_name": "box",
        #         "in_base_frame": True,
        #     },
        #     noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        # )
        # object_ang_vel_err = ObsTermCfg(
        #     func=interaction_mdp.object_angular_velocity_error,
        #     params={
        #         "asset_cfg": SceneEntityCfg("objects"),
        #         "reference_cfg": SceneEntityCfg("motion_reference"),
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "object_name": "box",
        #         "in_base_frame": True,
        #     },
        #     noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        # )

        # proprioception
        # base_lin_vel = ObsTermCfg(
        #     func=mdp.base_lin_vel,
        #     noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        # )
        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticObsCfg(ObsGroupCfg):
        """Critic observations for BeyondMimic."""

        # BeyondMimic specific reference observations
        joint_pos_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
        )
        joint_vel_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
        )
        position_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_ref_command"},
        )
        rotation_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
        )
        object_pos = ObsTermCfg(
            func=interaction_mdp.object_position,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "robot_cfg": SceneEntityCfg("robot"),
                "in_base_frame": False,
            },
        )
        object_ori = ObsTermCfg(
            func=interaction_mdp.object_orientation_tannorm,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "robot_cfg": SceneEntityCfg("robot"),
                "in_base_frame": False,
            },
        )
        object_pos_ref = ObsTermCfg(
            func=interaction_mdp.object_reference_position,
            params={
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "robot_cfg": SceneEntityCfg("robot"),
                "object_name": "box",
                "in_base_frame": False,
            },
        )
        object_ori_ref = ObsTermCfg(
            func=interaction_mdp.object_reference_orientation_tannorm,
            params={
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "robot_cfg": SceneEntityCfg("robot"),
                "object_name": "box",
                "in_base_frame": False,
            },
        )
        object_pos_err = ObsTermCfg(
            func=interaction_mdp.object_position_error,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "robot_cfg": SceneEntityCfg("robot"),
                "object_name": "box",
                "in_base_frame": False,
            },
        )
        object_ori_err = ObsTermCfg(
            func=interaction_mdp.object_orientation_error_tannorm,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "object_name": "box",
            },
        )
        wrist_object_contact = ObsTermCfg(
            func=interaction_mdp.wrist_object_contact,
            params={
                "left_sensor_cfg": SceneEntityCfg("left_wrist_object_contact"),
                "right_sensor_cfg": SceneEntityCfg("right_wrist_object_contact"),
                "threshold": 1.0,
            },
        )
        object_lin_vel = ObsTermCfg(
            func=interaction_mdp.object_linear_velocity,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "robot_cfg": SceneEntityCfg("robot"),
                "in_base_frame": False,
            },
        )
        object_ang_vel = ObsTermCfg(
            func=interaction_mdp.object_angular_velocity,
            params={
                "asset_cfg": SceneEntityCfg("objects"),
                "robot_cfg": SceneEntityCfg("robot"),
                "in_base_frame": False,
            },
        )
        # object_lin_vel_err = ObsTermCfg(
        #     func=interaction_mdp.object_linear_velocity_error,
        #     params={
        #         "asset_cfg": SceneEntityCfg("objects"),
        #         "reference_cfg": SceneEntityCfg("motion_reference"),
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "object_name": "box",
        #         "in_base_frame": False,
        #     },
        # )
        # object_ang_vel_err = ObsTermCfg(
        #     func=interaction_mdp.object_angular_velocity_error,
        #     params={
        #         "asset_cfg": SceneEntityCfg("objects"),
        #         "reference_cfg": SceneEntityCfg("motion_reference"),
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "object_name": "box",
        #         "in_base_frame": False,
        #     },
        # )

        # proprioception
        link_pos = ObsTermCfg(
            func=instinct_mdp.link_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=MISSING,
                    preserve_order=True,
                ),
            },
        )
        link_rot = ObsTermCfg(
            func=instinct_mdp.link_tannorm_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=MISSING,
                    preserve_order=True,
                ),
            },
        )
        base_lin_vel = ObsTermCfg(
            func=mdp.base_lin_vel,
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
        )
        last_action = ObsTermCfg(
            func=mdp.last_action,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy = PolicyObsCfg()
    critic = CriticObsCfg()


@configclass
class RewardGroupCfg:
    """BeyondMimic reward terms following their approach."""

    base_position_imitation_gauss = RewTermCfg(
        func=instinct_mdp.base_position_imitation_gauss,
        weight=0.5,
        params={
            "std": 0.3,
        },
    )
    base_rot_imitation_gauss = RewTermCfg(
        func=instinct_mdp.base_rot_imitation_gauss,
        weight=0.5,
        params={
            "std": 0.4,
            "difference_type": "axis_angle",
        },
    )
    link_pos_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_pos_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": 0.3,
        },
    )
    link_rot_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_rot_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": 0.4,
        },
    )
    link_lin_vel_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_lin_vel_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "std": 1.0,
        },
    )
    link_ang_vel_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_ang_vel_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "std": 3.14,
        },
    )
    object_pos_tracking_gauss = RewTermCfg(
        func=interaction_mdp.object_position_tracking_gauss,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("objects"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "box",
            # Looser tracking target than body tracking to avoid over-penalizing contact-heavy scenes.
            "tracking_torlerance": 0.25,
            "tracking_sigma": 0.8,
        },
    )
    object_rot_tracking_gauss = RewTermCfg(
        func=interaction_mdp.object_rotation_tracking_gauss,
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("objects"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "box",
            # Larger orientation tolerance for robust object interaction.
            "tracking_torlerance": 0.6,
            "tracking_sigma": 1.0,
        },
    )
    wrist_object_contact_ref_phase = RewTermCfg(
        func=interaction_mdp.object_contact_reference_phase,
        weight=3.0,
        params={
            "sensor_names": [
                "left_wrist_object_contact",
                "right_wrist_object_contact",
            ],
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "box",
            "threshold": 1.0,
            "normalize": True,
            "print_reason": False,
            "debug_label": "wrist_object_contact",
        },
    )
    # object_lin_vel_tracking_gauss = RewTermCfg(
    #     func=interaction_mdp.object_linear_velocity_tracking_gauss,
    #     weight=0.35,
    #     params={
    #         "asset_cfg": SceneEntityCfg("objects"),
    #         "reference_cfg": SceneEntityCfg("motion_reference"),
    #         "object_name": "box",
    #         "tracking_tolerance": 0.2,
    #         "tracking_sigma": 1.2,
    #     },
    # )
    # object_ang_vel_tracking_gauss = RewTermCfg(
    #     func=interaction_mdp.object_angular_velocity_tracking_gauss,
    #     weight=0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg("objects"),
    #         "reference_cfg": SceneEntityCfg("motion_reference"),
    #         "object_name": "box",
    #         "tracking_tolerance": 0.35,
    #         "tracking_sigma": 2.5,
    #     },
    # )
    action_rate_l2 = RewTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = None


@configclass
class SingleRewardsCfg(MultiRewardCfg):
    """Single reward configuration for BeyondMimic."""

    rewards: RewardGroupCfg = RewardGroupCfg()


@configclass
class EventCfg:
    """BeyondMimic events config such as termination conditions."""

    # startup
    physics_material = None

    add_joint_default_pos = None

    base_com = None

    # interval
    push_robot = None

    # for motion initialization and reset
    match_motion_ref_with_scene = EventTermCfg(
        func=interaction_mdp.match_motion_ref_with_scene,
        mode="startup",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
        },
    )
    reset_robot = EventTermCfg(
        func=interaction_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("robot"),
            "position_offset": [0.0, 0.0, 0.0],
            "dof_vel_ratio": 1.0,
            "base_lin_vel_ratio": 1.0,
            "base_ang_vel_ratio": 1.0,
            "randomize_pose_range": {},
            "randomize_velocity_range": {},
            "randomize_joint_pos_range": (0.0, 0.0),
        },
    )
    reset_objects = EventTermCfg(
        func=interaction_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("objects"),
            "object_name": "box",
            "position_offset": [0.0, 0.0, 0.0],
            "base_lin_vel_ratio": 1.0,
            "base_ang_vel_ratio": 1.0,
            "randomize_pose_range": {},
            "randomize_velocity_range": {},
        },
    )
    update_object_reference_vis = None
    reset_object_reference_trail = None
    bin_fail_counter_smoothing = EventTermCfg(
        func=interaction_mdp.beyondmimic_bin_fail_counter_smoothing,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # every environment step
        params={
            "curriculum_name": "beyond_adaptive_sampling",
        },
    )


@configclass
class CurriculumCfg:
    """BeyondMimic curriculum terms for the MDP."""

    beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
        func=interaction_mdp.BeyondMimicAdaptiveWeighting,
    )
    tracking_sigma_annealing = CurriculumTermCfg(  # type: ignore
        func=interaction_mdp.TrackingSigmaCurriculum,
        params={
            "term_names": [
                "object_pos_tracking_gauss",
                "object_rot_tracking_gauss",
            ],
            "initial_sigmas": [0.8, 1.0],
            "final_sigmas": [0.2, 0.35],
            "start_step": 250_000,
            "end_step": 3_000_000,
            "min_sigma": 0.1,
        },
    )


@configclass
class TerminationCfg:
    """BeyondMimic termination terms for the MDP."""

    time_out = DoneTermCfg(func=mdp.time_out, time_out=True)
    base_pos_too_far = DoneTermCfg(
        func=instinct_mdp.pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "distance_threshold": 0.25,
            "check_at_keyframe_threshold": -1,
            "print_reason": False,
            "height_only": True,
        },
    )
    base_pg_too_far = DoneTermCfg(
        func=instinct_mdp.projected_gravity_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "projected_gravity_threshold": 0.8,  # distance on z-axis of projected gravity
            "check_at_keyframe_threshold": -1,
            "z_only": False,  # find out useful if not z_only but beyondmimic default is z_only
            "print_reason": False,
        },
    )
    link_pos_too_far = DoneTermCfg(
        func=instinct_mdp.link_pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg(
                "motion_reference",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                preserve_order=True,
            ),
            "distance_threshold": 0.25,
            "in_base_frame": False,
            "check_at_keyframe_threshold": -1,
            "height_only": True,
            "print_reason": False,
        },
    )

    dataset_exhausted = DoneTermCfg(
        func=instinct_mdp.dataset_exhausted,
        time_out=True,
        params={
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "print_reason": False,
        },
    )
    out_of_border = DoneTermCfg(
        func=instinct_mdp.terrain_out_of_bounds,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
    )


@configclass
class MonitorCfg:

    # joint_torque = SceneEntityCfg("monitor_joint_torque") # NOTE: hurt the performance, so not used.
    # upper_joint_stat = MonitorTermCfg(
    #     func=JointStatMonitorTerm,
    #     params=dict(
    #         asset_cfg=SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_.*",
    #                 ".*_elbow_.*",
    #                 ".*_wrist_.*",
    #             ],
    #         ),
    #     ),
    # )
    # lower_joint_stat = MonitorTermCfg(
    #     func=JointStatMonitorTerm,
    #     params=dict(
    #         asset_cfg=SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "waist_.*",
    #                 ".*_ankle_.*",
    #                 ".*_hip_.*",
    #             ],
    #         ),
    #     ),
    # )
    # body_stat = MonitorTermCfg(
    #     func=BodyStatMonitorTerm,
    #     params=dict(
    #         asset_cfg=SceneEntityCfg(
    #             "robot",
    #             body_names=MISSING,
    #         ),
    #     ),
    # )

    dataset = MonitorTermCfg(
        func=MotionReferenceMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("motion_reference"),
            sample_stat_interval=500,
            top_n_samples=5,
        ),
    )
    shadowing_position = MonitorTermCfg(
        func=ShadowingPositionMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            check_at_keyframe_threshold=0.03,
        ),
    )
    shadowing_rotation = MonitorTermCfg(
        func=ShadowingRotationMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_joint_pos = MonitorTermCfg(
        func=ShadowingJointPosMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_joint_vel = MonitorTermCfg(
        func=ShadowingJointVelMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_link_pos = MonitorTermCfg(
        func=ShadowingLinkPosMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            masking=True,
        ),
    )


@configclass
class InteractionShadowingEnvCfg(InstinctLabRLEnvCfg):
    """Configuration for the shadowing environment."""

    scene: InteractionShadowingSceneCfg = InteractionShadowingSceneCfg(num_envs=4096)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandCfg = CommandCfg()
    rewards: SingleRewardsCfg = SingleRewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationCfg = TerminationCfg()
    monitors: MonitorCfg = MonitorCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 1.0 / 50.0 / self.decimation
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.run_name = "".join([])

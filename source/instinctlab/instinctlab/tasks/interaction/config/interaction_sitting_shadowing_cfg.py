from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.utils import configclass

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.interaction.config.interaction_shadowing_cfg as base_cfg
import instinctlab.tasks.interaction.mdp as interaction_mdp
from instinctlab.monitors import MonitorTermCfg


@configclass
class ObservationsCfg(base_cfg.ObservationsCfg):
    """Observation config for sitting interaction shadowing."""

    @configclass
    class PolicyObsCfg(base_cfg.ObservationsCfg.PolicyObsCfg):
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
    class CriticObsCfg(base_cfg.ObservationsCfg.CriticObsCfg):
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
class RewardGroupCfg(base_cfg.RewardGroupCfg):
    """Reward terms for sitting interaction shadowing."""

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
            "debug_vis": True,
            "debug_vis_max_envs": 1,
            "debug_vis_show_all_points": True,
            "debug_vis_show_nearest": True,
            "debug_vis_point_radius": 0.028,
            "debug_vis_nearest_point_radius": 0.04,
            "debug_vis_arrow_length_scale": 10.0,
            "debug_vis_arrow_thickness_scale": (1.0, 1.0, 1.0),
            "debug_vis_nearest_arrow_thickness_scale": (1.25, 1.25, 1.25),
        },
    )


@configclass
class SingleRewardsCfg(base_cfg.SingleRewardsCfg):
    rewards: RewardGroupCfg = RewardGroupCfg()


@configclass
class CurriculumCfg(base_cfg.CurriculumCfg):
    """Curriculum terms for sitting interaction shadowing."""

    tracking_sigma_annealing = None


@configclass
class TerminationCfg(base_cfg.TerminationCfg):
    """Termination terms for sitting interaction shadowing."""

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
class MonitorCfg(base_cfg.MonitorCfg):
    """Monitor terms for sparse sitting contact debugging."""

    sparse_contact_map = MonitorTermCfg(
        func=interaction_mdp.SparseContactMapMonitorTerm,
        params={
            "reward_group_name": "rewards",
            "reward_term_name": "sparse_contact_map",
        },
    )


@configclass
class InteractionSittingShadowingEnvCfg(base_cfg.InteractionShadowingEnvCfg):
    """Environment config for sitting interaction shadowing."""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: SingleRewardsCfg = SingleRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationCfg = TerminationCfg()
    monitors: MonitorCfg = MonitorCfg()

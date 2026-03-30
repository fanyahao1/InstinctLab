from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.utils import configclass

import instinctlab.tasks.interaction.config.interaction_shadowing_cfg as base_cfg
import instinctlab.tasks.interaction.mdp as interaction_mdp


@configclass
class ObservationsCfg(base_cfg.ObservationsCfg):
    """Observation config for sitting interaction shadowing."""

    @configclass
    class PolicyObsCfg(base_cfg.ObservationsCfg.PolicyObsCfg):
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

    @configclass
    class CriticObsCfg(base_cfg.ObservationsCfg.CriticObsCfg):
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

    object_pos_tracking_gauss = None
    object_rot_tracking_gauss = None
    wrist_object_contact_ref_phase = None
    seat_object_contact_ref_phase = RewTermCfg(
        func=interaction_mdp.object_contact_reference_phase,
        weight=3.0,
        params={
            "sensor_names": [
                "pelvis_object_contact",
                "left_hip_object_contact",
                "right_hip_object_contact",
            ],
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "object_name": "box",
            "threshold": 1.0,
            "normalize": True,
            "print_reason": False,
            "debug_label": "seat_object_contact",
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
class InteractionSittingShadowingEnvCfg(base_cfg.InteractionShadowingEnvCfg):
    """Environment config for sitting interaction shadowing."""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: SingleRewardsCfg = SingleRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationCfg = TerminationCfg()

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from collections.abc import Callable

import torch

from instinctlab.motion_reference.motion_reference_cfg import MotionBufferCfg

from .amass_motion_cfg import AmassMotionCfg
from .object_motion import ObjectMotion


@configclass
class ObjectMotionCfg(AmassMotionCfg):
    """Configuration for object motion data, which includes both human motion and object motion.

    This loader extends AmassMotion to handle datasets containing synchronized recordings of:
    - Human motion (joint poses, body pose, etc.)
    - Object motion (position, rotation, velocity, etc.)

    The object data is used for computing reward signals during training.
    """

    class_type: type = ObjectMotion

    object_data_keys: dict[str, str] = MISSING  # type: ignore
    """Mapping from object names to their data keys in the npz files.

    For example:
        object_data_keys: {
            "ball": "object_pos",
            "cup": "cup_pos",
        }

    The npz file should contain keys like:
        - "{object_name}_pos": object position, shape (N, 3)
        - "{object_name}_quat": object quaternion, shape (N, 4), optional
        - "{object_name}_lin_vel": object linear velocity, shape (N, 3), optional
        - "{object_name}_ang_vel": object angular velocity, shape (N, 3), optional

    If custom key names are used, map them here.
    """

    object_velocity_estimation_method: Literal["frontward", "backward", "frontbackward", None] = "frontward"
    """The method to estimate the velocity of the object motion data.
    - "frontward": use the frontward difference to estimate the velocity.
    - "backward": use the backward difference to estimate the velocity.
    - "frontbackward": use both frontward and backward difference to estimate the velocity.
    - None: do not estimate velocity (assumes velocity is provided in the data).
    """

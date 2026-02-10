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

    metadata_yaml: str | None = None
    """Path to the metadata YAML file that describes motion-object bindings.

    The YAML file should contain:
    - motion_files: list of motion files with their object_id
    - objects: list of object configurations with object_id and matching properties

    Example:
        motion_files:
        - motion_file: small_box_lift.npz
          object_id: 0
        - motion_file: large_box_push.npz
          object_id: 1
        objects:
        - object_id: 0
          size: small
          usd_path: small_box.usd
        - object_id: 1
          size: large
          usd_path: large_box.usd
    """

    # TODO: only support usd_path matching for now
    object_matching_key: str = "usd_path"
    """The property key used to match objects in the scene with object definitions in metadata.

    Common options:
    - "usd_path": Match by USD file path (default, most reliable)
    - "size": Match by object size category (small/medium/large)
    - "type": Match by object type (box/sphere/cylinder)
    - "name": Match by object name

    The matching key should exist in both:
    1. The metadata YAML's object definitions
    2. The scene's object spawn configuration or properties
    """

from __future__ import annotations

import os
import torch
import yaml
from typing import TYPE_CHECKING, Sequence

from instinctlab.motion_reference import MotionReferenceData, MotionSequence
from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion
from instinctlab.motion_reference.utils import estimate_angular_velocity, estimate_velocity

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from .object_motion_cfg import ObjectMotionCfg


class ObjectMotion(AmassMotion):
    """Motion buffer for human and object motion data.

    Extends AmassMotion to load synchronized recordings of human motion and multiple objects.
    Object data (position, orientation, velocities) is extracted from the motion files and
    can be accessed during simulation for reward computation.
    """

    cfg: ObjectMotionCfg

    def __init__(
        self,
        cfg: ObjectMotionCfg,
        *args,
        **kwargs,
    ):
        super().__init__(cfg, *args, **kwargs)

    def enable_trajectories(self, traj_ids: torch.Tensor | slice | None = None) -> None:
        if not traj_ids is None and hasattr(self, "_all_motion_object_ids"):
            self._all_motion_object_ids = self._all_motion_object_ids[traj_ids]
            self._all_motion_selectable_envs_mask = self._all_motion_selectable_envs_mask[traj_ids]
        super().enable_trajectories(traj_ids)

    def _refresh_motion_file_list(self):
        """Refresh the list of motion files based on the object configuration."""
        if self.cfg.metadata_yaml is None:
            super()._refresh_motion_file_list()
            return

        with open(self.cfg.metadata_yaml) as file:
            self.yaml_data = yaml.safe_load(file)

        self._all_motion_files = [os.path.join(self.cfg.path, f["motion_file"]) for f in self.yaml_data["motion_files"]]
        self._motion_weights = torch.tensor(
            [float(f.get("weight", 1.0)) for f in self.yaml_data["motion_files"]],
            dtype=torch.float,
            device=self.buffer_device,
        )

        self._all_motion_object_ids = torch.tensor(
            [int(f["object_id"]) for f in self.yaml_data["motion_files"]],
            dtype=torch.int,
            device=self.buffer_device,
        )

        self._all_motion_selectable_envs_mask = torch.ones(
            len(self._all_motion_files), self.num_assigned_envs, dtype=torch.bool, device=self.buffer_device
        )

    def match_scene(self, scene: InteractiveScene) -> None:
        """Match motion files with scene objects based on object_matching_key.

        This method:
        1. Reads object configurations from the scene
        2. Extracts the matching property (e.g., usd_path) for each environment's object
        3. Builds a mapping from object_id to env_ids
        4. Updates _all_motion_selectable_envs_mask to ensure each motion is only
           assigned to environments with matching objects
        """
        if self.cfg.metadata_yaml is None:
            print("[ObjectMotion] No metadata_yaml provided, skipping scene matching.")
            return

        if not hasattr(self, "yaml_data"):
            with open(self.cfg.metadata_yaml) as file:
                self.yaml_data = yaml.safe_load(file)

        objects_asset = scene.get("objects", None)
        if objects_asset is None:
            print("[ObjectMotion] Warning: No 'objects' found in scene. Skipping scene matching.")
            return

        object_id_to_matching_value = {
            obj["object_id"]: obj[self.cfg.object_matching_key] for obj in self.yaml_data["objects"]
        }

        env_object_properties = self._extract_object_properties_from_scene(objects_asset)

        object_id_to_env_ids = {obj["object_id"]: [] for obj in self.yaml_data["objects"]}

        for env_id, obj_property in enumerate(env_object_properties):
            for object_id, matching_value in object_id_to_matching_value.items():
                if self._match_object_property(obj_property, matching_value):
                    object_id_to_env_ids[object_id].append(env_id)
                    break

        self._all_motion_selectable_envs_mask.fill_(False)

        for object_id, env_ids in object_id_to_env_ids.items():
            num_envs_matched = len(env_ids)
            num_motions_matched = (self._all_motion_object_ids == object_id).sum().item()

            if num_envs_matched == 0:
                print(f"[ObjectMotion] Warning: Object {object_id} has no matching envs. Motions will be disabled.")
                continue

            if num_motions_matched == 0:
                print(f"[ObjectMotion] Warning: Object {object_id} has no matching motions.")
                continue

            print(
                f"[ObjectMotion] Matched object_id {object_id} ({object_id_to_matching_value[object_id]}): "
                f"{num_motions_matched} motions -> {num_envs_matched} envs"
            )

            assigned_env_ids = self.env_ids_to_assigned_ids(torch.tensor(env_ids, device=self.buffer_device)).to(
                self.buffer_device
            )
            self._all_motion_selectable_envs_mask[self._all_motion_object_ids == object_id, assigned_env_ids] = True

    def _extract_object_properties_from_scene(self, objects_asset) -> list[str]:
        """Extract object properties from scene for matching.

        Returns a list where each element is the matching property value for that environment.
        """
        num_envs = objects_asset.num_instances

        if self.cfg.object_matching_key == "usd_path":
            if hasattr(objects_asset.cfg.spawn, "usd_path"):
                if isinstance(objects_asset.cfg.spawn.usd_path, list):
                    print(
                        "[ObjectMotion] Error: Multiple USD paths found. "
                        "Cannot automatically determine per-env mapping. "
                        "Consider using RigidObjectCollectionCfg or custom matching logic."
                    )
                    return [""] * num_envs
                else:
                    return [os.path.basename(objects_asset.cfg.spawn.usd_path)] * num_envs
            else:
                print(f"[ObjectMotion] Warning: object_matching_key='{self.cfg.object_matching_key}' not found.")
                return [""] * num_envs
        else:
            print(
                f"[ObjectMotion] Custom matching key '{self.cfg.object_matching_key}' requires "
                "custom implementation of _extract_object_properties_from_scene."
            )
            return [""] * num_envs

    def _match_object_property(self, env_property: str, target_value: str) -> bool:
        """Check if an environment's object property matches the target value."""
        if self.cfg.object_matching_key == "usd_path":
            return os.path.basename(env_property) == os.path.basename(target_value)
        else:
            return env_property == target_value

    def _sample_assigned_env_starting_stub(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Sample motion starting points, ensuring object matching constraints."""
        if hasattr(self, "_all_motion_selectable_envs_mask"):
            self._safe_motion_resampling_for_objects(env_ids)

        super()._sample_assigned_env_starting_stub(env_ids)

    def _safe_motion_resampling_for_objects(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Ensure sampled motions are compatible with the environment's object."""
        if env_ids is None:
            env_ids = torch.arange(self.num_assigned_envs, device=self.buffer_device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.buffer_device)

        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        motion_ids = self._assigned_env_motion_selection[assigned_ids]

        invalid_mask = ~self._all_motion_selectable_envs_mask[motion_ids, assigned_ids]

        if invalid_mask.any():
            invalid_assigned_ids = assigned_ids[invalid_mask]

            valid_motion_mask_per_env = self._all_motion_selectable_envs_mask[:, invalid_assigned_ids]

            for i, assigned_id in enumerate(invalid_assigned_ids):
                valid_motions = torch.where(valid_motion_mask_per_env[:, i])[0]

                if len(valid_motions) == 0:
                    print(
                        f"[ObjectMotion] Error: No valid motions for env {assigned_id.item()}. "
                        "Check object matching configuration."
                    )
                    continue

                valid_weights = self._motion_weights[valid_motions]
                resampled_motion_id = valid_motions[torch.multinomial(valid_weights, 1, replacement=True).item()].item()

                self._assigned_env_motion_selection[assigned_id] = resampled_motion_id

    def _read_amass_motion_file(self, filepath: str) -> MotionSequence:
        """Read AMASS motion file and extract both human and object motion data."""
        motion_seq = super()._read_amass_motion_file(filepath)
        motion_data = torch.load(filepath)
        self._extract_object_motion_data(motion_seq, motion_data, filepath)
        return motion_seq

    def _read_retargetted_motion_file(self, filepath: str) -> MotionSequence:
        """Read retargeted motion file and extract both human and object motion data."""
        motion_seq = super()._read_retargetted_motion_file(filepath)
        motion_data = torch.load(filepath)
        self._extract_object_motion_data(motion_seq, motion_data, filepath)
        return motion_seq

    def _extract_object_motion_data(self, motion_seq: MotionSequence, motion_data: dict, filepath: str) -> None:
        """Extract object motion data from the motion file and attach to MotionSequence."""
        if not hasattr(motion_seq, "object_data"):
            motion_seq.object_data = {}

        for object_name, data_key_prefix in self.cfg.object_data_keys.items():
            object_info = self._load_object_component(motion_data, data_key_prefix, object_name, motion_seq, filepath)
            motion_seq.object_data[object_name] = object_info

    def _load_object_component(
        self,
        motion_data: dict,
        data_key_prefix: str,
        object_name: str,
        motion_seq: MotionSequence,
        filepath: str,
    ) -> dict[str, torch.Tensor | None]:
        """Load a single object's motion components."""
        pos_key = f"{data_key_prefix}_pos"
        quat_key = f"{data_key_prefix}_quat"
        lin_vel_key = f"{data_key_prefix}_lin_vel"
        ang_vel_key = f"{data_key_prefix}_ang_vel"

        object_info = {}

        object_pos = self._extract_tensor_from_data(motion_data, pos_key, filepath, object_name)
        object_info["pos"] = object_pos

        object_quat = self._extract_optional_tensor(motion_data, quat_key)
        object_info["quat"] = object_quat

        object_lin_vel = self._extract_optional_tensor(motion_data, lin_vel_key)
        if object_lin_vel is None and self.cfg.object_velocity_estimation_method is not None:
            object_lin_vel = self._estimate_object_velocity(
                object_pos, motion_seq.framerate, self.cfg.object_velocity_estimation_method
            )
        object_info["lin_vel"] = object_lin_vel

        object_ang_vel = self._extract_optional_tensor(motion_data, ang_vel_key)
        if (
            object_ang_vel is None
            and object_quat is not None
            and self.cfg.object_velocity_estimation_method is not None
        ):
            object_ang_vel = self._estimate_object_angular_velocity(
                object_quat, motion_seq.framerate, self.cfg.object_velocity_estimation_method
            )
        object_info["ang_vel"] = object_ang_vel

        return object_info

    def _extract_tensor_from_data(self, motion_data: dict, key: str, filepath: str, object_name: str) -> torch.Tensor:
        """Extract tensor from motion data, raising error if missing."""
        if key not in motion_data:
            raise ValueError(
                f"Object data key '{key}' not found for '{object_name}' in {filepath}. "
                f"Available keys: {list(motion_data.keys())}"
            )
        tensor = torch.from_numpy(motion_data[key]).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _extract_optional_tensor(self, motion_data: dict, key: str) -> torch.Tensor | None:
        """Extract optional tensor from motion data."""
        if key not in motion_data:
            return None
        tensor = torch.from_numpy(motion_data[key]).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _estimate_object_velocity(
        self, object_pos: torch.Tensor, framerate: torch.Tensor | float, method: str
    ) -> torch.Tensor:
        """Estimate object linear velocity from position."""
        if isinstance(framerate, torch.Tensor):
            framerate = framerate.item() if framerate.numel() == 1 else framerate
        lin_vel, _ = estimate_velocity(object_pos, framerate, method=method)
        return lin_vel

    def _estimate_object_angular_velocity(
        self, object_quat: torch.Tensor, framerate: torch.Tensor | float, method: str
    ) -> torch.Tensor:
        """Estimate object angular velocity from quaternion."""
        if isinstance(framerate, torch.Tensor):
            framerate = framerate.item() if framerate.numel() == 1 else framerate
        _, ang_vel = estimate_angular_velocity(object_quat, framerate, method=method)
        return ang_vel

    def fill_motion_data(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: MotionReferenceData,
    ) -> None:
        """Fill motion data including object data to the buffer."""
        super().fill_motion_data(env_ids, sample_timestamp, env_origins, data_buffer)

        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        frame_selections = torch.round(
            (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + sample_timestamp.to(self.buffer_device))
            * self._all_motion_sequences.framerate[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
        )

        frame_selections = torch.where(
            data_buffer.validity[env_ids].to(self.buffer_device),
            frame_selections,
            self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
            - 1,
        ).to(torch.long)

        assigned_ids_across_frame = assigned_ids.unsqueeze(-1).repeat(1, frame_selections.shape[1])
        motion_ids_across_frame = self._assigned_env_motion_selection[assigned_ids_across_frame]

        self._fill_object_motion_data(env_ids, motion_ids_across_frame, frame_selections, data_buffer)

    def _fill_object_motion_data(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        motion_ids_across_frame: torch.Tensor,
        frame_selections: torch.Tensor,
        data_buffer: MotionReferenceData,
    ) -> None:
        """Fill object motion data into the MotionReferenceData buffer."""
        if not hasattr(data_buffer, "object_data"):
            data_buffer.object_data = {}

        num_envs = len(env_ids) if isinstance(env_ids, (list, torch.Tensor)) else env_ids.stop - env_ids.start
        num_frames = frame_selections.shape[1]

        for object_name in self.cfg.object_data_keys.keys():
            if not self._has_object_data(object_name):
                continue

            if object_name not in data_buffer.object_data:
                self._initialize_object_data_buffer(data_buffer, object_name, num_envs, num_frames)

            self._fill_object_data_fields(data_buffer, object_name, env_ids, motion_ids_across_frame, frame_selections)

    def _has_object_data(self, object_name: str) -> bool:
        """Check if object data exists in motion sequences."""
        return (
            hasattr(self._all_motion_sequences, "object_data")
            and len(self._all_motion_sequences.object_data) > 0
            and object_name in self._all_motion_sequences.object_data
        )

    def _initialize_object_data_buffer(
        self, data_buffer: MotionReferenceData, object_name: str, num_envs: int, num_frames: int
    ) -> None:
        """Initialize object data fields in the buffer."""
        object_info = self._all_motion_sequences.object_data[object_name]

        data_buffer.object_data[object_name] = {
            "pos": torch.zeros(num_envs, num_frames, 3, device=self.output_device),
        }
        if object_info.get("quat") is not None:
            data_buffer.object_data[object_name]["quat"] = torch.zeros(
                num_envs, num_frames, 4, device=self.output_device
            )
        if object_info.get("lin_vel") is not None:
            data_buffer.object_data[object_name]["lin_vel"] = torch.zeros(
                num_envs, num_frames, 3, device=self.output_device
            )
        if object_info.get("ang_vel") is not None:
            data_buffer.object_data[object_name]["ang_vel"] = torch.zeros(
                num_envs, num_frames, 3, device=self.output_device
            )

    def _fill_object_data_fields(
        self,
        data_buffer: MotionReferenceData,
        object_name: str,
        env_ids: Sequence[int] | torch.Tensor,
        motion_ids_across_frame: torch.Tensor,
        frame_selections: torch.Tensor,
    ) -> None:
        """Fill individual object data fields."""
        object_info = self._all_motion_sequences.object_data[object_name]
        motion_ids_flat = motion_ids_across_frame.flatten()
        frame_selections_flat = frame_selections.flatten()

        if object_info.get("pos") is not None:
            object_pos = object_info["pos"][motion_ids_flat, frame_selections_flat]
            data_buffer.object_data[object_name]["pos"][env_ids] = object_pos.to(self.output_device).reshape(
                len(env_ids), -1, 3
            )

        if "quat" in data_buffer.object_data[object_name] and object_info.get("quat") is not None:
            object_quat = object_info["quat"][motion_ids_flat, frame_selections_flat]
            data_buffer.object_data[object_name]["quat"][env_ids] = object_quat.to(self.output_device).reshape(
                len(env_ids), -1, 4
            )

        if "lin_vel" in data_buffer.object_data[object_name] and object_info.get("lin_vel") is not None:
            object_lin_vel = object_info["lin_vel"][motion_ids_flat, frame_selections_flat]
            data_buffer.object_data[object_name]["lin_vel"][env_ids] = object_lin_vel.to(self.output_device).reshape(
                len(env_ids), -1, 3
            )

        if "ang_vel" in data_buffer.object_data[object_name] and object_info.get("ang_vel") is not None:
            object_ang_vel = object_info["ang_vel"][motion_ids_flat, frame_selections_flat]
            data_buffer.object_data[object_name]["ang_vel"][env_ids] = object_ang_vel.to(self.output_device).reshape(
                len(env_ids), -1, 3
            )

from __future__ import annotations

import numpy as np
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

    def _load_motion_sequences(self):
        """Load motion sequences and store object data from individual motions."""
        # Store the original motion sequences before they're processed by parent
        # We need to load them first to get the object_data
        print(f"[ObjectMotion] Loading motion files to extract object data...")
        all_motion_sequences = list(map(self._read_motion_file, range(len(self._all_motion_files))))
        print(f"[ObjectMotion] All {len(all_motion_sequences)} motion files loaded with object data.")

        # Extract object_data from each motion sequence before they're combined
        self._all_motion_object_data = {}
        for i, motion in enumerate(all_motion_sequences):
            if hasattr(motion, "object_data") and motion.object_data:
                self._all_motion_object_data[i] = motion.object_data

        # Call parent to load the combined motion sequences (without object_data)
        # We need to temporarily modify the method to not lose our object_data
        # Actually, let's call the grandparent's logic directly
        self._load_motion_sequences_impl(all_motion_sequences)

    def _load_motion_sequences_impl(self, all_motion_sequences):
        """Internal implementation to load motion sequences (copied from parent)."""
        import numpy as np

        # This is copied from AmassMotion._load_motion_sequences
        print(
            "[ObjectMotion] buffer lengths statistics:"
            f" mean: {np.array([motion.buffer_length for motion in all_motion_sequences]).mean()},"
            f" max: {np.array([motion.buffer_length for motion in all_motion_sequences]).max()},"
            f" min: {np.array([motion.buffer_length for motion in all_motion_sequences]).min()},"
        )
        self._all_motion_sequences = MotionSequence.make_emtpy_concat_batch(
            buffer_lengths=[int(motion.buffer_length) for motion in all_motion_sequences],
            num_joints=self.articulation_view.max_dofs,
            num_links=self.num_link_to_ref,
            device=self.buffer_device,
        )
        for i, motion in enumerate(all_motion_sequences):
            for attr in self._all_motion_sequences.attrs_with_frame_dim:
                getattr(self._all_motion_sequences, attr)[i, : motion.buffer_length] = getattr(motion, attr)
            for attr in self._all_motion_sequences.attrs_only_batch_dim:
                getattr(self._all_motion_sequences, attr)[i] = getattr(motion, attr)

        # Now add object_data to _all_motion_sequences
        self._copy_object_data_to_all_motion_sequences()

    def _copy_object_data_to_all_motion_sequences(self):
        """Copy object data from individual motion sequences to the combined buffer.

        The object data is stored as a 3-D tensor of shape
        [num_motions, max_object_frames, data_dim] for [motion_id, frame_id] indexing.
        Each motion's object frames may differ from its human-motion frames, so the actual
        tensor length (not the human-motion buffer_length) is used for padding.
        """
        if not hasattr(self, "_all_motion_object_data"):
            return

        num_motions = len(self._all_motion_sequences.buffer_length)

        # Reorganise data as {obj_name: {key: [tensor_or_None, ...]}} with one entry per motion.
        object_data_by_object: dict[str, dict[str, list]] = {}

        for motion_idx in range(num_motions):
            obj_data = self._all_motion_object_data.get(motion_idx, {})
            for obj_name, obj_fields in obj_data.items():
                if obj_name not in object_data_by_object:
                    object_data_by_object[obj_name] = {}
                for key, value in obj_fields.items():
                    if value is not None:
                        if key not in object_data_by_object[obj_name]:
                            object_data_by_object[obj_name][key] = [None] * num_motions
                        object_data_by_object[obj_name][key][motion_idx] = value.to(self.buffer_device)

        if object_data_by_object:
            self._all_motion_sequences.object_data = {}
            for obj_name, obj_fields in object_data_by_object.items():
                self._all_motion_sequences.object_data[obj_name] = {}
                for key, tensor_list in obj_fields.items():
                    self._all_motion_sequences.object_data[obj_name][key] = self._create_concat_batch_tensor(
                        tensor_list, self.buffer_device
                    )

    def _create_concat_batch_tensor(self, tensor_list: list[torch.Tensor | None], device):
        """Create a padded tensor of shape [num_motions, max_object_frames, data_dim].

        Each entry in tensor_list corresponds to one motion (None if that motion has no
        object data for this key).  Uses the actual object-tensor length per motion rather
        than the human-motion buffer length, so mismatched frame counts are handled safely.
        """
        valid_tensors = [t for t in tensor_list if t is not None]
        if not valid_tensors:
            return torch.zeros(len(tensor_list), 0, 1, device=device)

        data_dim = valid_tensors[0].shape[-1] if valid_tensors[0].ndim > 1 else 1
        max_len = max(t.shape[0] for t in valid_tensors)

        result = torch.zeros(len(tensor_list), max_len, data_dim, device=device)
        for i, tensor in enumerate(tensor_list):
            if tensor is not None:
                n_frames = tensor.shape[0]
                result[i, :n_frames] = tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor
        return result

    def enable_trajectories(self, traj_ids: torch.Tensor | slice | None = None) -> None:
        if not traj_ids is None and hasattr(self, "_all_motion_object_ids"):
            self._all_motion_object_ids = self._all_motion_object_ids[traj_ids]
            # _all_motion_selectable_envs_mask may not exist yet (created in set_env_ids_assignments)
            if hasattr(self, "_all_motion_selectable_envs_mask"):
                self._all_motion_selectable_envs_mask = self._all_motion_selectable_envs_mask[traj_ids]
        super().enable_trajectories(traj_ids)

    def set_env_ids_assignments(self, env_ids: slice) -> None:
        """Set the environment IDs assignments for the motion buffer.
        Also initializes the selectable envs mask and env-object fixed mapping after the slice is set.
        """
        super().set_env_ids_assignments(env_ids)

        # Initialize the selectable envs mask now that assigned_env_slice is set
        if hasattr(self, "_all_motion_files") and self.cfg.metadata_yaml is not None:
            self._all_motion_selectable_envs_mask = torch.ones(
                len(self._all_motion_files), self.num_assigned_envs, dtype=torch.bool, device=self.buffer_device
            )
            # DEBUG: Print mask initialization
            print(f"[ObjectMotion] ===== SET_ENV_IDS_ASSIGNMENTS =====")
            print(f"[ObjectMotion] assigned_env_slice: {self.assigned_env_slice}")
            print(f"[ObjectMotion] num_assigned_envs: {self.num_assigned_envs}")
            print(
                f"[ObjectMotion] _all_motion_selectable_envs_mask shape: {self._all_motion_selectable_envs_mask.shape}"
            )
            # If match_scene was called before the mask existed, process it now.
            if hasattr(self, "_pending_match_scene"):
                pending_scene = self._pending_match_scene
                del self._pending_match_scene
                self.match_scene(pending_scene)

            # Create env-object fixed mapping for efficient reset (no resample needed!)
            # This will be called either here (if _env_usd_paths already exists from pending match_scene)
            # or at the end of match_scene (when called as an event)
            if hasattr(self, "_env_usd_paths") and hasattr(self, "_assigned_env_motion_selection"):
                self._create_env_object_fixed_mapping()

            print(f"[ObjectMotion] ===== SET_ENV_IDS_ASSIGNMENTS END =====")

    def _create_env_object_fixed_mapping(self):
        """Create env-object fixed mapping for efficient reset.

        This method pre-computes:
        1. _assigned_env_object_ids: tensor of object_id for each assigned_env
        2. _assigned_env_motion_pools: dict {object_id: tensor of motion indices}

        After this, reset will sample directly from the correct motion pool
        without needing resample check.
        """
        if not hasattr(self, "_env_usd_paths"):
            print("[ObjectMotion] WARNING: _env_usd_paths not available, skipping fixed mapping")
            return

        env_usd = self._env_usd_paths  # list of basename, e.g., ['chair.usd', 'sofa.usd', ...]

        # Parse object_id from yaml: {'chair.usd': 0, 'sofa.usd': 1}
        obj_id_map = {}
        for obj in self.yaml_data.get("objects", []):
            basename = os.path.basename(obj["usd_path"])  # 'chair.usd', 'sofa.usd'
            obj_id_map[basename] = obj["object_id"]

        # Create object_id for each assigned_env
        env_object_ids = [obj_id_map.get(usd, 0) for usd in env_usd]
        self._assigned_env_object_ids = torch.tensor(env_object_ids, dtype=torch.int, device=self.buffer_device)

        # Create motion pools for each object_id
        self._assigned_env_motion_pools = {}  # {object_id: tensor of motion indices}
        unique_obj_ids = torch.unique(self._assigned_env_object_ids)
        for obj_id in unique_obj_ids:
            mask_for_obj = self._all_motion_object_ids == obj_id
            self._assigned_env_motion_pools[int(obj_id)] = torch.where(mask_for_obj)[0]

        # DEBUG: Print mapping info
        print(f"[ObjectMotion] ===== ENV-OBJECT FIXED MAPPING CREATED =====")
        print(f"  total assigned_envs: {len(env_usd)}")
        print(f"  object_id distribution: {torch.bincount(self._assigned_env_object_ids).tolist()}")
        for obj_id, pool in self._assigned_env_motion_pools.items():
            print(f"  object_id={obj_id}: {len(pool)} motions (indices: {pool[:5].tolist()}...)")

        # Verify: all selected motions should match env's object_id
        if hasattr(self, "_assigned_env_motion_selection"):
            selected_obj_ids = self._all_motion_object_ids[self._assigned_env_motion_selection]
            mismatches = (selected_obj_ids != self._assigned_env_object_ids).sum().item()
            print(f"  initial selection mismatches: {mismatches} / {len(self._assigned_env_object_ids)}")

        print(f"[ObjectMotion] ===== FIXED MAPPING END =====")

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

        # Note: _init_motion_bin_weights() is NOT called here.
        # It will be called in enable_trajectories() after _load_motion_sequences() is done.
        # This is because _init_motion_bin_weights() requires _all_motion_sequences which
        # is created in _load_motion_sequences().

        # Note: _all_motion_selectable_envs_mask is initialized in set_env_ids_assignments
        # after assigned_env_slice is set, since it requires num_assigned_envs

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

        if "objects" not in scene.keys():
            print("[ObjectMotion] Warning: No 'objects' found in scene. Skipping scene matching.")
            return

        obj_id_to_val = {obj["object_id"]: obj[self.cfg.object_matching_key] for obj in self.yaml_data["objects"]}
        env_properties = self._extract_object_properties_from_scene(scene["objects"])

        # DEBUG: Print matching info
        print(f"[ObjectMotion] ========== MATCH_SCENE DEBUG ==========")
        print(f"[ObjectMotion] object_matching_key: {self.cfg.object_matching_key}")
        print(f"[ObjectMotion] yaml obj_id -> usd_path mapping: {obj_id_to_val}")
        print(
            f"[ObjectMotion] env_properties (per env USD path): {env_properties[:10]}... (total"
            f" {len(env_properties)} envs)"
        )

        # Store env_properties for later use in reset debug
        self._env_usd_paths = env_properties

        if not hasattr(self, "_all_motion_selectable_envs_mask"):
            print(
                "[ObjectMotion] WARNING: _all_motion_selectable_envs_mask not initialized yet! Saving pending"
                " match_scene."
            )
            self._pending_match_scene = scene
            return

        self._all_motion_selectable_envs_mask.fill_(False)

        for obj_id, target_val in obj_id_to_val.items():
            matched_envs = [i for i, prop in enumerate(env_properties) if self._match_object_property(prop, target_val)]
            print(f"[ObjectMotion] object_id={obj_id} (target: {target_val}) -> matched_envs: {matched_envs[:20]}...")
            if not matched_envs or (self._all_motion_object_ids == obj_id).sum().item() == 0:
                print(f"[ObjectMotion]   -> No matching envs or no motions for this object_id!")
                continue

            env_tensor = self.env_ids_to_assigned_ids(torch.tensor(matched_envs, device=self.buffer_device))
            motion_indices = torch.where(self._all_motion_object_ids == obj_id)[0]
            print(
                f"[ObjectMotion]   -> motions with object_id={obj_id}: indices={motion_indices[:10]}...,"
                f" total={len(motion_indices)}"
            )
            self._all_motion_selectable_envs_mask[motion_indices[:, None], env_tensor[None, :]] = True

        # DEBUG: Print mask summary
        valid_per_env = self._all_motion_selectable_envs_mask.sum(dim=0)
        print(f"[ObjectMotion] Valid motions per env (first 20): {valid_per_env[:20].tolist()}")
        print(f"[ObjectMotion] ========== MATCH_SCENE END ==========")

        # Create env-object fixed mapping after match_scene is done
        # This is called here because _env_usd_paths is set in this method
        if hasattr(self, "_all_motion_files") and self.cfg.metadata_yaml is not None:
            if hasattr(self, "_assigned_env_motion_selection"):  # Only create after buffer is initialized
                self._create_env_object_fixed_mapping()

    def _extract_object_properties_from_scene(self, objects_asset) -> list[str]:
        """Extract object properties from scene for matching.

        Returns a list where each element is the matching property value for that environment.
        """
        num_envs = objects_asset.num_instances

        if self.cfg.object_matching_key == "usd_path":
            if hasattr(objects_asset.cfg.spawn, "usd_path"):
                if isinstance(objects_asset.cfg.spawn.usd_path, list):
                    # MultiUsdFileCfg: different envs may use different USDs. Query the actual prim references to determine per-env USD paths.
                    return self._get_per_env_usd_paths_from_prims(objects_asset, num_envs)
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

    def _get_per_env_usd_paths_from_prims(self, objects_asset, num_envs: int) -> list[str]:
        """Extract per-env USD basename by inspecting each spawned prim's reference arcs.

        Handles MultiUsdFileCfg where each environment may reference a different USD file.
        Searches from the asset-root prim (constructed from cfg.prim_path template) and
        walks down children to find a USD reference arc.
        """
        try:
            import re

            import omni.usd

            stage = omni.usd.get_context().get_stage()

            # cfg.prim_path is already the expanded absolute regex pattern, e.g.:
            #   "/World/envs/env_.*/Object"
            # Replace "env_.*" with "env_{i}" to get the concrete per-env prim path.
            cfg_prim_path: str = objects_asset.cfg.prim_path
            prim_paths = [re.sub(r"env_\.\*", f"env_{i}", cfg_prim_path) for i in range(num_envs)]
            print(f"[ObjectMotion] Probing per-env prim paths, e.g. '{prim_paths[0]}'")

            result: list[str] = []
            for prim_path in prim_paths:
                found_path = self._find_usd_reference_in_prim_tree(stage, prim_path)
                result.append(found_path)

            found_count = sum(1 for p in result if p)
            missing_count = num_envs - found_count

            if missing_count > 0:
                # When replicate_physics=True, only env_0 has real prim specs; other envs
                # are instanced and will return empty. Propagate env_0's result to all.
                fallback = result[0] if result else ""
                if fallback:
                    print(
                        f"[ObjectMotion] {missing_count}/{num_envs} envs could not resolve USD path. "
                        f"Using env_0 USD '{fallback}' for all missing envs (replicate_physics mode)."
                    )
                    result = [p if p else fallback for p in result]
                else:
                    print(
                        "[ObjectMotion] Could not resolve any per-env USD path from the USD stage. "
                        "Check that object prims exist at the expected paths."
                    )
                    prim0 = stage.GetPrimAtPath(prim_paths[0]) if prim_paths else None
                    if prim0 and prim0.IsValid():
                        print(
                            f"[ObjectMotion] env_0 prim '{prim_paths[0]}': "
                            f"type={prim0.GetTypeName()}, "
                            f"children={[c.GetPath().pathString for c in prim0.GetChildren()]}"
                        )

            print(f"[ObjectMotion] Per-env USD paths resolved: {set(result)}")
            return result

        except Exception as e:
            import traceback

            print(f"[ObjectMotion] Error querying per-env USD paths from prims: {e}")
            print(traceback.format_exc())
            return [""] * num_envs

    def _find_usd_reference_in_prim_tree(self, stage, prim_path: str, max_depth: int = 3) -> str:
        """Search prim and its descendants (BFS) for a USD reference arc.

        Returns the basename of the first .usd asset reference found, or empty string.
        """
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return ""

        # BFS over prim and its children up to max_depth
        queue = [(prim, 0)]
        while queue:
            current_prim, depth = queue.pop(0)
            for prim_spec in current_prim.GetPrimStack():
                all_refs = list(prim_spec.referenceList.prependedItems) + list(prim_spec.referenceList.appendedItems)
                for ref in all_refs:
                    asset_path = ref.assetPath
                    if asset_path and ".usd" in asset_path.lower():
                        return os.path.basename(asset_path)
            if depth < max_depth:
                for child in current_prim.GetChildren():
                    queue.append((child, depth + 1))
        return ""

    def _match_object_property(self, env_property: str, target_value: str) -> bool:
        """Check if an environment's object property matches the target value."""
        if self.cfg.object_matching_key == "usd_path":
            return os.path.basename(env_property) == os.path.basename(target_value)
        else:
            return env_property == target_value

    def _sample_assigned_env_starting_stub(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Sample motion using pre-computed object-specific pools (no resample needed!)"""
        if env_ids is None:
            env_ids = torch.arange(self.num_assigned_envs, device=self.buffer_device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.buffer_device)

        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        # Build necessary stubs if not exist
        if not hasattr(self, "_assigned_env_motion_selection"):
            self._assigned_env_motion_selection = torch.zeros(
                self.num_assigned_envs,
                dtype=torch.long,
                device=self.buffer_device,
            )
        if not hasattr(self, "_motion_buffer_start_time_s"):
            self._motion_buffer_start_time_s = torch.zeros(
                self.num_assigned_envs,
                dtype=torch.float,
                device=self.buffer_device,
            )

        # Use fixed pool sampling if available (env-object mapping was created)
        if hasattr(self, "_assigned_env_object_ids") and hasattr(self, "_assigned_env_motion_pools"):
            self._sample_from_fixed_pools(assigned_ids)
        else:
            # Fallback to parent implementation (will need resample in _safe_motion_resampling_for_objects)
            # This happens when metadata_yaml is not set or _env_usd_paths is not available
            self._sample_assigned_env_motion_selection_fallback(assigned_ids)

        # Sample start time (same as parent)
        self._sample_env_motion_start_time(assigned_ids)

    def _sample_from_fixed_pools(self, assigned_ids: torch.Tensor):
        """Sample motion from pre-computed object-specific pools (no resample needed!)."""
        # Get object_id for each assigned_env
        env_obj_ids = self._assigned_env_object_ids[assigned_ids]

        # Sample from each env's object-specific motion pool
        for i, assigned_id in enumerate(assigned_ids.tolist()):
            obj_id = env_obj_ids[i].item()
            motion_pool = self._assigned_env_motion_pools[obj_id]
            weights = self._motion_weights[motion_pool]
            selected_idx = torch.multinomial(weights, 1, replacement=True).item()
            self._assigned_env_motion_selection[assigned_id] = motion_pool[selected_idx].item()

    def _sample_assigned_env_motion_selection_fallback(self, assigned_ids: torch.Tensor):
        """Fallback to parent's random sampling (will need resample check)."""
        self._assigned_env_motion_selection[assigned_ids] = torch.multinomial(
            self._motion_weights,
            len(assigned_ids),
            replacement=True,
        ).to(self.buffer_device)

        # For fallback case, we still need to check and fix invalid selections
        # Convert assigned_ids back to env_ids for resampling check
        if hasattr(self, "_all_motion_selectable_envs_mask") and hasattr(self, "_env_usd_paths"):
            env_ids = assigned_ids.clone()  # In fallback, assigned_ids == env_ids (0-indexed in buffer)
            self._safe_motion_resampling_for_objects(env_ids)

    def _safe_motion_resampling_for_objects(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Ensure sampled motions are compatible with the environment's object."""
        if env_ids is None:
            env_ids = torch.arange(self.num_assigned_envs, device=self.buffer_device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.buffer_device)

        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        motion_ids = self._assigned_env_motion_selection[assigned_ids]

        invalid_mask = ~self._all_motion_selectable_envs_mask[motion_ids, assigned_ids]

        env_usd_paths = getattr(self, "_env_usd_paths", None)

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

    def get_current_object_matching_values(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        object_matching_key: str | None = None,
    ) -> list[str]:
        """Get the current per-env object matching values, such as USD basenames."""
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device) if env_ids is not None else None
        if assigned_ids is None:
            assigned_ids = torch.arange(self.num_assigned_envs, device=self.buffer_device)

        matching_key = self.cfg.object_matching_key if object_matching_key is None else object_matching_key
        object_id_to_value: dict[int, str] = {}
        for obj in getattr(self, "yaml_data", {}).get("objects", []):
            raw_value = str(obj.get(matching_key, ""))
            object_id_to_value[int(obj["object_id"])] = (
                os.path.basename(raw_value) if matching_key == "usd_path" else raw_value
            )

        if hasattr(self, "_assigned_env_object_ids"):
            object_ids = self._assigned_env_object_ids[assigned_ids].tolist()
            return [object_id_to_value.get(int(object_id), "") for object_id in object_ids]

        if matching_key == "usd_path" and hasattr(self, "_env_usd_paths"):
            return [os.path.basename(self._env_usd_paths[idx]) for idx in assigned_ids.tolist()]

        if len(object_id_to_value) == 1:
            return [next(iter(object_id_to_value.values()))] * len(assigned_ids)
        return [""] * len(assigned_ids)

    def get_current_object_keys(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> list[str]:
        """Get the current per-env object type labels."""
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device) if env_ids is not None else None
        if assigned_ids is None:
            assigned_ids = torch.arange(self.num_assigned_envs, device=self.buffer_device)

        object_id_to_key: dict[int, str] = {}
        for obj in getattr(self, "yaml_data", {}).get("objects", []):
            object_key = str(obj.get("type", obj.get("object_key", "")))
            if not object_key and "usd_path" in obj:
                object_key = os.path.splitext(os.path.basename(str(obj["usd_path"])))[0]
            object_id_to_key[int(obj["object_id"])] = object_key

        if hasattr(self, "_assigned_env_object_ids"):
            object_ids = self._assigned_env_object_ids[assigned_ids].tolist()
            return [object_id_to_key.get(int(object_id), "") for object_id in object_ids]

        matching_values = self.get_current_object_matching_values(env_ids=env_ids, object_matching_key="usd_path")
        return [os.path.splitext(os.path.basename(value))[0] if value else "" for value in matching_values]

    def _read_amass_motion_file(self, filepath: str) -> MotionSequence:
        """Read AMASS motion file and extract both human and object motion data."""
        motion_seq = super()._read_amass_motion_file(filepath)
        motion_data = np.load(filepath, allow_pickle=True)
        self._extract_object_motion_data(motion_seq, motion_data, filepath)
        return motion_seq

    def _read_retargetted_motion_file(self, filepath: str) -> MotionSequence:
        """Read retargeted motion file and extract both human and object motion data."""
        motion_seq = super()._read_retargetted_motion_file(filepath)
        motion_data = np.load(filepath, allow_pickle=True)
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

        object_contact = self._extract_optional_tensor_with_candidates(
            motion_data, [f"{data_key_prefix}_contact", "object_contact"]
        )
        object_info["contact"] = object_contact

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
            tensor = tensor.unsqueeze(-1)
        return tensor

    def _extract_optional_tensor_with_candidates(self, motion_data: dict, keys: list[str]) -> torch.Tensor | None:
        """Extract the first optional tensor found in the candidate keys."""
        for key in keys:
            tensor = self._extract_optional_tensor(motion_data, key)
            if tensor is not None:
                return tensor
        return None

    def _estimate_object_velocity(
        self, object_pos: torch.Tensor, framerate: torch.Tensor | float, estimation_type: str
    ) -> torch.Tensor:
        """Estimate object linear velocity from position."""
        if isinstance(framerate, torch.Tensor):
            framerate = framerate.item() if framerate.numel() == 1 else framerate
        # Add batch dimension if not present (shape: (num_frames, 3) -> (1, num_frames, 3))
        if object_pos.ndim == 2:
            object_pos = object_pos.unsqueeze(0)
        lin_vel = estimate_velocity(object_pos, 1.0 / framerate, estimation_type=estimation_type)
        return lin_vel.squeeze(0)  # Remove batch dimension

    def _estimate_object_angular_velocity(
        self, object_quat: torch.Tensor, framerate: torch.Tensor | float, estimation_type: str
    ) -> torch.Tensor:
        """Estimate object angular velocity from quaternion."""
        if isinstance(framerate, torch.Tensor):
            framerate = framerate.item() if framerate.numel() == 1 else framerate
        # Add batch dimension if not present (shape: (num_frames, 4) -> (1, num_frames, 4))
        if object_quat.ndim == 2:
            object_quat = object_quat.unsqueeze(0)
        ang_vel = estimate_angular_velocity(object_quat, 1.0 / framerate, estimation_type=estimation_type)
        return ang_vel.squeeze(0)  # Remove batch dimension

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

        self._fill_object_motion_data(env_ids, motion_ids_across_frame, frame_selections, data_buffer, env_origins)

    def _fill_object_motion_data(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        motion_ids_across_frame: torch.Tensor,
        frame_selections: torch.Tensor,
        data_buffer: MotionReferenceData,
        env_origins: torch.Tensor,
    ) -> None:
        """Fill object motion data into the MotionReferenceData buffer."""
        if not hasattr(data_buffer, "object_data"):
            data_buffer.object_data = {}

        num_envs = len(env_ids) if isinstance(env_ids, (list, torch.Tensor)) else env_ids.stop - env_ids.start
        num_frames = frame_selections.shape[1]

        motion_ids_across_frame = motion_ids_across_frame.to(self.buffer_device)
        frame_selections = frame_selections.to(self.buffer_device)
        buffer_lengths = self._all_motion_sequences.buffer_length[motion_ids_across_frame]

        # [Performance optimization] Compute environment origin offset once and place directly on output_device
        env_origin = self._get_motion_based_origin(env_origins, env_ids).to(self.output_device).unsqueeze(1)

        for object_name in self.cfg.object_data_keys.keys():
            if not self._has_object_data(object_name):
                continue

            if object_name not in data_buffer.object_data:
                self._initialize_object_data_buffer(data_buffer, object_name, num_envs, num_frames)

            self._fill_object_data_fields(
                data_buffer, object_name, env_ids, motion_ids_across_frame, frame_selections, buffer_lengths, env_origin
            )

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
        if object_info.get("contact") is not None:
            data_buffer.object_data[object_name]["contact"] = torch.zeros(num_envs, num_frames, device=self.output_device)

    def _fill_object_data_fields(
        self,
        data_buffer: MotionReferenceData,
        object_name: str,
        env_ids: Sequence[int] | torch.Tensor,
        motion_ids_across_frame: torch.Tensor,
        frame_selections: torch.Tensor,
        buffer_lengths: torch.Tensor,
        env_origin: torch.Tensor,
    ) -> None:
        """Fill individual object data fields."""
        object_info = self._all_motion_sequences.object_data[object_name]
        max_frames = object_info["pos"].shape[1]

        valid_buffer_lengths = buffer_lengths.clamp(max=max_frames - 1)
        frame_selections_final = torch.where(
            frame_selections < valid_buffer_lengths,
            frame_selections,
            valid_buffer_lengths,
        )

        if object_info.get("pos") is not None:
            object_pos = object_info["pos"][motion_ids_across_frame, frame_selections_final].to(self.output_device)
            data_buffer.object_data[object_name]["pos"][env_ids] = object_pos + env_origin

        if "quat" in data_buffer.object_data[object_name] and object_info.get("quat") is not None:
            data_buffer.object_data[object_name]["quat"][env_ids] = object_info["quat"][
                motion_ids_across_frame, frame_selections_final
            ].to(self.output_device)

        if "lin_vel" in data_buffer.object_data[object_name] and object_info.get("lin_vel") is not None:
            data_buffer.object_data[object_name]["lin_vel"][env_ids] = object_info["lin_vel"][
                motion_ids_across_frame, frame_selections_final
            ].to(self.output_device)

        if "ang_vel" in data_buffer.object_data[object_name] and object_info.get("ang_vel") is not None:
            data_buffer.object_data[object_name]["ang_vel"][env_ids] = object_info["ang_vel"][
                motion_ids_across_frame, frame_selections_final
            ].to(self.output_device)

        if "contact" in data_buffer.object_data[object_name] and object_info.get("contact") is not None:
            data_buffer.object_data[object_name]["contact"][env_ids] = object_info["contact"][
                motion_ids_across_frame, frame_selections_final
            ].to(self.output_device).squeeze(-1)

""" Code snippet from accessing prims in isaac sim (scene) """

import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import isaaclab.sim as sim_utils


def get_articulation_view(
    prim_path: str,
    physics_sim_view: physx.SimulationView,
) -> physx.ArticulationView:
    """create simulation view and the prim view (partially copied from `ray_caster.py`)"""
    found_supported_prim_class = False
    prim = sim_utils.find_first_matching_prim(prim_path)
    if prim is None:
        raise RuntimeError(f"Failed to find a prim at path expression: {prim_path}")

    matched_prim_path = prim.GetPath().pathString

    def _regex_path_from_resolved_path(resolved_path: str) -> str:
        if resolved_path == matched_prim_path:
            return prim_path

        if resolved_path.startswith(matched_prim_path):
            descendant_suffix = resolved_path[len(matched_prim_path) :]
            return prim_path + descendant_suffix

        if matched_prim_path.startswith(resolved_path):
            ancestor_suffix = matched_prim_path[len(resolved_path) :]
            if ancestor_suffix and prim_path.endswith(ancestor_suffix):
                return prim_path[: -len(ancestor_suffix)]

        return resolved_path

    articulation_prim_path = prim_path

    # create view based on the type of prim
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        found_supported_prim_class = True
    else:
        # Some imported robots put the articulation root on a child prim such as
        # Robot/torso_link instead of Robot itself.
        prims_to_visit = list(prim.GetChildren())
        while prims_to_visit:
            child_prim = prims_to_visit.pop(0)
            if child_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_prim_path = _regex_path_from_resolved_path(child_prim.GetPath().pathString)
                found_supported_prim_class = True
                break
            prims_to_visit.extend(list(child_prim.GetChildren()))

    if not found_supported_prim_class:
        # Other imported robots can expose a child link path while the articulation
        # root actually lives on an ancestor prim.
        parent_prim = prim.GetParent()
        while parent_prim.IsValid():
            if parent_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_prim_path = _regex_path_from_resolved_path(parent_prim.GetPath().pathString)
                found_supported_prim_class = True
                break
            parent_prim = parent_prim.GetParent()

    if found_supported_prim_class:
        articulation_view: physx.ArticulationView = physics_sim_view.create_articulation_view(
            articulation_prim_path.replace(".*", "*")
        )
    if not found_supported_prim_class:
        raise RuntimeError(
            f"Failed to find a valid prim view class for the prim paths: {prim_path}, For robot motion reference, only"
            " accept articulated prims for now."
        )

    return articulation_view

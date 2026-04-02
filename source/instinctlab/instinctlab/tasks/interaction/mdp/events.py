from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import torch
from pxr import Gf, UsdGeom

import isaacsim.core.utils.prims as prim_utils
from isaaclab.managers import SceneEntityCfg

from instinctlab.envs.mdp import (
    beyondmimic_bin_fail_counter_smoothing,
    match_motion_ref_with_scene,
    reset_robot_state_by_reference,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# TODO(interaction): object trail visualization is temporarily disabled.
# The local implementation had runtime issues and should be revisited before re-enabling.
# Disabled symbols:
# - update_object_reference_visualization
# - reset_object_reference_trail


__all__ = [
    "beyondmimic_bin_fail_counter_smoothing",
    "match_motion_ref_with_scene",
    "randomize_object_scale",
    "reset_robot_state_by_reference",
]


def randomize_object_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    scale_distribution_params: tuple[float, float] = (0.9, 1.1),
    operation: Literal["scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the root-prim scale of per-env rigid objects.

    This is intended for startup-time domain randomization where the object asset is
    spawned from a single USD but each environment receives a different isotropic scale.
    """

    asset = env.scene[asset_cfg.name]
    prim_path_template = asset.cfg.prim_path

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, dtype=torch.long)
    else:
        env_ids = env_ids.detach().cpu().to(torch.long)

    if distribution == "uniform":
        scale_samples = torch.empty(len(env_ids), dtype=torch.float32).uniform_(*scale_distribution_params)
    elif distribution == "log_uniform":
        log_min, log_max = torch.log(torch.tensor(scale_distribution_params, dtype=torch.float32))
        scale_samples = torch.empty(len(env_ids), dtype=torch.float32).uniform_(log_min.item(), log_max.item()).exp()
    elif distribution == "gaussian":
        mean, std = scale_distribution_params
        scale_samples = torch.normal(mean=mean, std=std, size=(len(env_ids),))
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'.")

    for env_id, sampled_scale in zip(env_ids.tolist(), scale_samples.tolist()):
        prim_path = re.sub(r"env_\.\*", f"env_{env_id}", prim_path_template)
        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Failed to find object prim at '{prim_path}' for scale randomization.")

        xformable = UsdGeom.Xformable(prim)
        scale_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)

        base_scale = scale_op.Get()
        if base_scale is None:
            base_scale = Gf.Vec3d(1.0, 1.0, 1.0)
        else:
            base_scale = Gf.Vec3d(base_scale[0], base_scale[1], base_scale[2])

        if operation == "scale":
            new_scale = Gf.Vec3d(
                float(base_scale[0]) * sampled_scale,
                float(base_scale[1]) * sampled_scale,
                float(base_scale[2]) * sampled_scale,
            )
        elif operation == "abs":
            new_scale = Gf.Vec3d(sampled_scale, sampled_scale, sampled_scale)
        else:
            raise ValueError(f"Unsupported operation '{operation}'.")

        scale_op.Set(new_scale)

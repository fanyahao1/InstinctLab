from isaaclab.utils import configclass

from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase

_PARKOUR_DATASETS = "/root/project/InstinctLab/assets_datasets/parkour"


@configclass
class _ParkourMotionCfgBase(AmassMotionCfgBase):
    """Shared defaults for all parkour motion configs."""
    retargetting_func = None
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear


@configclass
class ParkourMotionCfg(_ParkourMotionCfgBase):
    path = f"{_PARKOUR_DATASETS}/parkour_motion_reference"
    filtered_motion_selection_filepath = f"{_PARKOUR_DATASETS}/parkour_motion_reference/parkour_motion_without_run.yaml"
    motion_start_from_middle_range = [0.0, 0.9]
    velocity_estimation_method = "frontbackward"


@configclass
class AmassMotionCfg(_ParkourMotionCfgBase):
    path = f"{_PARKOUR_DATASETS}/motion_amass"
    filtered_motion_selection_filepath = f"{_PARKOUR_DATASETS}/motion_amass/metadata_all.yaml"
    motion_start_from_middle_range = [0.0, 0.2]
    velocity_estimation_method = "frontbackward"


@configclass
class Lafan1MotionCfg(_ParkourMotionCfgBase):
    path = f"{_PARKOUR_DATASETS}/motion_lafan1/LAFAN1_Instinct"
    filtered_motion_selection_filepath = f"{_PARKOUR_DATASETS}/motion_lafan1/LAFAN1_Instinct/metadata.yaml"
    motion_start_from_middle_range = [0.0, 0.2]
    velocity_estimation_method = "frontbackward"


@configclass
class Lafan1_50MotionCfg(_ParkourMotionCfgBase):
    path = f"{_PARKOUR_DATASETS}/motion_lafan1/LAFAN1_Instinct_50"
    filtered_motion_selection_filepath = f"{_PARKOUR_DATASETS}/motion_lafan1/LAFAN1_Instinct_50/metadata.yaml"
    motion_start_from_middle_range = [0.0, 0.2]
    velocity_estimation_method = "frontbackward"
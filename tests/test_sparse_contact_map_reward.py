from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _ensure_package(package_name: str) -> None:
    if package_name in sys.modules:
        return
    module = types.ModuleType(package_name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[package_name] = module


for package in [
    "instinctlab",
    "instinctlab.tasks",
    "instinctlab.tasks.interaction",
    "instinctlab.tasks.interaction.mdp",
]:
    _ensure_package(package)

object_utils_stub = types.ModuleType("instinctlab.tasks.interaction.mdp.object_utils")
object_utils_stub.get_current_object_keys = lambda *args, **kwargs: []  # type: ignore[attr-defined]
object_utils_stub.get_current_object_usd_basenames = lambda *args, **kwargs: []  # type: ignore[attr-defined]
object_utils_stub.get_object_state_w = lambda *args, **kwargs: ()  # type: ignore[attr-defined]
sys.modules["instinctlab.tasks.interaction.mdp.object_utils"] = object_utils_stub


def _load_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_name} at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


contact_geometry = _load_module(
    "instinctlab.tasks.interaction.mdp.contact_geometry",
    "source/instinctlab/instinctlab/tasks/interaction/mdp/contact_geometry.py",
)
contact_map_loader = _load_module(
    "instinctlab.tasks.interaction.mdp.contact_map_loader",
    "source/instinctlab/instinctlab/tasks/interaction/mdp/contact_map_loader.py",
)
sparse_contact_reward = _load_module(
    "instinctlab.tasks.interaction.mdp.sparse_contact_reward",
    "source/instinctlab/instinctlab/tasks/interaction/mdp/sparse_contact_reward.py",
)

compute_link_part_center_distance = contact_geometry.compute_link_part_center_distance
compute_link_part_nearest_distance = contact_geometry.compute_link_part_nearest_distance
transform_part_centers_to_world = contact_geometry.transform_part_centers_to_world
transform_part_points_to_world = contact_geometry.transform_part_points_to_world
load_sparse_contact_map_directory = contact_map_loader.load_sparse_contact_map_directory
SlidingWindowContactBuffer = sparse_contact_reward.SlidingWindowContactBuffer
compute_sparse_contact_reward_components = sparse_contact_reward.compute_sparse_contact_reward_components
extract_mandatory_contact_debug_data = sparse_contact_reward.extract_mandatory_contact_debug_data
resolve_direction_to_arrow_marker = sparse_contact_reward.resolve_direction_to_arrow_marker
bucketize_distance_visualization = sparse_contact_reward.bucketize_distance_visualization
get_visualizer_default_scale = sparse_contact_reward.get_visualizer_default_scale


def test_sparse_contact_metadata_directory_loads_both_objects():
    metadata_dir = ROOT / "datasets" / "interaction" / "output_npz_29dof_with_object" / "sparse_contact_maps"
    metadata = load_sparse_contact_map_directory(metadata_dir)

    assert set(metadata.keys()) == {"chair", "sofa"}
    assert tuple(metadata["chair"].robot_links) == ("pelvis", "torso", "left_hand", "right_hand")
    assert tuple(metadata["chair"].object_parts_order) == ("seat", "back", "arm_left", "arm_right")
    assert tuple(metadata["sofa"].relation.shape) == (4, 4)
    assert metadata["sofa"].exists_mask.tolist() == [True, True, True, True]


def test_contact_geometry_transform_and_distance_consistency():
    points_local = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32)
    centers_local = torch.tensor([[[0.5, 0.5, 0.0]]], dtype=torch.float32)
    object_pos_w = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    object_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    link_pos_w = torch.tensor([[[2.0, 2.0, 3.0]]], dtype=torch.float32)

    points_world = transform_part_points_to_world(points_local, object_pos_w, object_quat_w)
    centers_world = transform_part_centers_to_world(centers_local, object_pos_w, object_quat_w)
    d_nearest = compute_link_part_nearest_distance(link_pos_w, points_world)
    d_center = compute_link_part_center_distance(link_pos_w, centers_world)

    assert torch.allclose(points_world[0, 0, 0], torch.tensor([2.0, 2.0, 3.0]))
    assert torch.allclose(points_world[0, 0, 1], torch.tensor([1.0, 3.0, 3.0]))
    assert torch.allclose(centers_world[0, 0], torch.tensor([1.5, 2.5, 3.0]))
    assert torch.all(d_nearest <= d_center)


def test_sparse_contact_reward_components_follow_relation_masks():
    d_nearest = torch.tensor(
        [
            [
                [0.05, 0.20],
                [0.04, 0.05],
            ]
        ],
        dtype=torch.float32,
    )
    relation = torch.tensor([[1, 0], [-1, 0]], dtype=torch.int64)
    exists_mask = torch.tensor([True, True], dtype=torch.bool)
    gate_ref = torch.tensor([1.0], dtype=torch.float32)
    hold_values = torch.tensor(
        [
            [
                [0.50, 0.00],
                [0.00, 0.25],
            ]
        ],
        dtype=torch.float32,
    )

    components = compute_sparse_contact_reward_components(
        d_nearest=d_nearest,
        relation=relation,
        exists_mask=exists_mask,
        gate_ref=gate_ref,
        hold_values=hold_values,
        contact_distance_threshold=0.10,
        forbid_contact_distance_threshold=0.05,
        mandatory_alpha=10.0,
        optional_beta=5.0,
        mandatory_weights={"proximity": 0.5, "contact": 1.0, "hold": 0.2},
        optional_weights={"proximity": 0.1, "contact": 0.3},
        forbidden_weight=2.0,
    )

    mandatory_expected = 0.5 * torch.exp(torch.tensor(-0.5)) + 1.0 + 0.2 * 0.50
    optional_expected = (0.1 * torch.exp(torch.tensor(-1.0))) + (0.1 * torch.exp(torch.tensor(-0.25)) + 0.3)
    forbidden_expected = torch.tensor(2.0)

    assert torch.allclose(components.mandatory, mandatory_expected.unsqueeze(0))
    assert torch.allclose(components.optional, optional_expected.unsqueeze(0))
    assert torch.allclose(components.forbidden, forbidden_expected.unsqueeze(0))
    assert torch.allclose(components.total, (mandatory_expected + optional_expected - forbidden_expected).unsqueeze(0))


def test_sliding_window_contact_buffer_resets_selected_envs():
    buffer = SlidingWindowContactBuffer(num_envs=2, num_links=1, num_parts=1, window=3, device="cpu")

    first = buffer.update(torch.tensor([[[1.0]], [[0.0]]]))
    second = buffer.update(torch.tensor([[[0.0]], [[1.0]]]))
    buffer.reset([0])
    third = buffer.update(torch.tensor([[[1.0]], [[1.0]]]))

    assert torch.allclose(first[:, 0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(second[:, 0, 0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(third[:, 0, 0], torch.tensor([1.0, 2.0 / 3.0]))


def test_extract_mandatory_contact_debug_data_only_keeps_mandatory_points_and_nearest():
    link_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    part_points_w = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 5.0, 0.0], [0.0, 6.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    relation = torch.tensor([[1, 0], [-1, 0]], dtype=torch.int64)
    exists_mask = torch.tensor([True, True], dtype=torch.bool)
    point_valid_mask = torch.tensor(
        [
            [[True, True], [True, True]],
        ],
        dtype=torch.bool,
    )

    debug_data = extract_mandatory_contact_debug_data(
        link_pos_w=link_pos_w,
        part_points_w=part_points_w,
        relation=relation,
        exists_mask=exists_mask,
        point_valid_mask=point_valid_mask,
        max_envs=1,
    )

    assert debug_data.all_point_positions.shape == (2, 3)
    assert torch.allclose(
        debug_data.all_point_positions,
        torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        debug_data.all_arrow_start_positions,
        torch.zeros((2, 3), dtype=torch.float32),
    )
    assert debug_data.nearest_point_positions.shape == (1, 3)
    assert torch.allclose(debug_data.nearest_point_positions[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(debug_data.nearest_arrow_directions[0], torch.tensor([1.0, 0.0, 0.0]))


def test_resolve_direction_to_arrow_marker_preserves_length_and_start_direction():
    direction = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float32)
    start_point = torch.tensor([[1.0, 1.0, 1.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

    arrow_scale, arrow_quat, arrow_pos = resolve_direction_to_arrow_marker(
        direction=direction,
        start_point=start_point,
        default_scale=(0.1, 0.2, 0.3),
        length_scale_factor=10.0,
    )

    assert arrow_scale.shape == (2, 3)
    assert arrow_quat.shape == (2, 4)
    assert torch.allclose(arrow_scale[:, 0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(arrow_scale[:, 1], torch.tensor([0.2, 0.2]))
    assert torch.allclose(arrow_scale[:, 2], torch.tensor([0.3, 0.3]))
    assert torch.allclose(
        arrow_pos,
        torch.tensor([[1.5, 1.0, 1.0], [4.0, 5.75, 6.0]], dtype=torch.float32),
    )


def test_bucketize_distance_visualization_splits_near_mid_far():
    distances = torch.tensor([0.05, 0.18, 0.42], dtype=torch.float32)
    marker_indices = bucketize_distance_visualization(distances, near_threshold=0.10, mid_threshold=0.30)
    assert marker_indices.tolist() == [0, 1, 2]


def test_get_visualizer_default_scale_supports_named_or_fallback_markers():
    arrow_marker = types.SimpleNamespace(scale=(1.0, 2.0, 3.0))
    near_marker = types.SimpleNamespace(scale=(4.0, 5.0, 6.0))

    arrow_visualizer = types.SimpleNamespace(cfg=types.SimpleNamespace(markers={"arrow": arrow_marker}))
    nearest_visualizer = types.SimpleNamespace(cfg=types.SimpleNamespace(markers={"near": near_marker, "mid": arrow_marker}))

    assert get_visualizer_default_scale(arrow_visualizer, preferred_key="arrow") == (1.0, 2.0, 3.0)
    assert get_visualizer_default_scale(nearest_visualizer, preferred_key="near") == (4.0, 5.0, 6.0)
    assert get_visualizer_default_scale(nearest_visualizer, preferred_key="arrow") == (4.0, 5.0, 6.0)

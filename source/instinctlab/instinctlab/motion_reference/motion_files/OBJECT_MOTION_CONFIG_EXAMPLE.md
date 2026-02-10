# Object Motion配置示例完整指南

本文档提供Object Motion场景匹配的完整配置示例。

## 目录结构

```
assets_datasets/interaction/
├── metadata.yaml                    # 核心配置文件
├── small_box/
│   ├── lift_small_box_01.npz       # object_id = 0
│   ├── carry_small_box_01.npz
│   └── place_small_box_01.npz
├── medium_box/
│   ├── push_medium_box_01.npz      # object_id = 1
│   └── pull_medium_box_01.npz
└── large_box/
    ├── push_large_box_01.npz       # object_id = 2
    └── push_large_box_02.npz
```

## metadata.yaml完整配置

```yaml
# Object Motion Metadata Configuration
# 定义motion文件与object类型的绑定关系

motion_files:
  # 小箱子动作 (30cm cube, 可拿起)
  - motion_file: small_box/lift_small_box_01.npz
    object_id: 0
    weight: 1.0
  - motion_file: small_box/carry_small_box_01.npz
    object_id: 0
    weight: 1.0
  - motion_file: small_box/place_small_box_01.npz
    object_id: 0
    weight: 0.8  # 较低的weight，采样概率较小

  # 中箱子动作 (50cm cube, 需推拉)
  - motion_file: medium_box/push_medium_box_01.npz
    object_id: 1
    weight: 1.2  # 较高的weight，希望多练习推
  - motion_file: medium_box/pull_medium_box_01.npz
    object_id: 1
    weight: 1.0

  # 大箱子动作 (80cm cube, 重推)
  - motion_file: large_box/push_large_box_01.npz
    object_id: 2
    weight: 1.0
  - motion_file: large_box/push_large_box_02.npz
    object_id: 2
    weight: 1.5  # 推荐的动作，更高采样率

objects:
  - object_id: 0
    size: small
    usd_path: small_box_30cm.usd
    mass_kg: 2.0
    description: "Small 30cm cube, light weight, can be lifted"

  - object_id: 1
    size: medium
    usd_path: medium_box_50cm.usd
    mass_kg: 15.0
    description: "Medium 50cm cube, moderate weight, push/pull only"

  - object_id: 2
    size: large
    usd_path: large_box_80cm.usd
    mass_kg: 50.0
    description: "Large 80cm cube, heavy, requires full body pushing"
```

## Motion数据文件格式

每个`.npz`文件应包含人体motion和物体motion：

```python
# 示例：small_box/lift_small_box_01.npz
{
    # 人体运动数据（标准AMASS格式）
    'joint_pos': np.ndarray,      # [num_frames, num_joints]
    'joint_vel': np.ndarray,
    'base_pos': np.ndarray,       # [num_frames, 3]
    'base_quat': np.ndarray,      # [num_frames, 4]
    'base_lin_vel': np.ndarray,
    'base_ang_vel': np.ndarray,
    'framerate': float,

    # 物体运动数据（根据object_data_keys配置）
    'box_pos': np.ndarray,        # [num_frames, 3]
    'box_quat': np.ndarray,       # [num_frames, 4] (可选)
    'box_lin_vel': np.ndarray,    # [num_frames, 3] (可选，可自动估计)
    'box_ang_vel': np.ndarray,    # [num_frames, 3] (可选，可自动估计)
}
```

## 任务配置文件

### 1. Motion配置

```python
# g1_interaction_shadowing_cfg.py

import os
from instinctlab.motion_reference.motion_files.object_motion_cfg import ObjectMotionCfg as ObjectMotionCfgBase

@configclass
class InteractionMotionCfg(ObjectMotionCfgBase):
    # 数据路径
    path = os.path.expanduser("~/datasets/interaction")

    # ===== 物体数据配置 =====
    object_data_keys = {
        "box": "box",  # 从npz文件中读取box_pos, box_quat等
    }

    # 速度估计方法（如果npz没有velocity数据）
    object_velocity_estimation_method = "frontbackward"

    # ===== 场景匹配配置（新增）=====
    metadata_yaml = os.path.expanduser("~/datasets/interaction/metadata.yaml")
    object_matching_key = "usd_path"  # 使用USD路径匹配

    # ===== 其他配置 =====
    filtered_motion_selection_filepath = None
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    velocity_estimation_method = "frontbackward"
    motion_start_height_offset = 0.0
    motion_bin_length_s = 1.0
    env_starting_stub_sampling_strategy = "independent"
```

### 2. Motion Reference Manager配置

```python
motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",

    # Link of interests（用于link_pos/quat观测和奖励）
    link_of_interests=[
        "pelvis", "torso_link",
        "left_shoulder_roll_link", "right_shoulder_roll_link",
        "left_elbow_link", "right_elbow_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
        "left_hip_roll_link", "right_hip_roll_link",
        "left_knee_link", "right_knee_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
    ],

    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    data_start_from="current_time",

    # 注册motion buffer
    motion_buffers={
        "InteractionMotion": InteractionMotionCfg(),
    },

    mp_split_method="Even",
)
```

### 3. 物体配置

```python
OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=[
            "/path/to/small_box_30cm.usd",
            "/path/to/medium_box_50cm.usd",
            "/path/to/large_box_80cm.usd",
        ],
        random_choice=True,  # 每个env随机选一个
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
)
```

### 4. 场景配置

```python
@configclass
class InteractionSceneCfg(InteractiveSceneCfg):
    num_envs = 4096
    env_spacing = 4.0

    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    motion_reference = motion_reference_cfg
    objects = OBJECT_CFG  # 添加物体

    # Terrain（可选，平地即可）
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )
```

### 5. Event配置（关键！）

```python
@configclass
class EventCfg:
    # ===== Startup Events =====

    # 匹配motion与scene中的物体（必须添加）
    match_motion_ref_with_scene = EventTermCfg(
        func=instinct_mdp.match_motion_ref_with_scene,
        mode="startup",
        params={"motion_ref_cfg": SceneEntityCfg("motion_reference")},
    )

    # 物理材质随机化
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
        },
    )

    # ===== Reset Events =====

    # 根据motion reference重置机器人状态
    reset_robot_by_motion = EventTermCfg(
        func=instinct_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("robot"),
            "position_offset": [0.0, 0.0, 0.1],
        },
    )
    reset_objects_by_motion = EventTermCfg(
        func=instinct_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("objects"),
            "object_name": "box", # 关键！必须与object_data_keys中的key一致
            "position_offset": [0.0, 0.0, 0.1],
        },
    )
```

## 运行时行为示例

### 场景初始化（Startup阶段）

```
[ObjectMotion] Loading metadata from: ~/datasets/interaction/metadata.yaml
[ObjectMotion] Found 7 motion files, 3 object types
[ObjectMotion] Matching scene objects with metadata...
[ObjectMotion] Matched object_id 0 (small_box_30cm.usd): 3 motions -> 1365 envs
[ObjectMotion] Matched object_id 1 (medium_box_50cm.usd): 2 motions -> 1350 envs
[ObjectMotion] Matched object_id 2 (large_box_80cm.usd): 2 motions -> 1381 envs
[ObjectMotion] Scene matching completed successfully
```

### 环境Reset时

```python
# 内部逻辑示例
env_id = 42
object_usd = "small_box_30cm.usd"  # 该环境spawn的物体

# 1. 根据object_usd确定object_id
object_id = 0  # small box

# 2. 筛选可用的motions
available_motions = [
    "small_box/lift_small_box_01.npz",
    "small_box/carry_small_box_01.npz",
    "small_box/place_small_box_01.npz",
]

# 3. 根据weight采样一个motion
sampled_motion = "small_box/lift_small_box_01.npz"

# 4. 从motion中提取初始状态和参考数据
motion_ref_data = load_motion(sampled_motion)
robot.reset(motion_ref_data.initial_state)

# 5. 每个step，motion_reference_manager提供:
#    - 人体参考: joint_pos, base_pos, link_pos等
#    - 物体参考: object_data['box']['pos'], object_data['box']['quat']等
```

### 在奖励函数中使用

```python
def compute_object_interaction_reward(env, motion_data: MotionReferenceData):
    # 获取物体参考位置
    ref_box_pos = motion_data.object_data['box']['pos'][:, 0, :]  # [num_envs, 3]

    # 获取场景中实际的物体位置
    actual_box_pos = env.scene.objects.data.root_pos_w  # [num_envs, 3]

    # 计算位置误差
    pos_error = torch.norm(actual_box_pos - ref_box_pos, dim=-1)

    # 高斯奖励
    reward = torch.exp(-pos_error / 0.1)

    return reward
```

## 多物体扩展

如果同时有多个物体（如箱子+球）：

### metadata.yaml

```yaml
motion_files:
  - motion_file: box_ball/throw_ball_to_box_01.npz
    object_id: 0  # 箱子和球的组合

objects:
  - object_id: 0
    usd_path: box_and_ball_combo.usd
    primary_object: box
    secondary_object: ball
```

### Motion配置

```python
@configclass
class MultiObjectMotionCfg(ObjectMotionCfgBase):
    object_data_keys = {
        "box": "box",      # npz中的box_pos, box_quat
        "ball": "ball",    # npz中的ball_pos, ball_quat
    }
```

### 场景配置

需要使用`RigidObjectCollectionCfg`来spawn多个物体：

```python
objects = RigidObjectCollectionCfg(
    rigid_objects={
        "box": RigidObjectCfg(...),
        "ball": RigidObjectCfg(...),
    }
)
```

## 调试技巧

### 1. 验证匹配结果

在训练前添加打印：

```python
motion_buffer = env.scene.motion_reference._motion_buffers["InteractionMotion"]
print("Motion-to-env mask shape:", motion_buffer._all_motion_selectable_envs_mask.shape)
print("Valid motions per env:")
for env_id in range(min(10, num_envs)):
    valid_count = motion_buffer._all_motion_selectable_envs_mask[:, env_id].sum()
    print(f"  Env {env_id}: {valid_count} valid motions")
```

### 2. 可视化参考物体

```python
# 在render loop中
for env_id in range(num_envs):
    ref_pos = motion_data.object_data['box']['pos'][env_id, 0]
    draw_sphere(position=ref_pos, color=(0, 1, 0), radius=0.05)
```

### 3. 检查motion采样分布

```python
motion_ids_history = []
for reset_id in range(1000):
    env.reset(env_ids=[0])
    motion_id = motion_buffer._assigned_env_motion_selection[0]
    motion_ids_history.append(motion_id.item())

# 统计
from collections import Counter
print("Motion sampling distribution:", Counter(motion_ids_history))
```

## 常见问题

**Q: 使用MultiUsdFileCfg的random_choice后，如何正确匹配？**

A: 当前实现假设所有env使用同一USD。如果需要random_choice，有两种方案：
1. 使用`RigidObjectCollectionCfg`，每个object单独配置
2. 重写`_extract_object_properties_from_scene`，从physx view读取实际spawn的USD路径

**Q: 能否根据物体尺寸动态匹配？**

A: 可以，设置`object_matching_key="size"`，并重写匹配方法来读取scale属性。

**Q: weight如何设置？**

A: weight决定采样概率。设置原则：
- 重要/质量好的motion设置较高weight
- 简单/过渡性motion设置较低weight
- 所有weight会自动归一化，相对值重要

**Q: 能否在训练中动态调整weight？**

A: 当前不支持。但可以通过Curriculum逐步enable/disable某些motion。

## 总结

Object Motion的场景匹配系统提供了：
1. ✅ 自动motion-object绑定
2. ✅ 灵活的匹配策略
3. ✅ 与TerrainMotion一致的API
4. ✅ 详细的调试信息输出
5. ✅ 可扩展的自定义匹配逻辑

按照本文档配置后，每个环境将自动加载与其物体类型匹配的motion，确保训练的有效性和正确性。

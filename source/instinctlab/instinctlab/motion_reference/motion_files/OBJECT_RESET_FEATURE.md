# Object Reset功能说明

## 概述

已扩展`reset_robot_state_by_reference`函数以支持同时reset机器人(Articulation)和物体(RigidObject)。

## 功能特性

### 统一的Reset接口

同一个函数`reset_robot_state_by_reference`现在可以处理：
1. **Articulation (机器人)** - 使用motion_ref_init_state
2. **RigidObject (物体)** - 使用motion_ref.data.object_data

函数会自动检测asset类型并调用对应的内部实现。

### 对于机器人 (Articulation)

```python
reset_robot = EventTermCfg(
    func=instinct_mdp.reset_robot_state_by_reference,
    mode="reset",
    params={
        "motion_ref_cfg": SceneEntityCfg("motion_reference"),
        "asset_cfg": SceneEntityCfg("robot"),
        "position_offset": [0.0, 0.0, 0.0],
        "dof_vel_ratio": 1.0,
        "base_lin_vel_ratio": 1.0,
        "base_ang_vel_ratio": 1.0,
        "randomize_pose_range": {...},
        "randomize_velocity_range": {...},
        "randomize_joint_pos_range": (-0.1, 0.1),
    },
)
```

**Reset数据来源：** `motion_ref.get_init_reference_state(env_ids)`

### 对于物体 (RigidObject)

```python
reset_objects = EventTermCfg(
    func=instinct_mdp.reset_robot_state_by_reference,
    mode="reset",
    params={
        "motion_ref_cfg": SceneEntityCfg("motion_reference"),
        "asset_cfg": SceneEntityCfg("objects"),
        "object_name": "box",  # 必须指定！
        "position_offset": [0.0, 0.0, 0.0],
        "base_lin_vel_ratio": 0.0,
        "base_ang_vel_ratio": 0.0,
        "randomize_pose_range": {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.01, 0.01),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.1, 0.1),
        },
    },
)
```

**Reset数据来源：** `motion_ref.data.object_data[object_name]`

**关键参数：**
- `object_name`: **必须提供**，指定从哪个object_data字段读取数据
- `randomize_joint_pos_range`: 物体不使用（会被忽略）

## 工作流程

### 机器人Reset流程

```
1. 调用 reset_robot_state_by_reference()
   ↓
2. 检测asset类型 → Articulation
   ↓
3. 调用 _reset_articulation_state_by_reference()
   ↓
4. 从 motion_ref.get_init_reference_state(env_ids) 获取：
   - base_pos_w, base_quat_w
   - base_lin_vel_w, base_ang_vel_w
   - joint_pos, joint_vel
   ↓
5. 应用 position_offset
   ↓
6. 应用 randomize_pose_range
   ↓
7. 写入 robot 的 root pose
   ↓
8. 应用 velocity randomization
   ↓
9. 写入 robot 的 root velocity
   ↓
10. 应用 joint randomization
   ↓
11. 写入 robot 的 joint state
```

### 物体Reset流程

```
1. 调用 reset_robot_state_by_reference(object_name="box")
   ↓
2. 检测asset类型 → RigidObject
   ↓
3. 调用 _reset_object_state_by_reference()
   ↓
4. 从 motion_ref.data.object_data["box"] 获取：
   - pos: [env_ids, 0, :]
   - quat: [env_ids, 0, :] (可选，默认[1,0,0,0])
   - lin_vel: [env_ids, 0, :] (可选，默认零速度)
   - ang_vel: [env_ids, 0, :] (可选，默认零速度)
   ↓
5. 应用 position_offset
   ↓
6. 应用 randomize_pose_range
   ↓
7. 写入 object 的 root pose
   ↓
8. 应用 velocity ratio 和 randomization
   ↓
9. 写入 object 的 root velocity
```

## 数据对应关系

### Motion Reference Data结构

```python
motion_ref.data = MotionReferenceData(
    # 人体数据
    joint_pos: [num_envs, num_frames, num_joints],
    base_pos_w: [num_envs, num_frames, 3],
    ...

    # 物体数据 (由ObjectMotion填充)
    object_data: {
        "box": {
            "pos": [num_envs, num_frames, 3],
            "quat": [num_envs, num_frames, 4],
            "lin_vel": [num_envs, num_frames, 3],
            "ang_vel": [num_envs, num_frames, 3],
        },
        "ball": {...},  # 如果有多个物体
    }
)
```

### Reset时的数据提取

```python
# 机器人
init_state = motion_ref.get_init_reference_state(env_ids)
# 返回第0帧的数据：
# - joint_pos[env_ids]
# - base_pos_w[env_ids], base_quat_w[env_ids]
# - joint_vel[env_ids]
# - base_lin_vel_w[env_ids], base_ang_vel_w[env_ids]

# 物体
object_data = motion_ref.data.object_data["box"]
# 手动提取第0帧：
# - pos[env_ids, 0, :]
# - quat[env_ids, 0, :]
# - lin_vel[env_ids, 0, :]
# - ang_vel[env_ids, 0, :]
```

## 配置建议

### 机器人Reset参数

```python
"position_offset": [0.0, 0.0, 0.0],  # 通常不需要offset
"dof_vel_ratio": 1.0,                 # 保持原始关节速度
"base_lin_vel_ratio": 1.0,            # 保持原始线速度
"base_ang_vel_ratio": 1.0,            # 保持原始角速度
"randomize_pose_range": {
    "x": (-0.05, 0.05),               # ±5cm
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),               # ±1cm (高度变化小)
    "roll": (-0.1, 0.1),              # ±6度
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),               # ±11度
},
"randomize_joint_pos_range": (-0.1, 0.1),  # ±0.1 rad
```

### 物体Reset参数

```python
"position_offset": [0.0, 0.0, 0.0],
"base_lin_vel_ratio": 0.0,           # 物体从静止开始
"base_ang_vel_ratio": 0.0,
"randomize_pose_range": {
    "x": (-0.02, 0.02),               # ±2cm (比机器人小)
    "y": (-0.02, 0.02),
    "z": (-0.01, 0.01),
    "roll": (-0.05, 0.05),            # ±3度 (比机器人小)
    "pitch": (-0.05, 0.05),
    "yaw": (-0.1, 0.1),               # ±6度
},
"randomize_velocity_range": {},      # 物体不需要初始速度随机化
```

## 错误处理

函数会在以下情况抛出异常：

1. **物体名称未提供**
   ```
   ValueError: object_name must be provided when resetting RigidObject
   ```

2. **物体数据不存在**
   ```
   ValueError: Object 'box' not found in motion reference data.
   Available objects: ['ball', 'cube']
   ```

3. **物体位置数据缺失**
   ```
   ValueError: Object 'box' does not have position data
   ```

4. **不支持的asset类型**
   ```
   ValueError: Unsupported asset type: <class '...'>
   ```

## 使用示例

### 完整配置示例

```python
@configclass
class EventCfg:
    # 启动时匹配场景
    match_motion_ref_with_scene = EventTermCfg(
        func=instinct_mdp.match_motion_ref_with_scene,
        mode="startup",
    )

    # Reset机器人
    reset_robot = EventTermCfg(
        func=instinct_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("robot"),
            "position_offset": [0.0, 0.0, 0.0],
            "randomize_pose_range": {...},
        },
    )

    # Reset物体（使用同一个函数）
    reset_objects = EventTermCfg(
        func=instinct_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("objects"),
            "object_name": "box",  # 关键！
            "position_offset": [0.0, 0.0, 0.0],
            "base_lin_vel_ratio": 0.0,
            "randomize_pose_range": {...},
        },
    )
```

### 多物体Reset

如果有多个物体需要reset：

```python
reset_box = EventTermCfg(
    func=instinct_mdp.reset_robot_state_by_reference,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("objects", body_names="box"),
        "object_name": "box",
        ...
    },
)

reset_ball = EventTermCfg(
    func=instinct_mdp.reset_robot_state_by_reference,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("objects", body_names="ball"),
        "object_name": "ball",
        ...
    },
)
```

## 优势

1. ✅ **复用现有函数** - 无需创建新的reset函数
2. ✅ **统一接口** - 机器人和物体使用相同的API
3. ✅ **自动类型检测** - 根据asset类型自动选择处理逻辑
4. ✅ **灵活的随机化** - 支持pose和velocity的随机化
5. ✅ **清晰的错误提示** - 参数错误时有明确的提示信息

## 注意事项

1. 物体reset时**必须提供object_name**参数
2. object_name必须与metadata.yaml中的object_data_keys一致
3. 物体通常不需要初始速度（设置velocity_ratio=0.0）
4. 物体的randomization范围应该比机器人小
5. 确保ObjectMotion的match_scene已在startup时执行
6. 物体数据来自motion_ref.data.object_data的第0帧

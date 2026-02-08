# ObjectMotion 快速开始指南

## 概述

ObjectMotion 是一个运动数据加载器，用于处理包含**人类动作**和**物体动作**的同步录制。它扩展自 AmassMotion，支持为强化学习提供物体位置和速度数据用于 reward 计算。

## 核心特性

✅ **同步加载**: 人类和物体动作同步对齐
✅ **灵活格式**: 支持自定义 NPZ 键名映射
✅ **自动速度估计**: 缺少速度数据时自动计算
✅ **多物体支持**: 同时处理多个物体
✅ **深度集成**: 直接集成到 MotionReferenceData 供 reward 使用

## 数据准备

您的 NPZ 文件应包含：

```
motion_file.npz
├── poses (N, num_joints)           # 人类关节旋转 (SMPL)
├── trans (N, 3)                     # 人类根节点位置
├── mocap_framerate (scalar)         # 帧率
├── ball_pos (N, 3)                  # 球体位置 [必需]
├── ball_quat (N, 4)                 # 球体四元数 [可选]
├── ball_lin_vel (N, 3)              # 球体线速度 [可选/自动估计]
├── cup_pos (N, 3)                   # 杯子位置 [必需]
└── cup_quat (N, 4)                  # 杯子四元数 [可选]
```

## 配置示例

```python
from instinctlab.motion_reference.motion_files.object_motion_cfg import ObjectMotionCfg

cfg = ObjectMotionCfg(
    path="data/motions",

    # 物体名称 -> NPZ 数据键前缀的映射
    object_data_keys={
        "ball": "ball",
        "cup": "cup",
    },

    # 物体数量
    object_count=2,

    # 速度估计方法: "frontward", "backward", "frontbackward", None
    object_velocity_estimation_method="frontward",

    # 人类运动相关配置（继承自 AmassMotionCfg）
    supported_file_endings=["poses.npz", "retargeted.npz"],
    skip_frames=0,
    motion_target_framerate=50.0,
    velocity_estimation_method="frontward",
    retargetting_func=my_retarget_func,  # 必须提供
    motion_start_height_offset=0.0,
)
```

## 使用方式

```python
from instinctlab.motion_reference.motion_files import ObjectMotion

# 创建加载器
motion_buffer = ObjectMotion(
    cfg,
    articulation_view,
    link_of_interests,
    forward_kinematics_func,
    device,
)

# 在重置时初始化
motion_buffer.reset(env_ids, symmetric_augmentation_mask)
motion_buffer.fill_init_reference_state(env_ids, env_origins, state_buffer)

# 在步进中填充数据
motion_buffer.fill_motion_data(env_ids, sample_timestamp, env_origins, data_buffer)

# 访问物体数据
ball_pos = data_buffer.object_data["ball"]["pos"]        # (num_envs, num_frames, 3)
ball_quat = data_buffer.object_data["ball"]["quat"]      # (num_envs, num_frames, 4)
ball_lin_vel = data_buffer.object_data["ball"]["lin_vel"]  # (num_envs, num_frames, 3)
```

## Reward 计算

在您的环境 reward 函数中使用物体数据：

```python
def compute_object_tracking_error(object_data, target_pos, target_quat=None):
    """计算物体追踪误差"""

    # 位置误差
    pos_error = torch.norm(
        object_data["pos"] - target_pos,
        dim=-1
    )

    # 旋转误差（如果有四元数）
    rot_error = 0.0
    if "quat" in object_data and target_quat is not None:
        # 四元数距离
        dot_product = torch.sum(
            object_data["quat"] * target_quat,
            dim=-1
        )
        rot_error = 1.0 - torch.abs(dot_product)

    # 组合误差
    error = pos_error + 0.5 * rot_error

    # 转换为 reward（指数递减）
    reward = torch.exp(-10.0 * error)

    return reward

# 在环境 step 函数中使用
rewards = compute_object_tracking_error(
    data_buffer.object_data["ball"],
    target_ball_pos,
    target_ball_quat,
)
```

## 高级用法

### 多物体场景

```python
cfg = ObjectMotionCfg(
    path="data/complex_motions",
    object_data_keys={
        "primary": "primary_obj",
        "secondary": "secondary_obj",
        "goal": "goal_position",
        "obstacle_1": "obs1",
        "obstacle_2": "obs2",
    },
    object_count=5,
    object_velocity_estimation_method="frontbackward",  # 更平滑
    retargetting_func=my_retarget_func,
)
```

### 自定义键名

```python
# 如果您的 NPZ 使用非标准键名
cfg = ObjectMotionCfg(
    path="data/custom",
    object_data_keys={
        "target": "target_object",  # NPZ 包含: target_object_pos, etc.
        "hand": "hand_object",      # NPZ 包含: hand_object_pos, etc.
    },
    object_count=2,
    retargetting_func=my_retarget_func,
)
```

### 无速度数据的场景

```python
# 如果 NPZ 中没有速度数据，ObjectMotion 会自动估计
cfg = ObjectMotionCfg(
    path="data/position_only",
    object_data_keys={"ball": "ball"},
    object_count=1,
    object_velocity_estimation_method="frontward",  # 或 "backward", "frontbackward"
    retargetting_func=my_retarget_func,
)

# 现在 data_buffer.object_data["ball"]["lin_vel"] 会被自动填充
```

## 数据结构参考

### MotionSequence.object_data
```python
{
    "object_name": {
        "pos": torch.Tensor,      # (batch, num_frames, 3)
        "quat": torch.Tensor,     # (batch, num_frames, 4) or None
        "lin_vel": torch.Tensor,  # (batch, num_frames, 3) or None
        "ang_vel": torch.Tensor,  # (batch, num_frames, 3) or None
    }
}
```

### MotionReferenceData.object_data
```python
{
    "object_name": {
        "pos": torch.Tensor,      # (num_envs, num_frames, 3)
        "quat": torch.Tensor,     # (num_envs, num_frames, 4) or None
        "lin_vel": torch.Tensor,  # (num_envs, num_frames, 3) or None
        "ang_vel": torch.Tensor,  # (num_envs, num_frames, 3) or None
    }
}
```

## 常见问题

**Q: 物体数据是否必须在 NPZ 中？**
A: 仅位置（pos）是必需的。其他字段（quat, velocities）是可选的，可被自动估计。

**Q: 支持多少个物体？**
A: 理论上没有限制，但性能取决于内存和计算资源。

**Q: 是否可以在训练中动态改变物体数据？**
A: 当前实现在初始化时固定物体集合。运行时修改需要扩展实现。

**Q: 物体数据如何与 Reward 同步？**
A: fill_motion_data() 同时填充人类运动和物体数据，确保完全同步。

**Q: 是否支持物体数据的对称增强？**
A: 当前不支持。可根据需要扩展。

## 文件位置

- **配置类**: `instinctlab/motion_reference/motion_files/object_motion_cfg.py`
- **实现**: `instinctlab/motion_reference/motion_files/object_motion.py`
- **示例**: `instinctlab/motion_reference/motion_files/object_motion_examples.py`
- **文档**: `instinctlab/motion_reference/motion_files/OBJECT_MOTION_README.md`

## 下一步

1. 准备包含物体数据的 NPZ 文件
2. 创建 ObjectMotionCfg 配置
3. 集成到您的环境 reward 函数
4. 训练并验证物体追踪性能

更多详细信息请参考 [OBJECT_MOTION_README.md](./OBJECT_MOTION_README.md)。

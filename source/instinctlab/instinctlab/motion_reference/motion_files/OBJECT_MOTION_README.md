# ObjectMotion - Human and Object Motion Data Loader

ObjectMotion 是一个为 Project-Instinct 设计的数据集加载器，用于处理包含人类动作和物体动作的同步录制数据。

## 功能

- 从 NPZ 格式的运动数据文件中加载人类和物体的动作
- 自动估计物体的速度（线性和角速度）
- 支持多个物体的同时加载
- 将物体数据存储在 MotionReferenceData 中供 reward 计算使用

## 数据格式

NPZ 文件应包含以下键：

### 人类动作（由 AmassMotion 处理）
- `poses`: 关节旋转 (SMPL 格式), shape: (N, num_joints)
- `trans`: 根节点位置, shape: (N, 3)
- `mocap_framerate`: 动作帧率 (标量)

### 物体动作（ObjectMotion 处理）

对于每个物体 (名称为 `object_name`)：

**必需：**
- `{object_name}_pos`: 物体位置, shape: (N, 3)

**可选：**
- `{object_name}_quat`: 物体四元数 (w, x, y, z), shape: (N, 4)
- `{object_name}_lin_vel`: 物体线性速度, shape: (N, 3)
- `{object_name}_ang_vel`: 物体角速度, shape: (N, 3)

如果未提供速度数据，ObjectMotion 可以根据位置/旋转自动估计。

## 配置示例

```python
from instinctlab.motion_reference.motion_files import ObjectMotionCfg

cfg = ObjectMotionCfg(
    path="/path/to/motion/data",
    object_data_keys={
        "ball": "ball",  # NPZ 中数据键前缀
        "cup": "cup",
    },
    object_count=2,
    object_velocity_estimation_method="frontward",
    supported_file_endings=["poses.npz", "retargeted.npz"],
    skip_frames=0,
    motion_target_framerate=50.0,
    velocity_estimation_method="frontward",
    retargetting_func=your_retargeting_function,
    motion_start_height_offset=0.0,
)
```

## 使用方式

在运动参考管理器中使用 ObjectMotion：

```python
motion_buffer = ObjectMotion(cfg, articulation_view, link_names, forward_kinematics_func, device)

# 重置时初始化运动
motion_buffer.reset(env_ids, symmetric_augmentation_mask)

# 填充初始化状态
motion_buffer.fill_init_reference_state(env_ids, env_origins, state_buffer)

# 在步进中填充运动数据（包括物体数据）
motion_buffer.fill_motion_data(env_ids, sample_timestamp, env_origins, data_buffer)

# 访问物体数据用于 reward 计算
ball_pos = data_buffer.object_data["ball"]["pos"]  # shape: (num_envs, num_frames, 3)
ball_quat = data_buffer.object_data["ball"]["quat"]  # shape: (num_envs, num_frames, 4)
ball_lin_vel = data_buffer.object_data["ball"]["lin_vel"]  # shape: (num_envs, num_frames, 3)
```

## Reward 计算示例

```python
def compute_object_tracking_reward(target_object_pos, current_object_pos, scale=1.0):
    """计算物体跟踪误差作为 reward"""
    error = torch.norm(current_object_pos - target_object_pos, dim=-1)
    reward = torch.exp(-scale * error)
    return reward

# 在环境的 reward 函数中
object_error = compute_object_tracking_reward(
    data_buffer.object_data["ball"]["pos"],
    desired_ball_pos,
    scale=10.0
)
```

## 实现细节

### 对象数据存储

对象数据存储在 MotionSequence 和 MotionReferenceData 中作为字典：

```python
# MotionSequence 中
motion_seq.object_data = {
    "ball": {
        "pos": torch.Tensor,      # shape: (N, num_frames, 3)
        "quat": torch.Tensor,     # shape: (N, num_frames, 4) or None
        "lin_vel": torch.Tensor,  # shape: (N, num_frames, 3) or None
        "ang_vel": torch.Tensor,  # shape: (N, num_frames, 3) or None
    },
    "cup": { ... }
}

# MotionReferenceData 中
data_buffer.object_data = {
    "ball": {
        "pos": torch.Tensor,      # shape: (num_envs, num_frames, 3)
        "quat": torch.Tensor,     # shape: (num_envs, num_frames, 4) or None
        "lin_vel": torch.Tensor,  # shape: (num_envs, num_frames, 3) or None
        "ang_vel": torch.Tensor,  # shape: (num_envs, num_frames, 3) or None
    },
    "cup": { ... }
}
```

### 速度估计

如果 NPZ 文件中不存在速度数据，ObjectMotion 可以自动估计：

- **线性速度**: 从物体位置使用有限差分法
- **角速度**: 从物体四元数变化估计

支持的估计方法：
- `"frontward"`: 使用前向差分
- `"backward"`: 使用后向差分
- `"frontbackward"`: 同时使用前向和后向差分（结果更平滑）
- `None`: 不估计（假设数据已提供）

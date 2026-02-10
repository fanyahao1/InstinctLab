# Object Motion场景匹配功能 - 实现总结

本文档总结了为Object Motion添加的场景匹配功能，使其能够根据环境中spawn的物体类型自动加载正确的运动数据。

## 功能概述

参考TerrainMotion的实现，为ObjectMotion添加了**物体-运动绑定系统**，解决了以下问题：

### 核心问题
在交互任务中，不同大小/类型的物体需要不同的运动数据：
- 小箱子 → 拿起、搬运动作
- 中箱子 → 推、拉动作
- 大箱子 → 全身用力推动作

如果不匹配，会导致：
- ❌ 机器人尝试拿起一个80cm的大箱子（物理上不可能）
- ❌ 机器人推一个30cm的小箱子（任务不符合预期）

### 解决方案
通过`object_id`绑定机制，在环境初始化时：
1. 读取metadata.yaml中的motion-object映射关系
2. 检测每个环境spawn的物体类型
3. 为每个环境筛选匹配的motion
4. 确保motion采样时只选择有效的motion

## 实现的文件

### 1. 核心代码文件

#### `object_motion_cfg.py`
**新增配置项：**
```python
metadata_yaml: str | None = None
"""metadata.yaml文件路径，定义motion-object绑定"""

object_matching_key: str = "usd_path"
"""匹配键，支持: usd_path, size, type等"""
```

#### `object_motion.py`
**新增方法：**
- `enable_trajectories()` - 支持轨迹过滤时同步更新object相关属性
- `_refresh_motion_file_list()` - 从metadata.yaml读取配置，构建_all_motion_object_ids
- `match_scene()` - 核心方法，匹配场景中的物体与motion
- `_extract_object_properties_from_scene()` - 从场景提取物体属性
- `_match_object_property()` - 判断物体属性是否匹配
- `_sample_assigned_env_starting_stub()` - 重写采样逻辑，调用安全重采样
- `_safe_motion_resampling_for_objects()` - 确保只采样有效motion

**新增属性：**
```python
self._all_motion_object_ids: torch.Tensor  # [N_motions]
self._all_motion_selectable_envs_mask: torch.Tensor  # [N_motions, N_envs]
```

### 2. 配置文件

#### `metadata_example.yaml`
示例metadata文件，展示如何定义motion_files和objects的映射关系。

#### `g1_interaction_shadowing_cfg.py`
更新了InteractionMotionCfg，添加了：
```python
metadata_yaml = "PATH/TO/metadata.yaml"
object_matching_key = "usd_path"
```

### 3. 文档文件

#### `OBJECT_MOTION_MATCHING_README.md`
详细的使用指南，包括：
- 工作原理
- 配置选项说明
- 自定义匹配逻辑
- 与TerrainMotion对比
- 故障排查

#### `OBJECT_MOTION_CONFIG_EXAMPLE.md`
完整的配置示例，包括：
- 目录结构
- metadata.yaml完整示例
- Motion数据文件格式
- 任务配置文件
- 运行时行为
- 调试技巧

## 核心实现逻辑

### 数据结构

```python
# metadata.yaml结构
{
    'motion_files': [
        {'motion_file': 'small_box/lift.npz', 'object_id': 0, 'weight': 1.0},
        {'motion_file': 'large_box/push.npz', 'object_id': 2, 'weight': 1.0},
    ],
    'objects': [
        {'object_id': 0, 'usd_path': 'small_box.usd'},
        {'object_id': 2, 'usd_path': 'large_box.usd'},
    ]
}

# 内部数据结构
_all_motion_object_ids: [0, 0, 0, 2, 2, ...]  # shape: [N_motions]
_all_motion_selectable_envs_mask: [
    [True, False, True, ...],   # motion 0 可用于 env 0, 2, ...
    [True, False, True, ...],   # motion 1 可用于 env 0, 2, ...
    [False, True, False, ...],  # motion 3 可用于 env 1, ...
]  # shape: [N_motions, N_envs]
```

### 匹配流程

```
1. 启动阶段 (startup event)
   ↓
2. match_scene() 被调用
   ↓
3. 读取metadata.yaml
   ↓
4. 从scene中提取每个env的物体属性
   ├─ object_matching_key="usd_path" → 提取USD路径
   ├─ object_matching_key="size" → 计算物体尺寸
   └─ 自定义key → 自定义提取逻辑
   ↓
5. 构建 object_id → [env_ids] 映射
   ↓
6. 更新 _all_motion_selectable_envs_mask
   ↓
7. 训练过程中，每次采样motion时检查mask
```

### 安全采样机制

```python
def _safe_motion_resampling_for_objects(env_ids):
    # 检查当前分配的motion是否与env匹配
    motion_ids = self._assigned_env_motion_selection[assigned_ids]
    invalid_mask = ~self._all_motion_selectable_envs_mask[motion_ids, assigned_ids]

    # 对不匹配的env重新采样
    if invalid_mask.any():
        for assigned_id in invalid_assigned_ids:
            # 找到该env可用的所有motion
            valid_motions = torch.where(
                self._all_motion_selectable_envs_mask[:, assigned_id]
            )[0]

            # 根据weight采样
            resampled_motion_id = sample_from(valid_motions, weights)
            self._assigned_env_motion_selection[assigned_id] = resampled_motion_id
```

## 使用流程

### 1. 准备数据集

```bash
assets_datasets/interaction/
├── metadata.yaml
├── small_box/
│   ├── lift_01.npz
│   └── carry_01.npz
├── medium_box/
│   └── push_01.npz
└── large_box/
    └── push_01.npz
```

### 2. 编写metadata.yaml

```yaml
motion_files:
  - motion_file: small_box/lift_01.npz
    object_id: 0
objects:
  - object_id: 0
    usd_path: small_box_30cm.usd
```

### 3. 配置任务

```python
@configclass
class InteractionMotionCfg(ObjectMotionCfg):
    path = "~/datasets/interaction"
    metadata_yaml = "~/datasets/interaction/metadata.yaml"
    object_matching_key = "usd_path"
    object_data_keys = {"box": "box"}
```

### 4. 配置场景

```python
@configclass
class SceneCfg(InteractiveSceneCfg):
    objects = RigidObjectCfg(
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=[
                "small_box_30cm.usd",
                "large_box_80cm.usd",
            ],
            random_choice=True,
        ),
    )
```

### 5. 添加启动事件

```python
@configclass
class EventCfg:
    match_motion_ref_with_scene = EventTermCfg(
        func=instinct_mdp.match_motion_ref_with_scene,
        mode="startup",
    )
```

### 6. 运行训练

```bash
python scripts/instinct_rl/train.py \
    --task=Instinct-Interaction-G1-v0 \
    --num_envs=4096
```

终端输出：
```
[ObjectMotion] Matched object_id 0 (small_box_30cm.usd): 2 motions -> 2048 envs
[ObjectMotion] Matched object_id 2 (large_box_80cm.usd): 1 motions -> 2048 envs
[ObjectMotion] Scene matching completed successfully
```

## 与TerrainMotion的对比

| 特性 | TerrainMotion | ObjectMotion |
|------|---------------|--------------|
| **绑定对象** | terrain_id → terrain | object_id → object |
| **匹配方式** | 根据subterrain difficulty | 根据object spawn config |
| **数据结构** | origins_in_scene [N_motions, max_origins, 3] | selectable_envs_mask [N_motions, N_envs] |
| **采样约束** | 同terrain可选多origin | 同object_id只能用于匹配envs |
| **应用场景** | 地形自适应训练 | 物体交互训练 |
| **metadata** | terrain_id + terrain_file | object_id + object properties |

## 扩展性

### 支持多物体

```python
object_data_keys = {
    "box": "box",
    "ball": "ball",
}
```

metadata.yaml中可以使用组合object_id。

### 自定义匹配键

```python
object_matching_key = "size"  # 或 "type", "name" 等

# 重写匹配方法
def _extract_object_properties_from_scene(self, objects_asset):
    # 自定义逻辑读取size
    return ["small", "large", "medium", ...]

def _match_object_property(self, env_property, target_value):
    return env_property == target_value
```

### 动态权重调整

虽然当前不支持运行时调整weight，但可以通过Curriculum实现：

```python
@configclass
class CurriculumCfg:
    enable_advanced_motions = CurriculumTermCfg(
        func=enable_specific_motions,
        params={"motion_ids": [5, 6, 7]},  # 逐步解锁高难度motion
    )
```

## 测试建议

### 单元测试

```python
def test_object_motion_matching():
    # 1. 创建mock scene
    scene = create_mock_scene_with_objects()

    # 2. 创建ObjectMotion实例
    motion = ObjectMotion(cfg)

    # 3. 执行匹配
    motion.match_scene(scene)

    # 4. 验证mask正确性
    assert motion._all_motion_selectable_envs_mask.shape == (N_motions, N_envs)
    assert motion._all_motion_selectable_envs_mask[0, :10].all()  # motion 0适用于前10个env
```

### 集成测试

```bash
# 使用PLAY模式测试单个环境
python scripts/instinct_rl/train.py \
    --task=Instinct-Interaction-G1-Play-v0 \
    --headless
```

检查终端输出，确认：
- ✅ metadata加载成功
- ✅ 匹配信息正确
- ✅ 没有"No valid motions"警告

## 已知限制

1. **MultiUsdFileCfg的random_choice**
   - 当前实现假设所有env使用同一USD路径
   - 如需per-env不同USD，需使用RigidObjectCollectionCfg

2. **运行时weight调整**
   - 当前不支持训练中动态调整motion weight
   - 需要重启训练来应用新的weight

3. **多进程支持**
   - 与TerrainMotion一样，object matching在每个进程独立执行
   - mp_split_method="Even"确保motion均匀分布

## 未来改进方向

1. **自动物体属性检测**
   - 从USD文件自动读取尺寸、重量等属性
   - 减少手动配置metadata的工作量

2. **分层匹配**
   - 支持多级匹配（如：type → size → specific_model）
   - 更灵活的motion分组

3. **在线权重调整**
   - 根据训练进度动态调整motion采样分布
   - 实现自适应curriculum

4. **可视化工具**
   - 提供工具可视化motion-object匹配关系
   - 帮助调试和优化配置

## 总结

Object Motion场景匹配功能成功实现了：

✅ **自动化** - 无需手动为每个env指定motion
✅ **灵活性** - 支持多种匹配策略（usd_path, size, type等）
✅ **可靠性** - 通过mask确保采样的motion始终有效
✅ **可扩展** - 易于添加自定义匹配逻辑
✅ **一致性** - 与TerrainMotion API保持一致

该功能为物体交互任务的训练提供了坚实基础，确保每个环境使用正确的运动数据，避免物理不可行或任务不匹配的情况。

## 相关文档

- [OBJECT_MOTION_MATCHING_README.md](OBJECT_MOTION_MATCHING_README.md) - 详细使用指南
- [OBJECT_MOTION_CONFIG_EXAMPLE.md](OBJECT_MOTION_CONFIG_EXAMPLE.md) - 完整配置示例
- [metadata_example.yaml](InstinctLab/assets_datasets/interaction/metadata_example.yaml) - 示例metadata文件
- [MOTION_REFERENCE_ANALYSIS.md](MOTION_REFERENCE_ANALYSIS.md) - Motion Reference系统分析

## 快速开始

1. 复制 `metadata_example.yaml` 并修改为你的数据集
2. 在任务配置中添加 `metadata_yaml` 和 `object_matching_key`
3. 在EventCfg中添加 `match_motion_ref_with_scene`
4. 运行训练并检查终端输出

完成！你的Object Motion现在会自动匹配场景中的物体了。

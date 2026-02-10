# Object Motion Matching System

本文档解释如何使用Object Motion的场景匹配功能，以确保每个环境加载与其物体类型相匹配的运动数据。

## 概述

在交互任务中（如推箱子、拿物体等），不同大小或类型的物体需要不同的运动数据：
- **小箱子**：可以用双手拿起、搬运
- **中箱子**：需要推或拉
- **大箱子**：需要用全身力量推动

Object Motion系统通过**object_id绑定机制**，确保每个环境根据其spawned的物体类型，自动加载正确的运动数据。

## 工作原理

### 1. 数据准备

在`metadata.yaml`中定义运动文件与物体的对应关系：

```yaml
motion_files:
  - motion_file: small_box/lift_small_box.npz
    object_id: 0  # 小箱子的ID
  - motion_file: large_box/push_large_box.npz
    object_id: 2  # 大箱子的ID

objects:
  - object_id: 0
    usd_path: small_box_30cm.usd
  - object_id: 2
    usd_path: large_box_80cm.usd
```

### 2. 配置ObjectMotion

在任务配置文件中设置metadata路径：

```python
@configclass
class InteractionMotionCfg(ObjectMotionCfg):
    path = "PATH/TO/MOTION/DATA"
    object_data_keys = {"box": "box"}  # 物体数据的key前缀

    # 关键配置：metadata文件路径
    metadata_yaml = "PATH/TO/metadata.yaml"

    # 匹配方式（默认使用USD路径匹配）
    object_matching_key = "usd_path"
```

### 3. 场景初始化时的匹配

在环境的启动事件中调用`match_motion_ref_with_scene`：

```python
@configclass
class EventCfg:
    match_motion_with_scene = EventTermCfg(
        func=instinct_mdp.match_motion_ref_with_scene,
        mode="startup",
        params={"motion_ref_cfg": SceneEntityCfg("motion_reference")},
    )
```

### 4. 匹配流程

```
场景初始化
    ↓
读取metadata.yaml
    ↓
提取每个环境的物体属性（如USD路径）
    ↓
根据object_matching_key匹配object_id
    ↓
构建映射：object_id → env_ids列表
    ↓
更新_all_motion_selectable_envs_mask
    ↓
每次采样motion时检查mask，确保只采样有效的motion
```

## 配置选项

### object_matching_key

决定如何匹配场景中的物体与metadata中的定义：

#### `"usd_path"` (默认，推荐)
通过USD文件路径匹配，最可靠的方式。

**场景配置示例：**
```python
objects = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path="PATH/TO/small_box_30cm.usd"
    ),
)
```

**Metadata配置：**
```yaml
objects:
  - object_id: 0
    usd_path: small_box_30cm.usd  # 文件名匹配
```

#### `"size"` (需要自定义实现)
通过物体尺寸分类匹配（small/medium/large）。

**适用场景：**
- 同一USD文件但通过scale参数调整大小
- 需要根据物体实际尺寸动态匹配

**实现方式：**
需要重写`_extract_object_properties_from_scene()`来读取物体的scale或计算实际尺寸。

#### `"type"` (需要自定义实现)
通过物体类型匹配（box/sphere/cylinder）。

**适用场景：**
- 不同几何形状的物体需要不同的交互方式
- 例如：推箱子 vs 滚球

### 自定义匹配逻辑

如果默认的`usd_path`匹配不满足需求，可以重写相关方法：

```python
class CustomObjectMotion(ObjectMotion):
    def _extract_object_properties_from_scene(self, objects_asset) -> list[str]:
        """提取自定义属性"""
        properties = []
        for env_id in range(objects_asset.num_instances):
            # 示例：读取物体的scale
            scale = objects_asset.root_physx_view.get_scales()[env_id]
            avg_scale = scale.mean().item()

            # 根据scale分类
            if avg_scale < 0.4:
                properties.append("small")
            elif avg_scale < 0.7:
                properties.append("medium")
            else:
                properties.append("large")

        return properties

    def _match_object_property(self, env_property: str, target_value: str) -> bool:
        """自定义匹配逻辑"""
        return env_property == target_value
```

## 完整示例

### 1. 数据集结构

```
assets_datasets/interaction/
├── metadata.yaml
├── small_box/
│   ├── lift_01.npz
│   ├── carry_01.npz
│   └── place_01.npz
├── medium_box/
│   ├── push_01.npz
│   └── pull_01.npz
└── large_box/
    ├── push_01.npz
    └── push_02.npz
```

### 2. 任务配置

```python
@configclass
class InteractionMotionCfg(ObjectMotionCfg):
    path = os.path.expanduser("~/datasets/interaction")
    metadata_yaml = os.path.expanduser("~/datasets/interaction/metadata.yaml")
    object_data_keys = {"box": "box"}
    object_matching_key = "usd_path"
    object_velocity_estimation_method = "frontbackward"

motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    motion_buffers={
        "InteractionMotion": InteractionMotionCfg(),
    },
)

@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = G1_CFG
    motion_reference = motion_reference_cfg

    # 物体配置：每个环境spawn一个箱子
    objects = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=[
                "PATH/TO/small_box_30cm.usd",
                "PATH/TO/medium_box_50cm.usd",
                "PATH/TO/large_box_80cm.usd",
            ],
            random_choice=True,  # 随机选择一个
        ),
    )

@configclass
class EventCfg:
    match_motion_with_scene = EventTermCfg(
        func=instinct_mdp.match_motion_ref_with_scene,
        mode="startup",
    )
```

### 3. 运行时行为

```
环境0: spawn了small_box_30cm.usd
    → object_id = 0
    → 可用motion: lift_01, carry_01, place_01

环境1: spawn了large_box_80cm.usd
    → object_id = 2
    → 可用motion: push_01, push_02

环境2: spawn了medium_box_50cm.usd
    → object_id = 1
    → 可用motion: push_01, pull_01
```

## 与TerrainMotion的对比

| 特性 | TerrainMotion | ObjectMotion |
|------|---------------|--------------|
| **绑定对象** | terrain_id → terrain_origins | object_id → env_ids |
| **匹配依据** | Subterrain difficulty | Object spawn config |
| **数据结构** | origins_in_scene [N_motions, max_origins, 3] | selectable_envs_mask [N_motions, N_envs] |
| **采样约束** | 同一terrain_id的motions可选多个origin | 同一object_id的motions只能用于匹配的envs |
| **使用场景** | 地形自适应（parkour） | 物体交互（manipulation） |

## 注意事项

1. **metadata.yaml必须提供**
   - 如果不提供，系统会退化为普通的AmassMotion行为
   - 所有motion对所有环境都可用（不推荐）

2. **object_id必须唯一**
   - 每个motion_file必须有唯一的object_id
   - 每个object定义也必须有唯一的object_id

3. **确保所有envs都有匹配的motion**
   - 如果某个环境spawn的物体没有对应的motion，会打印警告
   - 该环境将无法采样到有效motion，可能导致训练异常

4. **MultiUsdFileCfg的random_choice**
   - 当前实现假设所有环境使用相同的USD路径
   - 如果使用`random_choice=True`，需要使用`RigidObjectCollectionCfg`或自定义逻辑

## 推荐工作流程

1. **数据录制阶段**
   - 为每种物体类型单独录制motion
   - 在motion文件中保存物体的pos/quat等信息
   - 使用一致的命名约定（如`{object_type}_{action}_##.npz`）

2. **数据整理阶段**
   - 创建metadata.yaml，定义object_id
   - 确保usd_path与场景配置中的路径一致
   - 设置合理的weight，控制motion采样分布

3. **任务配置阶段**
   - 配置`metadata_yaml`路径
   - 选择合适的`object_matching_key`
   - 在EventCfg中添加`match_motion_with_scene`

4. **测试验证**
   - 使用PLAY模式（num_envs=1）测试
   - 检查终端输出的匹配信息
   - 验证每个环境加载的motion是否正确

## 故障排查

### 问题：所有envs显示"No valid motions"

**原因：** 匹配失败，没有motion被分配到环境

**解决方案：**
1. 检查metadata.yaml中的`usd_path`是否与场景配置一致
2. 确认`object_matching_key`设置正确
3. 查看终端输出的匹配日志

### 问题：部分motion从未被采样

**原因：** weight设置不当或object_id配置错误

**解决方案：**
1. 调整metadata.yaml中的weight值
2. 确认motion_file的object_id正确
3. 增加对应object_id的环境数量

### 问题：Motion与物体大小不匹配

**原因：** object_id绑定错误

**解决方案：**
1. 重新检查metadata.yaml的object_id分配
2. 确认场景中spawn的USD文件与metadata一致
3. 考虑使用自定义匹配逻辑（如size-based）

## 总结

Object Motion的场景匹配系统提供了灵活且可靠的机制，确保：
- ✅ 每个环境自动加载与其物体类型匹配的motion
- ✅ 训练过程中motion采样始终有效
- ✅ 支持多物体类型的混合训练
- ✅ 易于扩展和自定义

参考`TerrainMotion`的成功经验，该系统已在地形自适应任务中得到验证，现在应用于物体交互场景。

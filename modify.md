# OSDaR23 LiDAR–相机标定升级 TODO（供 Cursor 执行的改造计划）

## 文档用途

本文档是一份**按执行顺序组织的、可直接交给 Cursor 使用的逐步改造 TODO 清单**。
目标是将当前项目从：

* 图像侧：`SAM 边缘 + LSD 线 + 距离场`
* LiDAR 侧：`深度边缘 + PCA 标签 + 时空权重 + 3D 线`

升级为一个**三阶段混合式标定流水线**，并严格按照以下顺序执行：

1. **BEV 粗初始化**
2. **语义概率场对齐**
3. **多帧迭代精修**

这份文件的设计目标是：你可以直接把它交给 Cursor，让 Cursor 按照这里的步骤逐阶段修改程序。

---

# 全局实施规则

## 规则 1：不要一次性重写整个项目

必须采用**小步、可测试**的方式逐步实现。
每完成一个阶段后，都要：

* 运行对应的单元测试或 smoke test
* 确保在新功能关闭时，旧流水线仍然可运行
* 尽量保持向后兼容

## 规则 2：尽量保持当前项目结构不被破坏

优先采用以下方式：

* **新增文件**
* **新增配置块**
* **新增命令行参数**
  而不是一开始就删除旧逻辑

## 规则 3：所有新功能都必须可通过配置开关控制

当对应配置为 false 时，新模块必须能够被干净地跳过。

## 规则 4：必须保证旧流水线仍可运行

当以下开关都关闭时，旧流程必须仍然可用：

* `bev.enabled = false`
* `semantic_calib.enabled = false`
* `refine.enabled = false`

## 规则 5：OSDaR23 的正确性优先于 KITTI 风格假设

如果旧代码中存在明显的 KITTI 风格假设，例如：

* 固定 64 线 range image
* 与 KITTI 绑定的 road z 阈值
* 只保留水平或垂直 2D 线
  这些逻辑在 OSDaR23 模式下必须被隔离、替换，或通过条件分支禁用。

---

# 最终目标流水线

最终的执行顺序应当变成：

```text
run_pipeline.py
  -> pipeline/runner.py
      -> image_feature_stage.py
      -> lidar_stage.py
      -> bev_stage.py
      -> calib_stage.py
      -> refine_stage.py
      -> visual_stage.py
```

其中：

* `image_feature_stage.py`

  * 生成图像语义概率图
  * 生成语义引导的边缘和线特征
  * 生成由图像语义推导出的 pseudo-BEV

* `lidar_stage.py`

  * 生成 LiDAR 语义点
  * 生成 LiDAR-BEV 特征图
  * 生成改进后的 3D 轨道与竖直结构特征

* `bev_stage.py`

  * 在 BEV 空间做粗标定
  * 主要优化 `yaw + tx + ty`

* `calib_stage.py`

  * 执行语义概率场对齐
  * 将边缘与线约束作为辅助正则项

* `refine_stage.py`

  * 执行滑动窗口的多帧小增量精修

---

# Phase 0 —— 建立安全分支与基线结果

## TODO 0.1

在修改代码前，先新建一个 git 分支。

建议分支名：

* `feature/osdar23-bev-semantic-refine`

## TODO 0.2

在当前代码不做任何修改的前提下，先对一个 OSDaR23 样例帧运行一次当前项目，并保存基线输出，包括：

* 投影结果图
* 当前的 `*_lines_2d.txt`
* 当前的 `*_lines_3d.txt`
* 当前的标定结果文件
* 控制台运行日志

将这些基线结果保存到：

```text
result_osdar23/baseline_before_upgrade/
```

## TODO 0.3

把当前运行命令保存到一个说明文件中：

```text
result_osdar23/baseline_before_upgrade/run_command.txt
```

**验收标准**

* 基线输出存在
* 在任何修改开始之前，当前流水线能正常运行

---

# Phase 1 —— 添加配置结构与执行骨架

## 目标

先把新配置块和新阶段的执行骨架接进去，但暂时不要实现具体算法。

## TODO 1.1 —— 修改 `configs/base.yaml`

新增以下顶层配置块：

```yaml
image_features:
  enable_semantic_probs: true
  enable_edges: true
  enable_lines: true
  semantic_classes: [rail, ballast, pole, signal, platform, building, road, vehicle, vegetation, sky]
  save_logits: true
  rail_class_names: [rail, ballast]
  vertical_structure_classes: [pole, signal]
  bottom_crop_ratio_for_edges: 0.0
  keep_diagonal_lines: true
  restrict_lsd_by_semantics: true

bev:
  enabled: true
  x_range: [0.0, 80.0]
  y_range: [-20.0, 20.0]
  z_range: [-2.0, 5.0]
  resolution: 0.1
  channels: [density, max_height, min_height, mean_height, intensity, verticality, rail_prob]
  optimize_dofs: [yaw, tx, ty]
  rail_prior_weight: 1.0
  pole_prior_weight: 0.6

semantic_calib:
  enabled: true
  pyramid_scales: [1.0, 0.5, 0.25]
  semantic_js_weight: 3.0
  histogram_weight: 0.5
  edge_weight: 1.0
  line_weight: 0.6
  anchor_fixed_sampling: true
  direction_aware_yaw_weighting: true
  class_weights:
    rail: 3.0
    ballast: 2.0
    pole: 2.0
    signal: 2.0
    platform: 1.5
    building: 1.0
    road: 0.8
    vehicle: 1.0
    vegetation: 0.2
    sky: 0.0

refine:
  enabled: true
  window_size: 5
  min_observability_score: 0.35
  max_pose_jump_deg: 0.8
  max_pose_jump_m: 0.15
  temporal_smoothing_lambda: 0.4
  update_stride: 1
```

## TODO 1.2 —— 修改 `configs/osdar23.yaml`

加入或覆盖 OSDaR23 专用配置：

```yaml
dataset:
  reference_plane: rail_top
  reference_z: 0.0

bev:
  x_range: [0.0, 100.0]
  y_range: [-25.0, 25.0]
  z_range: [-1.5, 6.0]
  resolution: 0.05
  use_rail_plane_prior: true
  near_ground_band: [-0.25, 0.35]

image_features:
  bottom_crop_ratio_for_edges: 0.0
  keep_diagonal_lines: true
  restrict_lsd_by_semantics: true

semantic_calib:
  class_weights:
    rail: 4.0
    ballast: 2.0
    pole: 2.0
    signal: 2.5
    platform: 2.0
    building: 1.2
    road: 0.6
    vehicle: 1.0
    vegetation: 0.2
    sky: 0.0
```

## TODO 1.3 —— 修改 `pipeline/context.py`

扩展运行时上下文，使其能够存储：

* `current_pose_init`
* `current_pose_bev`
* `current_pose_semantic`
* `current_pose_refined`
* `frame_bundle_ids`
* 各阶段的输出路径：

  * `image_features`
  * `lidar_features`
  * `bev_init`
  * `calibration`
  * `refinement`

同时添加以下辅助能力：

* 创建各阶段输出目录
* 根据当前帧解析滑动窗口帧 ID
* 如果存在历史精修状态，能够读取它

## TODO 1.4 —— 添加阶段占位文件

新增以下文件：

```text
pipeline/stages/image_feature_stage.py
pipeline/stages/bev_stage.py
pipeline/stages/refine_stage.py
pipeline/refinement_state.py
```

此阶段中，这些文件只需要是最小占位实现，至少包含：

* 一个 `run(context, ...)` 函数
* 基本日志输出
* 输出目录创建
* 暂时不包含重算法实现

## TODO 1.5 —— 修改 `pipeline/runner.py`

将阶段顺序改为：

```python
image_feature_stage.run(...)
lidar_stage.run(...)
bev_stage.run(...)
calib_stage.run(...)
refine_stage.run(...)
visual_stage.run(...)
```

当前阶段要求：

* 当某个阶段在配置中关闭时，必须能够被干净跳过
* 保持向后兼容

**验收标准**

* 添加占位阶段后，流水线仍可运行
* 此时还没有引入新的算法行为
* 旧功能没有被破坏

---

# Phase 2 —— 构建图像特征前端

## 目标

将图像侧从“边缘优先”升级为“语义优先”。

## TODO 2.1 —— 实现 `pipeline/stages/image_feature_stage.py`

该阶段对每帧应输出以下产物：

```text
result_osdar23/image_features/<frame_id>/
  semantic_probs.npy
  semantic_logits.npy
  semantic_argmax.png
  rail_prob.png
  pole_prob.png
  edge_map.png
  edge_weight.png
  line_map.png
  lines_2d.txt
  pseudo_bev.npz
```

职责包括：

1. 读取图像
2. 调用图像特征提取器
3. 保存语义概率图
4. 构造语义引导边缘图
5. 构造语义引导 2D 线特征
6. 构造图像侧 pseudo-BEV 输入

## TODO 2.2 —— 重构 `tools/sam_extractor.py`

### 当前问题

这个文件目前是“边缘为中心”的。
现在要把它改成“语义为中心”的实现。

### 需要新增的函数

```python
def extract_semantic_probabilities(image, config):
    """返回每个类别的概率图。"""

def build_semantic_edge_map(semantic_probs, config):
    """生成突出 rail、pole、platform、building 边界的语义引导边缘图。"""

def extract_lines_2d(image, semantic_probs, edge_map, config):
    """提取带有语义支持的 2D 线，而不是只保留水平或垂直线。"""

def save_image_feature_bundle(output_dir, bundle):
    """统一保存图像侧的全部输出。"""
```

### 必须完成的行为修改

#### A. 去掉“只保留水平或垂直线”的硬限制

替换为：

* 保留 rail-like 的斜线
* 保留竖直结构线
* 保留强结构边界线

#### B. 让 LSD 受到语义支持约束

一条线仅当它与以下区域有足够重叠时才保留：

* `rail_prob`
* `pole_prob`
* `platform/building` 的边界区域

#### C. 底部裁切不能再盲目抑制轨道区域

必须尊重配置项。
在 OSDaR23 中，此值应为 0 或非常小。

#### D. 保留对旧边缘输出的兼容性

例如 `edge_map` 和 `edge_weight` 仍然要正常生成。

## TODO 2.3 —— 按需修改 `tools/run_sam.py`

如果当前入口脚本只支持边缘输出，则需要修改它，使其能够：

* 支持语义概率图输出
* 把相关配置项正确传进去

尽量保持旧 CLI 参数不失效。

## TODO 2.4 —— 新增 `tools/semantic_to_bev.py`

新增一个工具，用于把图像语义概率图转换成 pseudo-BEV。

建议函数接口：

```python
def semantic_probs_to_pseudo_bev(
    semantic_probs,
    intrinsics,
    extrinsics,
    bev_config,
    dataset_meta,
):
    ...
```

### 行为要求

* 使用当前位姿估计
* 把选定类别投影到轨顶面或地面参考平面
* 输出以下类别的 BEV 特征图：

  * rail
  * pole 和 signal
  * platform 和 building
  * road

### 输出

保存：

```text
pseudo_bev.npz
```

**验收标准**

* 图像特征阶段可运行
* `semantic_probs.npy` 已生成
* `lines_2d.txt` 已生成，且包含语义支持字段
* `pseudo_bev.npz` 已生成

---

# Phase 3 —— 升级 LiDAR 特征提取

## 目标

增加 LiDAR 语义点、LiDAR-BEV，以及更稳定的轨道与竖直结构提取能力。

## TODO 3.1 —— 修改 `pipeline/stages/lidar_stage.py`

该阶段需要能够请求并保存以下输出：

* `semantic_points.txt`
* `bev_maps.npz`
* 改进后的 `lines_3d.txt`
* 改进后的 `edge_points.txt`

调用 C++ 提取器时新增命令行参数：

* `--save_semantic_points`
* `--save_bev_maps`
* `--bev_xmin`
* `--bev_xmax`
* `--bev_ymin`
* `--bev_ymax`
* `--bev_resolution`
* `--reference_plane_z`
* `--rail_band_zmin`
* `--rail_band_zmax`

## TODO 3.2 —— 新增 `cpp/include/bev_builder.h`

定义：

* `BEVGridSpec`
* `BEVChannels`

建议的通道包括：

* density
* max height
* min height
* mean height
* intensity
* verticality
* rail probability

## TODO 3.3 —— 新增 `cpp/bev_builder.cpp`

实现从 LiDAR 点云生成 BEV 特征图。

建议接口：

```cpp
bool BuildLidarBEV(
    const pcl::PointCloud<PointT>::Ptr& cloud,
    const BEVGridSpec& spec,
    BEVChannels* out);
```

### 必须实现的逻辑

* 在 OSDaR23 参考系中对点云进行栅格化
* 计算密度和高度统计量
* 计算 verticality 和 linearity 等结构特征
* 生成近地面的 rail-like 响应图
* 将这些图导出保存

## TODO 3.4 —— 新增 `cpp/include/rail_bev_extractor.h`

## TODO 3.5 —— 新增 `cpp/rail_bev_extractor.cpp`

实现基于 BEV 的轨道结构提取。

建议接口：

```cpp
struct RailBEVResult {
    std::vector<cv::Point2f> centerline_pts;
    std::vector<cv::Point2f> left_rail_pts;
    std::vector<cv::Point2f> right_rail_pts;
    float confidence;
};

RailBEVResult ExtractRailStructureFromBEV(
    const BEVChannels& bev,
    const RailPriorConfig& cfg);
```

### 必须满足的要求

* 不能再简单地只拟合两条全局直线
* 必须支持：

  * 平行轨道
  * 道岔或分叉区域
  * 弯曲或分段结构
* 先从近地面的 rail-like 区域出发
* 从 BEV 结果反推出结构化的 3D 轨道线

## TODO 3.6 —— 新增 `cpp/include/vertical_structure_extractor.h`

## TODO 3.7 —— 新增 `cpp/vertical_structure_extractor.cpp`

用“候选筛选 + 聚类 + 局部拟合”替代脆弱的全局 pole RANSAC。

建议接口：

```cpp
std::vector<Line3D> ExtractVerticalStructures(
    const pcl::PointCloud<PointT>::Ptr& cloud,
    const SemanticLabels& labels,
    const VerticalStructureConfig& cfg);
```

### 必须满足的要求

* 先做语义和几何候选筛选
* 先聚类，再在每个簇内局部拟合
* 不允许再对整片 elevated points 直接做全局 RANSAC

## TODO 3.8 —— 重构 `cpp/lidar_extractor.cpp`

### 去除 OSDaR23 对固定 64x1024 range-image 假设的依赖

旧逻辑只保留在兼容模式下。

### 新的流程应为

1. 读取点云
2. 预处理或融合
3. 估计法向与结构度量
4. 构建语义点标签
5. 构建 LiDAR-BEV
6. 从 BEV 中提取轨道结构
7. 从聚类中提取竖直结构
8. 保存输出结果

### 重要要求

如果旧函数仍需保留，则必须通过以下条件隔离：

* `dataset_format == "kitti"` 或
* `use_legacy_range_image_mode`

## TODO 3.9 —— 修改 `cpp/include/common.h`

新增以下数据结构：

* semantic point
* 扩展 2D 线结构
* 扩展 3D 线结构
* score breakdown
* pose delta

## TODO 3.10 —— 修改 `CMakeLists.txt`

确保这些新文件被正确编译到以下目标中：

* `lidar_extractor`
* 未来的 `bev_initializer`
* 需要时也可被 `optimizer` 复用

**验收标准**

* LiDAR 阶段可以生成 `bev_maps.npz`
* 轨道提取不再只依赖全局直线 RANSAC
* 竖直结构提取改为基于聚类的方式

---

# Phase 4 —— 实现 BEV 粗初始化

## 目标

在语义精配准之前，新增一个基于 BEV 的粗位姿初始化模块。

## TODO 4.1 —— 实现 `pipeline/stages/bev_stage.py`

输入：

* 图像 pseudo-BEV
* LiDAR-BEV 特征图
* 初始位姿

输出：

```text
result_osdar23/bev_init/<frame_id>/
  pose_delta.txt
  pose_after_bev.txt
  debug_bev_overlay.png
  debug_bev_match.png
  debug_bev_score.json
```

### 行为要求

* 若 `bev.enabled == false`，则跳过该阶段
* 否则调用新的 C++ BEV 初始化器可执行文件
* 把输出位姿保存到 context 中的 `current_pose_bev`

## TODO 4.2 —— 新增 `cpp/include/bev_matcher.h`

## TODO 4.3 —— 新增 `cpp/bev_matcher.cpp`

实现以下两者之间的粗匹配：

* LiDAR-BEV
* 图像 pseudo-BEV

建议接口：

```cpp
PoseDelta EstimateBEVDelta(
    const BEVChannels& lidar_bev,
    const ImageBEV& image_bev,
    const Pose& init_pose,
    const BEVOptimizeConfig& cfg,
    ScoreBreakdown* debug_score);
```

### 推荐的优化范围

只优化：

* yaw
* tx
* ty

将以下量留到后续阶段处理：

* roll
* pitch
* z

### 推荐使用的匹配线索

* rail BEV 响应对齐
* vertical structure 热点对齐
* platform 和 building 边界一致性

## TODO 4.4 —— 新增 `cpp/bev_initializer.cpp`

这个文件应编译成一个独立可执行程序：

```text
build/bev_initializer
```

职责包括：

* 读取图像 pseudo-BEV
* 读取 LiDAR-BEV
* 读取初始位姿
* 估计 `ΔT_bev`
* 保存结果

## TODO 4.5 —— 修改 `CMakeLists.txt`

新增 target：

```cmake
add_executable(bev_initializer
    cpp/bev_initializer.cpp
    cpp/bev_builder.cpp
    cpp/bev_matcher.cpp
    cpp/io_utils.cpp
)
```

**验收标准**

* BEV 初始化器可以独立运行
* 位姿增量结果被正确写出
* 流水线能够读取并使用 `current_pose_bev`

---

# Phase 5 —— 将语义概率场对齐加入优化器

## 目标

让语义概率场对齐成为精配准阶段的主目标函数。

## TODO 5.1 —— 修改 `pipeline/stages/calib_stage.py`

该阶段现在的输入必须包括：

* 若存在则使用来自 BEV 阶段的位姿
* 图像语义概率图
* LiDAR 语义点
* 边缘图
* 边缘权重图
* 2D 线
* 3D 线

传给优化器的新命令行参数应包括：

* `--semantic_probs`
* `--lidar_semantic_points`
* `--init_pose_from_bev`
* `--semantic_js_weight`
* `--histogram_weight`
* `--class_weights`
* `--pyramid_scales`

并在本阶段结束后设置：

* `current_pose_semantic`

## TODO 5.2 —— 修改 `cpp/main.cpp`

扩展优化器的命令行解析，支持以下输入：

* 语义概率图路径
* LiDAR 语义点路径
* 来自 BEV 阶段的初始位姿
* 语义损失相关参数
* 优化模式（`full_calib` 或 `refine_only`）

## TODO 5.3 —— 修改 `cpp/optimizer_data_loader.cpp`

新增以下数据加载能力：

* 语义概率图
* 若需要则加载语义 logits
* 语义点
* BEV 阶段输出的初始位姿

建议新增函数：

```cpp
bool LoadSemanticProbabilityMaps(...);
bool LoadSemanticPoints(...);
bool LoadInitPoseFromBEV(...);
```

## TODO 5.4 —— 新增 `cpp/include/optimizer_semantic_scoring.h`

## TODO 5.5 —— 新增 `cpp/optimizer_semantic_scoring.cpp`

实现以下语义评分项：

1. 多尺度语义分布差异
2. 类别直方图一致性
3. 可选的 anchor-fixed sampling

建议接口：

```cpp
double ComputeSemanticJSDivergence(
    const ProjectedSemanticMaps& lidar_proj,
    const SemanticProbMaps& image_probs,
    const SemanticScoringConfig& cfg,
    ScoreBreakdown* breakdown);
```

### 必须满足的要求

* 比较的是 LiDAR 投影语义分布与图像语义概率图
* 不能只比较硬 argmax mask
* 必须通过类别权重处理类别不平衡
* 必须输出 score breakdown 便于调试

## TODO 5.6 —— 修改 `cpp/optimizer_scoring.cpp`

将旧的总分逻辑替换为类似下面的形式：

```text
total_score =
    w_semantic_js   * semantic_js_score
  + w_semantic_hist * semantic_hist_score
  + w_edge          * edge_dist_score
  + w_line          * line_align_score
```

### 必须遵守的原则

语义项必须是主导项。
边缘项和线项仍然保留，但只作为正则项。

## TODO 5.7 —— 修改 `cpp/edge_calibrator.cpp`

将标定流程从：

* coarse search
* fine optimization

重构为：

1. 如果存在，则先应用 BEV 初值
2. 执行 semantic coarse optimization
3. 执行 semantic fine optimization
4. 执行 geometric regularized refinement

建议新增的高层方法：

```cpp
ApplyPoseFromBEV(...)
PerformSemanticCoarseOptimization(...)
PerformSemanticFineOptimization(...)
PerformGeometricRegularizedRefinement(...)
```

**验收标准**

* 优化器可以在带有语义输入时运行
* score breakdown 中包含语义项
* 语义项能真正影响最终位姿结果

---

# Phase 6 —— 增加多帧滑动窗口精修

## 目标

利用时间连续性稳定输出，减少单帧失败的影响。

## TODO 6.1 —— 实现 `pipeline/refinement_state.py`

该文件需要管理以下状态：

* 最近若干帧 ID
* 最近若干帧位姿
* 最近若干帧 score breakdown
* 可观测性分数
* anchor pose
* 最近一次成功更新的时间或帧号

建议提供的方法：

```python
load_state(...)
save_state(...)
append_frame_result(...)
get_active_window(...)
should_update(...)
```

状态持久化到：

```text
result_osdar23/refinement/state.json
```

## TODO 6.2 —— 实现 `pipeline/stages/refine_stage.py`

输入：

* 最近若干帧的 `current_pose_semantic`
* 最近若干帧的 score breakdown
* 最近若干帧的特征质量与可观测性
* 上一次 refined pose

输出：

* `current_pose_refined`
* 精修调试文件

建议输出：

```text
result_osdar23/refinement/
  state.json
  <frame_id>_window_pose.txt
  <frame_id>_window_debug.json
  <frame_id>_trajectory.png
```

### 必须实现的逻辑

* 使用大小为 `refine.window_size` 的滑动窗口
* 只有在可观测性足够时才执行更新
* 用配置中的阈值限制单步位姿跳变
* 应用时间平滑

## TODO 6.3 —— 增加可观测性打分

可观测性分数应综合以下因素：

* rail structure confidence
* vertical structure confidence
* semantic map confidence
* edge support quality
* 当前帧中可见的重要类别数量

这部分先在 Python 中实现，便于快速迭代。

## TODO 6.4 —— 修改 `cpp/edge_calibrator.cpp`

支持一个更小搜索范围的 refinement 模式。

新增模式：

```cpp
enum class OptimizeMode {
    FULL_CALIB,
    REFINE_ONLY
};
```

在 `REFINE_ONLY` 模式下：

* 限制旋转和平移增量
* 使用当前 refined pose 作为 anchor
* 优先利用稳定类别，如 rail、pole、platform、building 边界

## TODO 6.5 —— 修改 `cpp/main.cpp`

新增以下命令行参数：

* `--mode refine_only`
* `--max_delta_deg`
* `--max_delta_m`
* `--window_index`
* `--anchor_pose`

**验收标准**

* refinement 状态能跨帧保存
* refined pose 比单帧 pose 更平滑
* 大的帧间跳变被有效抑制

---

# Phase 7 —— 升级可视化与诊断

## 目标

让新流水线具备可调试性。

## TODO 7.1 —— 修改 `tools/visualize.py`

新增三类调试图支持：

### A. BEV 调试图

* LiDAR BEV
* 图像 pseudo-BEV
* 两者叠加图

### B. 语义对齐调试图

* 图像语义 argmax
* LiDAR 投影语义图
* 语义不一致热图

### C. 精修调试图

* 位姿轨迹曲线
* score breakdown 曲线
* 可观测性分数曲线

## TODO 7.2 —— 在 `experiments/` 下新增诊断脚本

新增：

```text
experiments/check_bev_alignment.py
experiments/check_semantic_alignment.py
experiments/check_refine_window.py
```

### 用途

* 验证 BEV 初始化器的行为
* 验证语义对齐效果
* 验证时间窗口精修的稳定性

## TODO 7.3 —— 统一保存 score breakdown JSON

对 BEV、语义标定和精修阶段，都保存统一结构的 JSON 调试文件，其中应包含：

* 输入 pose
* 输出 pose
* 子项分数 breakdown
* 运行时间
* confidence 或 observability

**验收标准**

* 每个阶段都有可检查的调试产物
* 一旦失败，能够定位是哪个阶段出了问题

---

# Phase 8 —— 测试与上线验证

## 目标

补充足够的测试，确保后续改动不容易破坏新流水线。

## TODO 8.1 —— 增加或更新配置加载测试

确保新配置字段能被正确加载。

可能需要修改的文件：

* `tests/test_config_loading.py`

## TODO 8.2 —— 增加或更新 OSDaR23 数据集处理测试

验证：

* calibration 解析仍然正确
* OSDaR23 的参考平面假设被保留
* pseudo-BEV 阶段能拿到正确的内参与外参

可能需要修改的文件：

* `tests/test_osdar23_dataset.py`

## TODO 8.3 —— 为新阶段增加 smoke test

为以下模块增加最小可运行测试：

* image feature stage
* BEV stage
* semantic calibration mode
* refine stage

建议新增测试文件：

```text
tests/test_image_feature_stage.py
tests/test_bev_stage.py
tests/test_semantic_calib_stage.py
tests/test_refine_stage.py
```

## TODO 8.4 —— 增加一个端到端 OSDaR23 smoke test

先运行一帧，配置为：

* BEV enabled
* semantic calibration enabled
* refine disabled

再运行一个小窗口，配置为：

* refine enabled

---

# 实施顺序总览（严格执行顺序）

必须严格按照以下顺序实施：

## Step 1

备份基线结果并创建分支。

## Step 2

更新配置：

* `configs/base.yaml`
* `configs/osdar23.yaml`

## Step 3

搭建执行骨架：

* `pipeline/context.py`
* `pipeline/runner.py`
* 创建占位阶段文件

## Step 4

完成图像特征前端：

* `pipeline/stages/image_feature_stage.py`
* `tools/sam_extractor.py`
* `tools/run_sam.py`
* `tools/semantic_to_bev.py`

## Step 5

升级 LiDAR 特征：

* `pipeline/stages/lidar_stage.py`
* `cpp/include/bev_builder.h`
* `cpp/bev_builder.cpp`
* `cpp/include/rail_bev_extractor.h`
* `cpp/rail_bev_extractor.cpp`
* `cpp/include/vertical_structure_extractor.h`
* `cpp/vertical_structure_extractor.cpp`
* `cpp/lidar_extractor.cpp`
* `cpp/include/common.h`

## Step 6

实现 BEV 粗初始化：

* `pipeline/stages/bev_stage.py`
* `cpp/include/bev_matcher.h`
* `cpp/bev_matcher.cpp`
* `cpp/bev_initializer.cpp`
* `CMakeLists.txt`

## Step 7

实现语义对齐优化器：

* `pipeline/stages/calib_stage.py`
* `cpp/main.cpp`
* `cpp/optimizer_data_loader.cpp`
* `cpp/include/optimizer_semantic_scoring.h`
* `cpp/optimizer_semantic_scoring.cpp`
* `cpp/optimizer_scoring.cpp`
* `cpp/edge_calibrator.cpp`

## Step 8

实现多帧精修：

* `pipeline/refinement_state.py`
* `pipeline/stages/refine_stage.py`
* `cpp/main.cpp`
* `cpp/edge_calibrator.cpp`

## Step 9

升级可视化与诊断：

* `tools/visualize.py`
* `experiments/check_bev_alignment.py`
* `experiments/check_semantic_alignment.py`
* `experiments/check_refine_window.py`

## Step 10

补充测试：

* 更新或新增 `tests/` 下的测试文件

---

# 给 Cursor 的执行说明

在执行这份 TODO 时，必须遵守以下要求：

1. 一次只完成一个 Phase
2. 每完成一个 Phase 后，都要：

   * 展示修改过的文件
   * 总结本阶段完成了什么
   * 运行对应的测试或 smoke 命令
3. 不允许静默删除旧逻辑
4. 所有新行为都必须可以通过 feature flag 开关控制
5. 优先新增辅助类或辅助函数，而不是做超大规模单体式重写

---

# 最终预期交付结果

当全部 TODO 完成后，项目应支持：

* 面向 OSDaR23 的 **BEV 粗位姿初始化**
* **语义概率场驱动的精配准**
* **多帧滑动窗口精修**
* 更适合铁路场景的轨道与竖直结构提取
* 更好的诊断与可视化能力，便于问题定位

---

# 最小里程碑检查点

## Milestone A

在所有新阶段关闭时，旧流水线仍然可运行。

## Milestone B

图像阶段可以产出语义概率图与 pseudo-BEV。

## Milestone C

LiDAR 阶段可以产出 LiDAR-BEV，以及改进后的轨道与竖直结构特征。

## Milestone D

BEV 阶段能够输出一个合理的 `ΔT_bev`。

## Milestone E

语义优化器中，语义分数成为主导项。

## Milestone F

精修阶段能让多帧位姿更加平滑。

## Milestone G

可视化中能看到 BEV 叠加图、语义叠加图以及时间曲线。

---

# TODO 结束

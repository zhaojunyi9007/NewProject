# OSDaR23 项目下一步修改计划 TODO（供 Cursor 直接执行）

## 文档目标

本文档用于指导 Cursor 对当前项目进行**下一阶段的重点修改**。
目标不是继续扩展新模块，而是优先把已经搭起来但尚未闭环的主链补齐，使项目真正形成：

**BEV 粗初始化 → 语义概率场精配准 → 多帧滑动窗口精修**

当前代码的实际状态可以概括为：

* **Phase 1（配置与骨架）**：基本完成
* **Phase 2（图像语义前端）**：大体完成
* **Phase 3（LiDAR 特征升级）**：部分完成，但存在明显缺口
* **Phase 4（BEV 粗初始化）**：基本完成，但实现简化
* **Phase 5（语义概率场精配准）**：核心未完成，是当前最大缺口
* **Phase 6（多帧精修）**：基础版已实现，但还没真正吃到语义评分
* **Phase 7（可视化与诊断）**：基础版已实现
* **Phase 8（测试）**：明显不足

因此，下一步的核心策略是：

1. **先修工程阻塞项**
2. **优先补完 Phase 5（语义概率场精配准）**
3. **补强 Phase 3（LiDAR 语义与结构特征）**
4. **让 Phase 6（多帧精修）真正建立在语义分数上**
5. **最后补测试**

---

# 全局执行规则

## 规则 1：不要一次性做大重构

每次只完成一个 Phase 或一个明确的小步骤。
每完成一步后：

* 展示修改过的文件
* 总结本步修改内容
* 运行对应的 smoke test 或最小验证命令

## 规则 2：优先补齐主链，不要继续发散扩展

当前最重要的是把：

**BEV 初值 → 语义精配准 → 多帧精修**

这条链路真正打通，而不是增加更多新模块。

## 规则 3：保留旧逻辑，但逐步降级其地位

例如：

* 旧 edge / mask 驱动逻辑可以暂时保留
* 但必须逐步退居辅助项
* 新的语义项必须成为主导项

## 规则 4：所有新改动都应尽量保持配置可控

如果增加新功能，应优先通过配置或参数控制，而不是直接破坏旧流程。

## 规则 5：OSDaR23 的参考系优先于 KITTI 风格假设

如果存在以下旧假设：

* 固定 64 线 range-image
* 与 KITTI 绑定的地面 z 阈值
* 只保留水平/垂直 2D 线

则在 OSDaR23 模式下必须被替换、隔离或配置化。

---

# 总体执行顺序（严格按照下面顺序进行）

## Step 1

修工程阻塞项，保证工程能正常编译和跑通。

## Step 2

统一路径和输出格式，保证各阶段之间真正对接。

## Step 3

补建语义评分模块，让优化器真正具备语义概率场评分能力。

## Step 4

打通语义数据通路，让 `calib_stage` 和 `optimizer` 真正使用语义输入。

## Step 5

改写优化目标函数和优化流程，让语义项成为主导项。

## Step 6

补强 LiDAR 语义和结构特征，让上游输入真正有意义。

## Step 7

让多帧精修真正使用语义 breakdown，而不是只挂着接口。

## Step 8

最后补 smoke tests 和端到端测试。

---

# Phase A —— 修复工程阻塞项（最高优先级）

## 目标

先让工程达到：

* 能编译
* 能运行
* 能正确读写新阶段产物

---

## TODO A1 —— 修复 `CMakeLists.txt` 中缺失源文件问题

### 涉及文件

* `CMakeLists.txt`

### 任务

检查 `optimizer` target 中是否引用了不存在的：

* `cpp/optimizer_semantic_scoring.cpp`

### 处理要求

有两种可接受方案：

#### 方案 A（推荐）

直接补建以下文件：

* `cpp/include/optimizer_semantic_scoring.h`
* `cpp/optimizer_semantic_scoring.cpp`

#### 方案 B

如果当前阶段暂不实现，则先从 `CMakeLists.txt` 中移除该文件引用，等后续再加回。

### 验收标准

* `cmake` 配置成功
* `optimizer` 可编译
* `lidar_extractor` 可编译
* `bev_initializer` 可编译

---

## TODO A2 —— 统一图像特征目录读取路径

### 涉及文件

* `pipeline/stages/image_feature_stage.py`
* `pipeline/observability.py`
* `pipeline/stages/calib_stage.py`
* `pipeline/stages/visual_stage.py`
* 任何仍然依赖旧 `sam_dir` 的文件

### 任务

全项目统一使用：

```text
result_osdar23/image_features/<frame_id>/
```

作为图像语义与图像特征的唯一来源。

### 必须统一的文件输出包括

* `semantic_probs.npy`
* `semantic_argmax.png`
* `edge_map.png`
* `edge_weight.png`
* `lines_2d.txt`
* `pseudo_bev.npz`

### 明确要求

* 不允许 `observability.py` 再去旧目录里查 `semantic_probs.npy`
* 不允许 `calib_stage.py` 同时混用新旧目录
* 如有兼容逻辑，必须写清楚 fallback 顺序

### 验收标准

* `observability.py` 能正确读取新目录下的 `semantic_probs.npy`
* `calib_stage.py` 只从新目录读取图像侧输入
* 运行日志中不再出现旧 `sam_dir` 路径依赖

---

## TODO A3 —— 统一标定结果输出格式

### 涉及文件

* `cpp/edge_calibrator.cpp`
* `cpp/main.cpp`
* `pipeline/stages/calib_stage.py`
* `pipeline/stages/refine_stage.py`

### 任务

当前下游代码已经假设结果文件中会有：

* `semantic_js_divergence`
* `semantic_hist_similarity`
* `edge_term_norm`
* `line_term_norm`

但上游优化器并没有真正输出这些内容。现在要统一输出格式。

### 建议的结果文件格式

```text
r: ...
t: ...
Score: ...
semantic_js_divergence: ...
semantic_hist_similarity: ...
edge_term_norm: ...
line_term_norm: ...
rail_confidence: ...
vertical_structure_confidence: ...
```

### 要求

* 输出格式必须稳定
* 下游解析逻辑必须和上游写入逻辑完全一致
* 不能再靠“字段不存在时跳过”的隐式容错维持运行

### 验收标准

* `calib_stage.py` 能稳定解析结果文件
* `refine_stage.py` 能读取 breakdown 字段
* debug 文件中能看到这些字段

---

# Phase B —— 补完 Phase 5，让语义概率场精配准真正落地

## 目标

这是当前项目最重要的缺口。
必须让优化器真正从“旧版 edge / mask 驱动”升级为“语义概率场驱动”。

---

## TODO B1 —— 新建真正的语义评分模块

### 涉及文件

* 新增 `cpp/include/optimizer_semantic_scoring.h`
* 新增 `cpp/optimizer_semantic_scoring.cpp`

### 任务

实现一个**最小可用版**语义评分模块，至少完成：

1. 将 `LiDAR semantic points` 投影到图像平面
2. 从 `semantic_probs.npy` 中读取图像语义概率图
3. 计算基础版语义一致性指标：

   * `semantic_js_divergence`
   * `semantic_hist_similarity`

### 第一版要求

* 不要求一开始就做复杂的 `anchor-fixed sampling`
* 先保证语义概率图真的参与评分
* 接口必须可扩展，便于后续增加：

  * 多尺度评分
  * anchor 机制
  * 类别权重

### 建议新增函数

```cpp
double ComputeSemanticJSDivergence(...);
double ComputeSemanticHistogramConsistency(...);
```

### 验收标准

* 新模块可编译
* 优化器能调用语义评分函数
* 返回的语义评分不是占位常数，而是与输入有关

---

## TODO B2 —— 扩展 `main.cpp` 的优化器入口参数

### 涉及文件

* `cpp/main.cpp`

### 任务

为 `optimizer` 增加以下 CLI 参数解析能力：

* `--semantic_probs`
* `--lidar_semantic_points`
* `--init_pose_from_bev`
* `--semantic_js_weight`
* `--histogram_weight`
* `--class_weights`
* `--pyramid_scales`
* `--mode`
* `--max_delta_deg`
* `--max_delta_m`

### 要求

* 保持旧参数兼容
* 新参数优先服务于 OSDaR23 新流程
* 参数要能进入内部配置结构，而不是只解析后丢弃

### 验收标准

* `optimizer --help` 中出现新参数
* 用新参数调用时程序不报错
* 日志中能看到这些参数被正确接收

---

## TODO B3 —— 扩展 `optimizer_data_loader.cpp`

### 涉及文件

* `cpp/optimizer_data_loader.cpp`

### 任务

新增以下加载能力：

* 加载 `semantic_probs.npy`
* 加载 `semantic_points.txt`
* 加载 `pose_after_bev.txt`
* 加载类别权重
* 加载多尺度配置

### 建议新增函数

```cpp
bool LoadSemanticProbabilityMaps(...);
bool LoadSemanticPoints(...);
bool LoadInitPoseFromBEV(...);
```

### 要求

* 不再只加载旧的 `_semantic_map.png`
* 新版 loader 必须真正支持概率图输入

### 验收标准

* `semantic_probs.npy` 能被读入
* `semantic_points.txt` 能被读入
* `pose_after_bev.txt` 能被解析
* 相关数据能传递到优化器主流程

---

## TODO B4 —— 改写 `optimizer_scoring.cpp`，让语义项成为主导项

### 涉及文件

* `cpp/optimizer_scoring.cpp`

### 任务

将总目标函数调整为：

```text
total_score =
    w_semantic_js   * semantic_js_score
  + w_semantic_hist * semantic_hist_score
  + w_edge          * edge_dist_score
  + w_line          * line_align_score
```

### 要求

* `semantic_js_score` 与 `semantic_hist_score` 必须成为主导项
* `edge_dist_score` 和 `line_align_score` 只能作为辅助正则
* 不能再让旧版 mask/edge 逻辑主导最终位姿

### 验收标准

* score breakdown 中明确显示语义项
* 改变 `semantic_js_weight` 会显著影响总分
* 禁用语义项时结果与启用语义项时明显不同

---

## TODO B5 —— 重构 `edge_calibrator.cpp`

### 涉及文件

* `cpp/edge_calibrator.cpp`
* 如有必要，同步修改 `cpp/include/edge_calibrator.h`

### 任务

将当前旧式流程：

* `PerformCoarseSearch()`
* `PerformFineOptimization()`

重构为更符合新主链的流程：

1. `ApplyPoseFromBEV()`
2. `PerformSemanticCoarseOptimization()`
3. `PerformSemanticFineOptimization()`
4. `PerformGeometricRegularizedRefinement()`

### 第一版要求

* 先重构流程框架
* 不要求每个内部优化阶段一开始就做到很复杂
* 但 `T_bev` 必须真正成为初始位姿，而不是只保存在文件里

### 验收标准

* `T_bev` 真正参与优化初始化
* semantic coarse / fine 被实际调用
* 结果文件中包含语义相关 breakdown

---

## TODO B6 —— 修改 `pipeline/stages/calib_stage.py`

### 涉及文件

* `pipeline/stages/calib_stage.py`

### 任务

确保该阶段真正向优化器传递以下输入：

* `current_pose_bev`
* `semantic_probs.npy`
* `semantic_points.txt`
* `edge_map.png`
* `edge_weight.png`
* `lines_2d.txt`
* `lines_3d.txt`

并确保该阶段输出：

* `current_pose_semantic`
* 语义 breakdown
* 正确的结果文件路径

### 验收标准

* `calib_stage.py` 的参数与 `main.cpp` 保持一致
* 本阶段不再只是“调用旧 optimizer”
* `current_pose_semantic` 被成功更新

---

# Phase C —— 补强 LiDAR 语义与结构特征，让上游输入真正有意义

## 目标

当前 Phase 3 已有实现，但语义粒度过粗、铁路适配不足。
这一阶段的任务是让 LiDAR 语义和结构特征真正能支撑语义精配准。

---

## TODO C1 —— 细化 `semantic_points.txt` 的标签体系

### 涉及文件

* `cpp/lidar_extractor.cpp`
* `cpp/include/common.h`

### 任务

将当前过粗的标签体系，从类似：

* `ROAD`
* `VEGETATION`
* `STRUCTURE`

升级为更适合铁路场景的简化标签集，例如：

* `rail_like`
* `ballast_ground`
* `vertical_structure`
* `platform_or_building`
* `vehicle_like`
* `vegetation_like`
* `unknown`

### 要求

* 第一版不要求完整深度学习点云语义分割
* 但必须让 `semantic_points.txt` 成为真正有用的语义输入，而不是粗糙占位信息

### 验收标准

* `semantic_points.txt` 中不再只有粗粒度 3 类
* rail / vertical structure / ground 能被区分
* 后续语义投影能够利用这些类别

---

## TODO C2 —— 让 `vertical_structure_extractor.cpp` 真正使用语义筛候选

### 涉及文件

* `cpp/vertical_structure_extractor.cpp`
* `cpp/lidar_extractor.cpp`

### 任务

当前虽然用了聚类，但调用时传的是 `-1`，等于未使用语义过滤。
现在需要改成：

* 优先从 `vertical_structure` 类中筛候选
* 再做聚类
* 再做局部拟合

### 要求

* 不再只靠高度区间筛选
* 必须把 LiDAR 语义真正接入竖直结构提取流程

### 验收标准

* `ExtractVerticalStructures(..., structure_label_id)` 被真实使用
* 相比当前版本，竖直结构误检明显减少
* 输出的竖直结构更稳定

---

## TODO C3 —— 清理 `edge_points.txt` 中不适合 OSDaR23 的高度阈值残留

### 涉及文件

* `cpp/lidar_extractor.cpp`

### 任务

把仍然写死的高度阈值，例如：

* `p.z <= -1.2`
* `p.z > -1.4`

替换为配置驱动逻辑，例如：

* `reference_z`
* `near_ground_band`
* `rail_band_zmin / rail_band_zmax`

### 要求

* OSDaR23 模式下不能继续依赖 KITTI 风格硬编码阈值
* 必须基于 OSDaR23 的轨顶面参考系 `Z=0`

### 验收标准

* `edge_points.txt` 的生成不再写死 `-1.2/-1.4`
* 改为由配置和参考平面控制
* 生成的边缘点更贴近轨道、站台、结构边界

---

## TODO C4 —— 补强 `rail_bev_extractor.cpp`

### 涉及文件

* `cpp/rail_bev_extractor.cpp`

### 任务

在现有实现基础上，再增加两项能力：

1. 支持**分段轨道中心线**
2. 支持**简单分叉区检测**

### 目标

* 不一定一开始就上 spline
* 先做 piecewise line / 分段中心线表达
* 提高对道岔与分叉场景的适应性

### 验收标准

* 道岔区域不再强制退化为单一方向主轴
* 输出结果可区分主轨与分叉趋势
* `rail_confidence` 的定义更合理

---

## TODO C5 —— 完善 `common.h` 的扩展数据结构

### 涉及文件

* `cpp/include/common.h`

### 任务

补齐以下结构扩展：

* 扩展 2D 线结构（包含 `semantic_support`、`class_id`、`confidence`）
* 扩展 3D 线结构（包含 `class_id`、`confidence`）
* 统一 `ScoreBreakdown`
* 统一 `PoseDelta`

### 验收标准

* 2D/3D 线特征在 C++ 中有明确结构表达
* 优化器能直接使用这些字段
* 调试时不再依赖零散数组或临时字段

---

# Phase D —— 让多帧精修真正建立在语义 breakdown 上

## 目标

当前多帧精修框架已经有了，但还没有真正建立在语义对齐分数之上。
现在要让它真正用上语义结果。

---

## TODO D1 —— 修改 `pipeline/observability.py`

### 涉及文件

* `pipeline/observability.py`

### 任务

让可观测性分数真正由以下因素组成：

* `rail_confidence`
* `vertical_structure_confidence`
* `semantic_js_divergence`
* `semantic_hist_similarity`
* 图像语义质量
* 结构边缘质量

### 建议拆成四个子项

* 轨道结构可观测性
* 竖直结构可观测性
* 图像语义质量
* 语义-几何一致性

再融合为总分。

### 验收标准

* `observability.py` 不再只是旧式启发式打分
* observability 对语义优化质量有真实响应
* debug 输出能看到子项分数

---

## TODO D2 —— 修改 `pipeline/stages/refine_stage.py`

### 涉及文件

* `pipeline/stages/refine_stage.py`

### 任务

让多帧精修真正读取并利用：

* `semantic_js_divergence`
* `semantic_hist_similarity`
* `rail_confidence`
* `vertical_structure_confidence`

而不再仅依赖 pose 和基础 score。

### 要求

* 每帧在滑动窗口中的贡献与 observability 和 semantic breakdown 有关
* refine 决策必须可解释

### 验收标准

* 精修阶段能输出每帧权重或贡献度
* 低 observability 帧不会强行更新 pose
* refined pose 比单帧语义标定更平滑

---

## TODO D3 —— 最后再启用 `optimizer --mode refine_only`

### 涉及文件

* `cpp/main.cpp`
* `cpp/edge_calibrator.cpp`
* `pipeline/stages/refine_stage.py`

### 任务

当前只有在 Phase B 和 Phase C 稳定之后，才启用：

* `optimizer --mode refine_only`

### 要求

在 `REFINE_ONLY` 模式下：

* 只允许小增量搜索
* 使用当前 refined pose 作为 anchor
* 更偏向稳定类别约束

### 验收标准

* refinement 真正调用优化器做小步修正
* 每步增量满足 `max_delta_deg` / `max_delta_m`
* 精修结果更稳，不引入抖动

---

# Phase E —— 最后补测试，建立最小闭环验证

## 目标

在主链闭环后，补足最小测试，避免后续继续改坏。

---

## TODO E1 —— 增加基础 smoke tests

### 涉及文件

新增：

* `tests/test_bev_stage.py`
* `tests/test_semantic_calib_stage.py`
* `tests/test_refine_stage.py`

### 任务

先做最小可运行验证：

* 阶段能被调用
* 输入输出文件存在
* 关键字段可读到

### 验收标准

* 三个新阶段至少都有 smoke test
* 本地或 CI 中最小样例可稳定通过

---

## TODO E2 —— 增加一个端到端小样本测试

### 涉及文件

可新增：

* `tests/test_osdar23_end_to_end.py`

### 任务

至少覆盖两种模式：

#### 模式 1：单帧

* `BEV + semantic calib`

#### 模式 2：小窗口

* `BEV + semantic calib + refine`

### 验收标准

* 单帧跑通
* 多帧跑通
* pose 文件、debug 文件、score breakdown 文件都存在

---

# 给 Cursor 的执行要求

在执行本 TODO 时，必须严格遵守以下要求：

1. **一次只完成一个 Phase 或一个子步骤**
2. 每完成一步后，必须输出：

   * 修改过的文件列表
   * 本步修改内容摘要
   * 已执行的测试或验证命令
3. **不要静默删除旧逻辑**
4. 如果引入新行为，应优先通过配置或参数控制
5. 在主链没闭环前，不要继续引入新的大型模块
6. 当前最高优先级是：

```text
修工程阻塞项
→ 补语义评分模块
→ 打通语义输入链
→ 让语义项主导优化
→ 补强 LiDAR 上游语义与结构
→ 让 refine 真正使用语义 breakdown
→ 最后补测试
```

---

# 最终一句话目标

当前这轮修改的目标不是“继续堆模块”，而是：

**优先把 BEV 粗初始化、语义概率场精配准、多帧滑动窗口精修这三段主链真正闭环，并让上游 LiDAR 特征和下游 refinement 都围绕语义项协同工作。**

---

# TODO 结束

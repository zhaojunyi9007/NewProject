# OSDaR23 迁移实施方案（分阶段）

> 目标：在**不破坏现有 KITTI 工作流**的前提下，使当前 EdgeCalib 项目完整支持 OSDaR23（优先序列 `1_calibration_1.1`）。
> 
> 本文档给出“按阶段可落地”的代码修改计划，每阶段都包含：目标、要改的文件、改动内容、验收标准与风险。

---

## 0. 前置基线与原则

### 0.1 适用数据范围（当前阶段）
- 图像：`rgb_center/`（5MP）
- 点云：`lidar/`（merged PCD）
- 标定：`calibration.txt`
- 参考标签：`1_calibration_1.1_labels.json`（用于坐标系/外参交叉核验）

### 0.2 改造原则
1. **双制式并存**：保留 `kitti` 默认路径与语义；新增 `osdar23` 分支。
2. **最小侵入**：先改 I/O 适配，再改标定链路，再做参数调优。
3. **每阶段可回滚**：每阶段独立提交，保证“通过阶段验收后再进入下一阶段”。
4. **先打通再做优**：先确保跑通，再做精度提升与性能优化。

### 0.3 建议分支策略
- `feature/osdar23-stage1-io`
- `feature/osdar23-stage2-calib`
- `feature/osdar23-stage3-robustness`

---

## 阶段 1：配置与数据发现层改造（不改算法）

### 1.1 目标
让 Pipeline 能识别数据集类型，并解析 OSDaR23 的文件组织与命名（`counter_timestamp`）。

### 1.2 修改文件
- `configs/kitti.yaml`（以及 `_base: base.yaml`）
- `pipeline/context.py`
- （新增）`pipeline/datasets/resolver.py`

### 1.3 具体改动

#### A) 配置扩展
在 `data` 节点新增：

```yaml
data:
  dataset_format: "kitti"   # kitti | osdar23
  osdar_sequence_root: ""
  image_sensor: "rgb_center"  # osdar23 生效
```

并约定：
- `kitti`：沿用现有 `image_dir`、`velodyne_dir`。
- `osdar23`：若提供 `osdar_sequence_root`，自动推导：
  - `image_dir = <root>/<image_sensor>`
  - `velodyne_dir = <root>/lidar`
  - `calib_file = <root>/calibration.txt`

#### B) 统一“帧索引 -> 文件路径”机制
新增 `dataset_resolver.py`，抽象接口：
- `resolve_image(frame_id) -> path | None`
- `resolve_lidar(frame_id) -> path | list[path] | None`
- `list_available_frames() -> list[int]`

实现：
- `KittiResolver`：`{frame_id:010d}.png/.bin`
- `OSDaRResolver`：`{counter}_*.png/.pcd`（按前缀匹配）

#### C) `get_frame_list` 改造
`frames.mode=all` 时，不再硬编码“扫描 PNG 数字文件名”，改为调用 resolver 的 `list_available_frames()`。

### 1.4 验收标准
- `dataset_format=kitti` 时行为与当前一致。
- `dataset_format=osdar23` 时，`frames.mode=all` 能正确列出 `counter`。
- 以 `frame_ids=[0..9]` 时，能打印出每帧实际匹配到的图像和点云路径。

### 1.5 风险与对策
- **风险**：同一 counter 多文件（重复采样）
- **对策**：策略固定为“按时间戳排序取最新/最早”，写入日志并可配置。

---

## 阶段 2：SAM 与 LiDAR 输入链路适配（I/O 层）

### 2.1 目标
使阶段1/2（特征提取）可直接读取 OSDaR23 数据。

### 2.2 修改文件
- `pipeline/stages/sam_stage.py`
- `pipeline/stages/lidar_stage.py`
- `cpp/lidar_extractor.cpp`

### 2.3 具体改动

#### A) `sam_stage.py`
- 当前硬编码：`{frame_id:010d}.png`。
- 改为：通过 resolver 获取 `img_path`。
- 输出文件前缀仍用 `frame_id`（建议统一零填充，例如 `0000000009`），保持下游兼容。

#### B) `lidar_stage.py`
- 当前硬编码：融合帧 `bin_paths = ... .bin`。
- 改为：融合窗口内每个 frame_id 通过 resolver 获取点云路径。
- 对于 OSDaR23，输入为 `.pcd`。

#### C) `cpp/lidar_extractor.cpp`
- 在 `LocalIO::LoadKittiBin` 旁新增 `LoadPCD()`。
- `main` 中按输入扩展名分支：
  - `.bin` -> `LoadKittiBin`
  - `.pcd` -> `pcl::io::loadPCDFile`
- 对 PCD 字段兼容：
  - 优先读 `x y z intensity`
  - 缺 intensity 时赋默认值（例如 1.0）
  - 可选保留 `sensor_index`（后续用于加权）

### 2.4 验收标准
- 阶段1、2在 OSDaR23 前 10 帧可全部跑完。
- 产物文件完整：`_edge_dist.png`, `_points.txt`, `_lines_3d.txt` 等。
- 与 kitti 回归对比，输出数量级不异常（点数、线数）。

### 2.5 风险与对策
- **风险**：PCD 格式多样（ascii/binary/binary_compressed）
- **对策**：统一依赖 PCL loader，读取失败时打印 header 诊断。

---

## 阶段 3：标定参数读取与投影链路适配（核心）

### 3.1 目标
正确接入 OSDaR23 的相机内参与外参，保证“LiDAR 点 -> 图像像素”几何链路正确。

### 3.2 修改文件
- `cpp/io_utils.cpp`
- `include/common.h`（必要时新增解析声明）
- `pipeline/stages/calib_stage.py`
- `cpp/calibrator.cpp`（仅在参数接口需要时）

### 3.3 具体改动

#### A) 新增 OSDaR 标定读取函数
在 `io_utils.cpp` 新增：
- `LoadOSDaRCalib(calib_file, camera_name, K, T_lidar_to_cam_optical, ...)`

读取来源：
- `intrinsics_pinhole`
- `pose_wrt_parent`（LiDAR -> 非光学相机系）
- 固定旋转四元数：`x=0.5, y=-0.5, z=0.5, w=-0.5`

#### B) 坐标系转换链路显式化
在注释和日志中明确：
1. `p_cam_nonopt = T_lidar_to_cam_nonopt * p_lidar`
2. `p_cam_opt = R_nonopt_to_opt * p_cam_nonopt`
3. `u,v = K * p_cam_opt`

#### C) `calib_stage.py` 初值策略分支
- `kitti`：保留当前 `velo_to_cam + cam_to_cam` 逻辑。
- `osdar23`：
  - 不再解析 `Tr_velo_to_cam`。
  - 由 OSDaR calibration 读取初始外参（或从 labels 校验后覆盖）。

#### D) fallback 机制
- OSDaR 标定解析失败时：
  - 明确报错并停止（不建议 silently 使用 KITTI 默认 K）。

### 3.4 验收标准
- 初始位姿下 `in_image` 比例明显大于 0（不是全出界）。
- 优化日志中 `behind_ratio` / `out_of_bounds_ratio` 可解释且稳定。
- 使用 `rgb_center` 时投影可视化与场景结构一致（轨道/立柱方向正确）。

### 3.5 风险与对策
- **风险**：非光学->光学变换漏乘导致整体偏移
- **对策**：写单元测试固定 3D 点投影，输出像素做回归。

---

## 阶段 4：文件命名与中间产物约定统一

### 4.1 目标
避免 OSDaR 的命名方式影响下游所有模块。

### 4.2 修改文件
- `pipeline/stages/sam_stage.py`
- `pipeline/stages/lidar_stage.py`
- `pipeline/stages/calib_stage.py`
- `tools/visualize.py`

### 4.3 具体改动
- 内部统一使用 `logical_frame_id`（int）作为中间产物前缀。
- 原始文件名（带 timestamp）仅在 resolver 层处理。
- 运行日志同时输出：
  - logical_frame_id
  - source_image
  - source_lidar

### 4.4 验收标准
- 任何阶段都不再依赖“原始文件名是否 10 位数字”。
- 产物命名稳定，可直接复用现有可视化与评估脚本。

---

## 阶段 5：参数重整（针对铁路场景）

### 5.1 目标
把 KITTI 参数迁移到 OSDaR 场景下的可用区间。

### 5.2 修改文件
- `configs/kitti.yaml`（以及 `_base: base.yaml`）
- （新增）`configs/osdar23.yaml`

### 5.3 建议参数方向
- `lidar.ground_z_min/max`：按 OSDaR 坐标范围重设。
- `rail_ransac_threshold`：适配轨道几何尺度。
- `coarse_*_range/step`：先放宽后收紧。
- `t_prior_*`：早期建议减弱，避免拉回错误 basin。

### 5.4 验收标准
- 多帧结果中平移/旋转抖动减小。
- 轨道/立柱线约束命中率上升。

---

## 阶段 6：测试与质量保障

### 6.1 测试补齐
新增测试建议：
1. `tests/test_dataset_resolver_osdar.py`
   - counter 匹配、all 模式帧扫描、缺帧处理。
2. `tests/test_lidar_loader_pcd.py`
   - ascii/binary pcd 读取，字段缺失降级。
3. `tests/test_osdar_calib_parse.py`
   - intrinsics/extrinsics 解析与坐标变换正确性。
4. `tests/test_smoke_osdar23_10frames.py`
   - 10 帧端到端 smoke。

### 6.2 观测指标
- 每帧：`in_image_ratio`, `behind_ratio`, `score_before/after`, 优化耗时。
- 全序列：均值/方差、失败帧索引。

---


## 附录 A：实施顺序（推荐）

1. 阶段1（resolver + config）
2. 阶段2（sam/lidar I/O）
3. 阶段3（calib 几何链路）
4. 阶段4（命名统一）
5. 阶段5（参数调优）
6. 阶段6（测试）


---

## 附录 B：每阶段完成定义（DoD）模板

每阶段至少满足：
1. 有代码变更清单。
2. 有可复现命令。
3. 有通过/失败判据。
4. 有回滚方案。


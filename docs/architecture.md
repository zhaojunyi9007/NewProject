# EdgeCalib 当前架构冻结说明（Stage 1）

> 本文档用于“冻结现状 + 建立边界”。本阶段不修改算法行为，只整理结构边界与入口。

## 1. 当前主流程（保持不变）

主入口为 `run_pipeline.py`，阶段顺序如下：

1. `run_sam_extraction()`：调用 `python/run_sam.py` 生成图像侧特征
2. `run_lidar_extraction()`：调用 `build/lidar_extractor` 生成点云侧特征
3. `run_calibration()`：调用 `build/optimizer` 进行标定优化
4. `run_visualization()`：调用 `visual_result.py` 输出可视化结果

本阶段不改变上述执行顺序、参数语义和默认行为。

## 2. 当前输入/输出契约（冻结）

### 2.1 配置入口
- 主配置：`config.yaml`
- Stage 1 新增镜像入口：`configs/default.yaml`（内容与 `config.yaml` 一致）

### 2.2 关键输出目录（由配置 `data.*_output_dir` 决定）
- SAM 特征：`result/sam_features`
- LiDAR 特征：`result/lidar_features`
- 标定结果：`result/calibration`
- 可视化：`result/visualization`

### 2.3 关键中间文件命名（保持）
- 图像侧：`<frame>_edge_map.png`, `<frame>_edge_dist.png`, `<frame>_lines_2d.txt` 等
- 点云侧：`<frame>_points.txt`, `<frame>_edge_points.txt`, `<frame>_lines_3d.txt`
- 标定：`<frame>_calib_result.txt`

## 3. 边界调整（Stage 1）

仅进行结构边界调整，不改算法：

- 将实验脚本迁移到 `experiments/` 目录：
  - `experiments/line_constraint_ab_test.py`
  - `experiments/check_lines.py`
  - `experiments/check_features.py`
- 在仓库根目录保留同名兼容入口脚本，转发至 `experiments/` 中对应实现。

## 4. 非目标（本阶段明确不做）

- 不改 `cpp/optimizer.cpp`、`cpp/lidar_extractor.cpp` 的算法逻辑
- 不改 `python/sam_extractor.py` 的过滤策略
- 不接入 `mask alignment` 或其他新方案
- 不统一 ENV/YAML/CLI 的优先级规则（该项留在后续阶段）

## 5. 回滚策略

若 Stage 1 调整引起任何问题，可按提交粒度整体回滚：
- 回滚目录迁移与兼容入口
- 回滚 `docs/architecture.md` 与 `configs/default.yaml`
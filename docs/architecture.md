# EdgeCalib 当前架构冻结说明（Stage 1）

> 本文档用于“冻结现状 + 建立边界”。本阶段不修改算法行为，只整理结构边界与入口。

## 1. 当前主流程（保持核心行为不变）

主入口为 `run_pipeline.py`，通过 `pipeline/runner.py` 调度阶段模块，阶段顺序如下：

1. `pipeline/stages/sam_stage.py`：调用 SAM 特征插件（默认 `pipeline/sam/subprocess_plugin.py`）
2. `pipeline/stages/lidar_stage.py`：调用 `build/lidar_extractor` 生成点云侧特征
3. `pipeline/stages/calib_stage.py`：调用 `build/optimizer` 进行标定优化
4. `pipeline/stages/visual_stage.py`：调用 `tools/visualize.py` 输出可视化结果

本阶段不改变上述执行顺序、参数语义和默认行为。

## 2. 当前输入/输出契约（冻结）

### 2.1 配置入口
- 配置入口：`configs/kitti.yaml` / `configs/osdar23.yaml`（通过 `_base: base.yaml` 复用公共配置）

### 2.2 关键输出目录（由配置 `data.*_output_dir` 决定）
- SAM 特征：`result/sam_features`
- LiDAR 特征：`result/lidar_features`
- 标定结果：`result/calibration`
- 可视化：`result/visualization`

### 2.3 关键中间文件命名（保持）
- 图像侧：`<frame>_edge_map.png`, `<frame>_edge_dist.png`, `<frame>_rail_region.png`, `<frame>_rail_centerline.png`, `<frame>_rail_dist.png` 等
- 点云侧：`<frame>_points.txt`, `<frame>_edge_points.txt`, `<frame>_lines_3d.txt`
- 标定：`<frame>_calib_result.txt`

## 3. 边界调整（Stage 1）

仅进行结构边界调整，不改算法：

- 将实验脚本迁移到 `experiments/` 目录：
  - `experiments/check_features.py`
- 已移除仓库根目录 legacy 兼容脚本入口，避免影子入口。

## 3.1 插件化接口（Stage 4）

- 图像特征阶段插件接口：`pipeline/sam/interfaces.py`。  
- 默认特征插件实现：`pipeline/sam/subprocess_plugin.py`（保持原有子进程行为）。  
- 优化器约束新增适配器接口：`pipeline/optimizer_constraint_adapter.py`。  
- 默认约束适配器仍使用既有 `ab_experiment -> ENV` 映射，行为保持不变。

## 4. 非目标（本阶段明确不做）

- 不改 `cpp/optimizer.cpp`、`cpp/lidar_extractor.cpp` 的算法逻辑
- 不改 `python/sam_extractor.py` 的过滤策略
- 不接入 `mask alignment` 或其他新方案
- 不统一 ENV/YAML/CLI 的优先级规则（该项留在后续阶段）

## 5. 回滚策略

若 Stage 1 调整引起任何问题，可按提交粒度整体回滚：
- 回滚目录迁移与兼容入口
- 回滚 `docs/architecture.md` 与配置拆分（`configs/base.yaml` 等）
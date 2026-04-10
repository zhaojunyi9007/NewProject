# TODO List：基于 SAM 的 2D 轨道特征重构方案（替代 LSD）
## 目标
- 删除 LSD 作为 2D 轨道主特征的整条链路。
- 保留 SAM / semantic_probs / edge 分支。
- 新增 `rail_region + rail_centerline + rail_dist + rail_weight` 分支。
- 用 `rail term` 替代 `line term`。

---

## A. 先改配置，先停用旧 line 分支

### 1. 修改 `configs/base.yaml`
- [ ] `image_features.enable_lines: false`
- [ ] `image_features.enable_rail_region: true`
- [ ] 新增：
  - [ ] `rail_seed_threshold: 0.42`
  - [ ] `rail_support_threshold: 0.22`
  - [ ] `rail_mask_min_area: 400`
  - [ ] `rail_close_kernel: 9`
  - [ ] `rail_open_kernel: 5`
  - [ ] `rail_top_ignore_ratio: 0.18`
  - [ ] `rail_bottom_keep_ratio: 0.75`
  - [ ] `rail_component_min_height_ratio: 0.05`
  - [ ] `rail_skeleton_min_length_px: 40`
  - [ ] `rail_dist_max_ratio: 0.08`
  - [ ] `rail_weight_dilate_kernel: 9`
- [ ] `semantic_calib.line_weight: 0.0`
- [ ] `semantic_calib.rail_weight: 1.2`

### 2. 修改 `configs/osdar23.yaml`
- [ ] `image_features.enable_lines: false`
- [ ] `image_features.enable_rail_region: true`
- [ ] 写入 OSDaR23 rail 参数：
  - [ ] `rail_seed_threshold: 0.42`
  - [ ] `rail_support_threshold: 0.22`
  - [ ] `rail_mask_min_area: 500`
  - [ ] `rail_close_kernel: 9`
  - [ ] `rail_open_kernel: 5`
  - [ ] `rail_top_ignore_ratio: 0.18`
  - [ ] `rail_bottom_keep_ratio: 0.78`
  - [ ] `rail_component_min_height_ratio: 0.05`
  - [ ] `rail_skeleton_min_length_px: 50`
  - [ ] `rail_dist_max_ratio: 0.08`
  - [ ] `rail_weight_dilate_kernel: 11`
- [ ] `calibration.ab_experiment.use_line_constraint: false`

### 3. 立即删除/注释这些旧配置项
- [ ] `keep_diagonal_lines`
- [ ] `restrict_lsd_by_semantics`
- [ ] `line_semantic_support_threshold`
- [ ] `global_min_line_length_ratio`
- [ ] `ground_min_line_length_ratio`
- [ ] `require_edge_overlap`

---

## B. 改 Python 前端：停用 LSD，新增 rail 分支

### 4. 文件：`tools/sam_extractor.py`

#### 4.1 停用旧调用链
- [ ] 从 `process_image_feature_bundle(...)` 中删除 `extract_lines_2d()` 的主调用。
- [ ] 删除 `line_mask` 参与 `fused_edge_map` 的逻辑。
- [ ] 将 `fused_edge_map` 改为：`edge_sem OR edge_map_sam`。

#### 4.2 新增函数
- [ ] 新增 `_build_rail_probability_maps(...)`
- [ ] 新增 `_build_rail_region_from_masks(...)`
- [ ] 新增 `_skeletonize_binary_mask(...)`
- [ ] 新增 `_extract_centerline_polylines(...)`
- [ ] 新增 `_build_distance_map_from_centerline(...)`

#### 4.3 新流程替代 LSD
在 `process_image_feature_bundle(...)` 中改成：
- [ ] 生成 `semantic_probs`
- [ ] 生成 `edge_sem`
- [ ] 生成 `edge_map_sam / edge_weight / mask_id_map`
- [ ] 生成 `rail_region_mask`
- [ ] 生成 `rail_centerline_mask`
- [ ] 生成 `rail_centerlines_2d`
- [ ] 生成 `rail_dist`
- [ ] 生成 `rail_weight`

#### 4.4 新增输出文件
在 `save_image_feature_bundle(...)` 和 `sam_output_base` 同步输出：
- [ ] `rail_region.png`
- [ ] `rail_centerline.png`
- [ ] `rail_dist.png`
- [ ] `rail_weight.png`
- [ ] `rail_centerlines_2d.txt`

#### 4.5 立即删除旧输出逻辑
- [ ] 删除 `line_map.png` 的写出逻辑。
- [ ] 删除 `lines_2d.txt` 的写出逻辑。
- [ ] 删除 `sam_output_base + "_line_map.png"`
- [ ] 删除 `sam_output_base + "_lines_2d.txt"`

#### 4.6 顺手修掉错误写图
- [ ] 把 `semantic_map.png` 从 `mask_id_map` 改成真实 `argmax_vis`。

---

## C. 改 observability：不再依赖 `lines_2d.txt`

### 5. 文件：`pipeline/observability.py`
- [ ] 删除基于 `lines_2d.txt` 数量的评分逻辑。
- [ ] 改成优先统计 `rail_centerlines_2d.txt`。
- [ ] 如果没有 txt，则 fallback 到 `rail_centerline.png` 的非零像素占比。

---

## D. 改 C++ 数据结构：新增 rail term，停用 line term

### 6. 文件：`cpp/include/common.h`
- [ ] 在 `ScoreBreakdown` 中新增：
  - [ ] `double rail_score_norm = 0.0;`
  - [ ] `double rail_score = 0.0;`

### 7. 文件：`cpp/include/edge_calibrator.h`
- [ ] 在 `EdgeCalibratorConfig` 中新增：
  - [ ] `double rail_weight = 1.2;`
- [ ] 在 `EdgeCalibrator` 成员中新增：
  - [ ] `cv::Mat rail_dist_;`
  - [ ] `cv::Mat rail_weight_;`
  - [ ] `cv::Mat rail_region_;`
  - [ ] `cv::Mat rail_centerline_;`
  - [ ] `std::vector<PointFeature> rail_sample_points_;`
- [ ] 新增私有函数：`BuildRailSamplePoints();`

### 8. 文件：`cpp/main.cpp`
- [ ] 新增命令行参数：`--rail_weight`

### 9. 文件：`pipeline/stages/calib_stage.py`
- [ ] 在 optimizer 命令里传入 `--rail_weight`

---

## E. 改 C++ 加载逻辑：读取新 rail 图

### 10. 文件：`cpp/edge_calibrator.cpp`
在 `LoadData()` 中新增读取：
- [ ] `*_rail_dist.png`
- [ ] `*_rail_weight.png`
- [ ] `*_rail_region.png`
- [ ] `*_rail_centerline.png`

然后：
- [ ] 在 `lines3d_` 读取完成后调用 `BuildRailSamplePoints()`。

### 11. 实现 `BuildRailSamplePoints()`
- [ ] 只对 `SEM_RAIL_LIKE` 或 `type==0` 的 3D rail line 采样。
- [ ] 采样间隔先用 `0.25m`。
- [ ] 采样结果写入 `rail_sample_points_`。

---

## F. 改 scoring：新增 rail score，停用 line score

### 12. 文件：`cpp/include/optimizer_scoring.h`
- [ ] 扩展 `ComputeTotalCalibrationScoreSemanticDominant(...)` 的参数：
  - [ ] `rail_points`
  - [ ] `rail_dist`
  - [ ] `rail_weight`
  - [ ] `w_rail`

### 13. 文件：`cpp/optimizer_scoring.cpp`
- [ ] 新增 `rail_score_norm` 计算。
- [ ] 直接复用 `EdgeAttractionScore(...)`，对 `rail_sample_points_` 在 `rail_dist` 上打分。
- [ ] 把 `rail_score` 加入总分。
- [ ] 保留 `line_score` 代码仅作临时过渡，但不再参与总分。

---

## G. 改 fine stage：加入 rail residual

### 14. 文件：`cpp/edge_calibrator.cpp`
在 fine optimization 中：
- [ ] 保留原有 `generic edge residual`
- [ ] 新增 `rail_dist` residual
- [ ] 对 `rail_sample_points_` 使用 `SinglePointEdgeCost`
- [ ] 给 rail residual 加 `HuberLoss(0.05)`

---

## H. 改结果输出和可视化

### 15. 文件：`cpp/edge_calibrator.cpp`
- [ ] 在 `SaveResult()` 中新增输出：`rail_term_norm`
- [ ] `line_term_norm` 不再输出，或在 rail 分支稳定后彻底删除

### 16. 文件：可视化脚本（如 `tools/visualize.py` / `tools/visualize_diag.py`）
- [ ] 新增叠加显示：
  - [ ] `rail_region.png`
  - [ ] `rail_centerline.png`
- [ ] 让调试主图从 `line_map.png` 切换到 `rail_region/rail_centerline/rail_dist`

### 17. 删除旧可视化依赖
- [ ] 删除 `line_map.png` 作为主调试图的逻辑。
- [ ] 删除 `lines_2d.txt` 相关可视化逻辑。

---

## I. 彻底删除旧 LSD / line 分支（在 rail 分支跑通后执行）

### 18. Python 端删除项
- [ ] 删除 `extract_lines_2d()` 函数本体。
- [ ] 删除 `_line_class_to_railway_semantic_id(...)`（若不再有其他用途）。
- [ ] 删除所有 LSD 专属参数解析。
- [ ] 删除所有 `line_map / lines_2d` 输出代码。

### 19. C++ 端删除项
- [ ] 删除 `LoadLines2D(...)` 调用链。
- [ ] 删除 `lines2d_` 成员。
- [ ] 删除 `ComputeLineAlignmentScoreWeighted(...)`。
- [ ] 删除 `line_score_norm / line_score`。
- [ ] 删除 fine stage 的 line residual / line constraint。
- [ ] 删除这些仅服务 LSD 的参数：
  - [ ] `--use_line_constraint`
  - [ ] `--line_weight`
  - [ ] `--line_match_threshold`
  - [ ] `--line_soft_penalty`
  - [ ] `--line_soft_cap`

### 20. 删除旧结果文件
删除旧产物，避免结果混淆：
- [ ] `result_osdar23/sam_features/*_line_map.png`
- [ ] `result_osdar23/sam_features/*_lines_2d.txt`
- [ ] `result_osdar23/image_features/*/line_map.png`
- [ ] `result_osdar23/image_features/*/lines_2d.txt`

---

## J. 运行前必须清理

### 21. 清空旧结果目录中的旧产物
至少删除：
- [ ] `result_osdar23/sam_features/*_line_map.png`
- [ ] `result_osdar23/sam_features/*_lines_2d.txt`
- [ ] `result_osdar23/sam_features/*_rail_region.png`
- [ ] `result_osdar23/sam_features/*_rail_centerline.png`
- [ ] `result_osdar23/sam_features/*_rail_dist.png`
- [ ] `result_osdar23/sam_features/*_rail_weight.png`
- [ ] `result_osdar23/sam_features/*_rail_centerlines_2d.txt`
- [ ] `result_osdar23/calibration/*`
- [ ] `result_osdar23/visualization/*`
- [ ] `result_osdar23/refinement/*`

### 22. 重编译
- [ ] 重新编译 C++。

---

## K. 验收标准（只保留必查项）

### 23. Python 前端验收
- [ ] `rail_region.png` 覆盖轨道区域，而不是建筑边。
- [ ] `rail_centerline.png` 落在轨道中轴/轨道带上。
- [ ] `rail_dist.png` 在轨道附近低、离开轨道后高。
- [ ] 不再生成 `line_map.png` / `lines_2d.txt`。

### 24. 优化器验收
- [ ] `calib_result.txt` 中出现 `rail_term_norm`。
- [ ] `rail_term_norm > 0`。
- [ ] `line_term_norm` 不再参与结果分析。
- [ ] 最终投影比当前版本更贴近轨道。

### 25. 删除验收
- [ ] 全项目主流程里不再有 LSD 轨道路径。
- [ ] 输出目录里不再有旧 `line_*` 文件。
- [ ] 配置文件里不再保留无用 LSD 参数。
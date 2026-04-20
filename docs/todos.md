# EdgeCalib v2.0 OsDaR23 待办事项清单

> 基于代码分析报告（`docs/analysis_report.md`）与可视化结果诊断生成  
> 更新日期：2026-04-20

---

## 优先级说明

- **P0 - 阻塞级**：当前结果完全错误的直接原因，必须最先修复
- **P1 - 高优先级**：影响运行正确性，修复后结果可显著改善
- **P2 - 中优先级**：影响结果质量或鲁棒性
- **P3 - 低优先级**：代码整洁、文案清理，不影响运行

---

## P0：阻塞级 Bug（导致当前可视化结果完全错误）

### T-01 `bev_stage.py`：增加 BEV 粗初始化置信度门槛

**问题**：BEV 粗初始化的 `rail_score ≈ 7.6e-4 ≈ 0`，但仍无条件应用了 `tx=-2m, ty=+1m, yaw=5°` 的大幅修正，导致后续优化的初始外参偏离真值 2.13m，最终轨道特征完全飞出图像（u ≈ -1398px）。

**修改文件**：`pipeline/stages/bev_stage.py`

**修改内容**：
- 在 `bev_stage.py` 中，读取 `debug_bev_score.json` 的 `rail_score` 值
- 新增配置项 `bev.min_rail_score_to_apply`（建议默认值 `0.01`）
- 低于门槛时拒绝应用 BEV delta，保留原始 `init_pose`，打印警告

**对应配置**：在 `configs/base.yaml` 的 `bev` 段新增：
```yaml
bev:
  min_rail_score_to_apply: 0.01
```

**验证方法**：重新运行后，`pose_after_bev.txt` 应与 `init_pose.txt` 保持一致（tx ≈ -0.28m），绿色轨道线应出现在图像中。

---

### T-02 `tools/run_sam.py`：删除硬编码 BEV 参数

**问题**：`run_sam.py` 在 `--semantic_bundle` 分支内硬编码 BEV 参数，与 `osdar23.yaml` 和 LiDAR C++ 实际输出均不一致，导致图像侧 pseudo-BEV 与 LiDAR 侧 BEV 网格系统性错位，BEV 粗搜索基于错误对齐。

| 来源 | x 范围 | y 范围 | 分辨率 |
|------|--------|--------|--------|
| `run_sam.py`（硬编码，错误） | 0–80 m | ±20 m | 0.1 m |
| `osdar23.yaml`（配置，应以此为准） | 0–100 m | ±25 m | 0.05 m |
| LiDAR C++ 实际输出（`_bev_meta.json`） | 0–100 m | ±25 m | 0.2 m |

**修改文件**：`tools/run_sam.py`（约 171–175 行）、`pipeline/stages/image_feature_stage.py`

**修改内容**：
- 删除 `run_sam.py` 内的硬编码 `bev_cfg` 字典
- 改为新增命令行参数 `--bev_x_min`、`--bev_x_max`、`--bev_y_min`、`--bev_y_max`、`--bev_resolution`
- `image_feature_stage.py` 调用 `run_sam.py` 时，从 `config["bev"]` 读取并传入上述参数

---

## P1：高优先级（影响运行正确性）

### T-03 `visual_stage.py`：`image_sensor` 默认值不一致

**问题**：`visual_stage.py` 中 `image_sensor` 默认为 `""`（空字符串），而 `calib_stage.py`、`osdar23.py`、`run_sam.py` 等均默认 `"rgb_center"`。配置未显式写 `image_sensor` 时，可视化不传 `--image_sensor` 参数，`visualize.py` 从图像路径推断相机，可能与标定所用相机不一致。

**修改文件**：`pipeline/stages/visual_stage.py`（第 24 行）

**修改内容**：
```python
# 修改前
img_sensor = str(context.config.get("data", {}).get("image_sensor", "") or "")

# 修改后
img_sensor = str(context.config.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center")
```

---

### T-04 `sam_extractor.py`：放宽轨道检测的垂直 ROI 限制

**问题**：`rail_top_ignore_ratio: 0.18` 和 `rail_bottom_keep_ratio: 0.78` 设置，在长直铁路强透视下将远处轨道（图像中上部）系统性排除。实测结果中，所有轨道中心线点集中在 `v ≈ 758–761`（图像最底部 3–4 行），导致 `rail_dist` 吸引场覆盖极窄，`visible_count < 50` 触发哨兵值。

**修改文件**：`configs/osdar23.yaml`（约第 61–62 行）

**修改内容**：
```yaml
# 修改前
rail_top_ignore_ratio: 0.18
rail_bottom_keep_ratio: 0.78

# 修改后（建议值，需根据实际效果调整）
rail_top_ignore_ratio: 0.05
rail_bottom_keep_ratio: 0.95
```

---

## P2：中优先级（提升代码质量与结果鲁棒性）

### T-05 `observability.py`：消除 `semantic_probs.npy` 重复加载

**问题**：`compute_frame_observability` 函数中对同一 `npy` 文件执行了两次 `np.load`（第 107 行的 `_safe_mean_max_prob` 内部一次，第 112 行又一次），造成不必要的 IO 开销。

**修改文件**：`pipeline/observability.py`

**修改内容**：合并为一次加载，将 `z` 数组传给两处计算逻辑，删除重复 `np.load`。

---

### T-06 `configs/osdar23.yaml`：删除 KITTI 时代遗留字段

**问题**：`velo_to_cam_file: ""` 是 KITTI 时代的字段，`OSDaR23Adapter` 完全未使用该字段，长期保留会误导使用者。

**修改文件**：`configs/osdar23.yaml`（第 21 行）

**修改内容**：删除该行：
```yaml
velo_to_cam_file: ""   # 删除此行
```

---

### T-07 `configs/base.yaml`：为 `temporal_filter` 补充 `projection_threshold` 注释说明

**问题**：在上一轮修改中新增了 `projection_threshold` 字段，用于区分 `EDGECALIB_LIDAR_TEMPORAL_POS_THRESH` 和 `EDGECALIB_LIDAR_TEMPORAL_PROJ_THRESH`，但缺少注释说明两个阈值的含义差异。

**修改文件**：`configs/base.yaml`（约第 80–82 行）

**修改内容**：补充注释：
```yaml
temporal_filter:
  enabled: true
  position_threshold: 0.5    # 3D 空间位置一致性阈值 (m)，对应 TEMPORAL_POS_THRESH
  projection_threshold: 0.5  # 球面投影距离一致性阈值 (m)，对应 TEMPORAL_PROJ_THRESH
  static_weight: 1.0
  dynamic_weight: 0.1
```

---

## P3：低优先级（文案清理，不影响运行）

### T-08 `pipeline/datasets/resolver.py`：更新模块文档字符串

**问题**：模块头部文档仍描述 KITTI 文件名格式（`{frame_id:010d}.png/.bin`），当前仅保留 `OSDaRResolver`。

**修改文件**：`pipeline/datasets/resolver.py`（第 4–7 行）

**修改内容**：将文档字符串改为仅描述 OsDaR23 格式。

---

### T-09 `pipeline/datasets/base.py`：更新文档字符串

**问题**：基类文档仍写 "KITTI vs OSDaR23"，应改为仅描述 OsDaR23。

**修改文件**：`pipeline/datasets/base.py`

---

### T-10 `configs/osdar23.yaml`：去掉注释中的 KITTI 字样

**问题**：第 23 行注释写 "独立于 KITTI 的 result/"，已无意义。

**修改文件**：`configs/osdar23.yaml`（第 23 行）

---

## 后续（测试设计，待以上 P0–P1 完成后再执行）

### T-11 重新设计测试文件

在 P0–P1 修复完成、程序可正常运行后，针对以下核心逻辑补充测试：

| 测试文件（待创建） | 覆盖内容 |
|---------------------|----------|
| `tests/test_bev_stage_rail_score_gate.py` | `rail_score < 门槛` 时 BEV delta 不应用；`rail_score > 门槛` 时正常应用 |
| `tests/test_run_sam_bev_params.py` | `--bev_x_max` / `--bev_resolution` 参数正确传递，不再使用硬编码值 |
| `tests/test_osdar_frame_resolver.py` | OsDaR23 文件名解析（前缀计数器匹配，重复策略） |
| `tests/test_observability_no_double_load.py` | `compute_frame_observability` 只触发一次文件 IO |

---

## 完成状态跟踪

| 编号 | 描述 | 优先级 | 状态 |
|------|------|--------|------|
| T-01 | bev_stage 增加 rail_score 门槛 | P0 | 待办 |
| T-02 | run_sam.py 删除硬编码 BEV 参数 | P0 | 待办 |
| T-03 | visual_stage image_sensor 默认值 | P1 | 待办 |
| T-04 | sam_extractor 放宽轨道 ROI 限制 | P1 | 待办 |
| T-05 | observability 消除重复 np.load | P2 | 待办 |
| T-06 | 删除 velo_to_cam_file 遗留字段 | P2 | 待办 |
| T-07 | base.yaml temporal_filter 注释 | P2 | 待办 |
| T-08 | resolver.py 文档字符串清理 | P3 | 待办 |
| T-09 | base.py 文档字符串清理 | P3 | 待办 |
| T-10 | osdar23.yaml 注释去 KITTI 字样 | P3 | 待办 |
| T-11 | 重新设计测试文件 | 后续 | 等待 P0-P1 完成 |

---

*文档结束*

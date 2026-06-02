# experiments 目录说明

此目录存放实验性/一次性诊断脚本，不属于 `run_pipeline.py` 默认主流程依赖。

## 当前脚本

- `check_bev_alignment.py`：读取单帧 `bev_init/<frame_id>/` 调试产物，可选生成 BEV 诊断拼图。
- `check_refine_window.py`：读取 `refinement/state.json`，汇总多帧精修窗口稳定性。
- `check_semantic_alignment.py`：读取单帧 calibration score breakdown，检查语义对齐诊断信息。
- `evaluate_mask_alignment.py`：汇总 `result/mask_alignment/*.json` 的 mask alignment 指标。

## 使用原则

1. 仅用于实验分析与验证，不影响 `run_pipeline.py` 默认执行路径。
2. 主流程产物可被实验脚本读取；实验脚本不应反向修改主流程逻辑。
3. 若实验脚本升级为长期工具，应迁移到 `tools/` 并补充测试。

## 示例

```bash
python experiments/check_bev_alignment.py \
  --bev_frame_dir result_osdar23/bev_init/0000000012 \
  --lidar_base result_osdar23/lidar_features/0000000012 \
  --image_feat_dir result_osdar23/image_features/0000000012

python experiments/check_refine_window.py \
  --refinement_dir result_osdar23/refinement

python experiments/check_semantic_alignment.py \
  --calib_dir result_osdar23/calibration \
  --frame_id 0000000012

python experiments/evaluate_mask_alignment.py \
  --metrics_dir result/mask_alignment \
  --topk 10
```

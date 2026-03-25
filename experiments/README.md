# experiments 目录说明

此目录存放实验性/一次性诊断脚本，不属于默认主流程依赖。

## 当前脚本
- `line_constraint_ab_test.py`：line constraint 开关 A/B 对比实验。
- `check_lines.py`：单帧 3D 线投影诊断。
- `check_features.py`：单帧点特征投影诊断。

## 使用原则
1. 仅用于实验分析与验证，不影响 `run_pipeline.py` 默认执行路径。
2. 主流程产物可被实验脚本读取；实验脚本不应反向修改主流程逻辑。
3. 若需长期保留，请补充输入/输出说明与复现实验命令。
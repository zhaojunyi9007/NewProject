m# EdgeCalib v2.0 使用说明

## 📋 概述

EdgeCalib v2.0 是完整实现了 README2.0.md 中描述方案的城市轨道交通LiDAR-相机标定系统。

### 主要改进

✅ **完整实现README2.0方案**
- 第一阶段：SAM语义边缘 + LSD几何线 + 边缘吸引场
- 第二阶段：NDT多帧融合增密 + 时空一致性加权过滤
- 第三阶段：粗精两阶段优化 + 置信度检查与时序平滑

✅ **灵活的帧选择系统**
- 通过配置文件 `config.yaml` 自由选择要处理的帧
- 支持单帧、多帧或全数据集处理

✅ **统一的结果管理**
- 所有结果统一保存在 `result/` 目录
- 每类结果独立子文件夹，便于管理

---

## 🚀 快速开始

### 1. 环境准备

#### C++ 依赖
```bash
# Ubuntu/Debian
sudo apt-get install -y cmake build-essential \
    libpcl-dev libopencv-dev libeigen3-dev libceres-dev
```

#### Python 依赖
```bash
# 安装依赖
pip install -r requirements.txt

# 安装 SAM (Segment Anything Model)
pip install git+https://github.com/facebookresearch/segment-anything.git

# 下载 SAM 模型权重
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

### 2. 编译 C++ 程序

```bash
mkdir -p build
cd build
cmake ..
make -j4
cd ..
```

编译成功后会生成：
- `build/lidar_extractor`: LiDAR特征提取器（含NDT融合）
- `build/optimizer`: 两阶段标定优化器（含时序平滑）

### 3. 配置数据路径

编辑 `config.yaml`：

```yaml
data:
  velodyne_dir: "data/velodyne_points/data"   # 点云目录
  image_dir: "data/image_02/data"             # 图像目录
  calib_file: "data/calib_cam_to_cam.txt"     # 标定文件(可选)

frames:
  mode: "select"        # "select" 或 "all"
  frame_ids: [0, 10, 20]  # 指定要处理的帧ID
  fusion_window: 3      # NDT融合窗口大小
```

### 4. 运行完整流程

#### 方式1：使用主控脚本（推荐）

```bash
python run_pipeline.py
```

#### 方式2：使用示例脚本

```bash
chmod +x run_example.sh
./run_example.sh
```

#### 方式3：分阶段运行

```bash
# 阶段1: SAM特征提取
python run_pipeline.py --stage sam

# 阶段2: LiDAR特征提取（含NDT融合）
python run_pipeline.py --stage lidar

# 阶段3: 标定优化
python run_pipeline.py --stage calib

# 阶段4: 可视化
python run_pipeline.py --stage visual
```

---

## 📂 目录结构

```
EdgeCalib v2.0/
├── config.yaml                 # 配置文件 ⭐新增⭐
├── run_pipeline.py             # 主控脚本 ⭐新增⭐
├── run_example.sh              # 快速示例脚本 ⭐新增⭐
│
├── cpp/                        # C++ 源码
│   ├── lidar_extractor.cpp     # ⭐已更新：NDT融合+时空加权⭐
│   ├── optimizer.cpp           # ⭐已更新：置信度检查+时序平滑⭐
│   └── io_utils.cpp            # ⭐已更新：支持weight字段⭐
│
├── include/
│   └── common.h                # ⭐已更新：PointFeature添加weight⭐
│
├── python/
│   ├── sam_extractor.py        # SAM特征提取
│   └── run_sam.py              # SAM运行脚本
│
├── visual_result.py            # ⭐已更新：适配新格式⭐
│
└── result/                     # ⭐统一输出目录⭐
    ├── sam_features/           # SAM提取的特征
    ├── lidar_features/         # LiDAR提取的特征
    ├── calibration/            # 标定结果
    └── visualization/          # 可视化结果
```

---

## ⚙️ 配置文件详解

### `config.yaml` 主要配置项

#### 1. 帧选择配置
```yaml
frames:
  mode: "select"           # "select": 指定帧  "all": 全部帧
  frame_ids: [0, 1, 2]     # 要处理的帧ID列表
  fusion_window: 3         # NDT融合窗口（当前帧+前N-1帧）
```

**示例场景**：
- 单帧测试：`frame_ids: [0]`, `fusion_window: 1`
- Three-in-one融合：`frame_ids: [2, 5, 8]`, `fusion_window: 3`
- 处理所有帧：`mode: "all"`

#### 2. NDT配置（README2.0 第二阶段 2.1）
```yaml
lidar:
  ndt:
    enabled: true
    transformation_epsilon: 0.01  # 变换收敛阈值
    step_size: 0.1                # 线搜索步长
    resolution: 1.0               # NDT网格分辨率(m)
    max_iterations: 30            # 最大迭代次数
```

#### 3. 时空一致性配置（README2.0 第二阶段 2.3）
```yaml
lidar:
  temporal_filter:
    enabled: true
    position_threshold: 0.5   # 位置一致性阈值(m)
    static_weight: 1.0        # 静态物体权重
    dynamic_weight: 0.1       # 动态物体权重
```

#### 4. 标定优化配置
```yaml
calibration:
  coarse:
    iterations: 1000          # 粗搜索迭代次数
    angle_range: 0.15         # 角度搜索范围(rad)
    position_range: 0.8       # 位置搜索范围(m)
  
  temporal_validation:        # README2.0 第三阶段 3.3
    enabled: true
    smoothness_threshold: 0.05  # 平滑性阈值
    history_frames: 3           # 历史帧数
```

---

## 📊 输出文件说明

### 1. SAM特征（`result/sam_features/`）

每帧生成：
- `XXXXXX_edge_map.png`: SAM边缘图
- `XXXXXX_line_map.png`: LSD线特征图
- `XXXXXX_edge_fused.png`: 融合边缘图
- `XXXXXX_edge_dist.png`: 边缘吸引场（16-bit）
- `XXXXXX_edge_weight.png`: 边缘权重图（16-bit）
- `XXXXXX_lines_2d.txt`: 2D线特征（u1 v1 u2 v2 type）

### 2. LiDAR特征（`result/lidar_features/`）

每帧生成：
- `XXXXXX_points.txt`: 点特征
  ```
  # x y z intensity nx ny nz label weight
  1.234 2.345 -1.234 45.6 0.123 0.234 0.345 3 0.95
  ```
  - `label`: 0=Unknown, 1=Road, 2=Vegetation, 3=Structure
  - `weight`: 时空一致性权重（0.0~1.0，越高越静态）

- `XXXXXX_lines_3d.txt`: 3D线特征
  ```
  # x1 y1 z1 x2 y2 z2 type
  1.0 2.0 -1.5 10.0 12.0 -1.5 0
  ```
  - `type`: 0=Rail(铁轨), 1=Pole(立柱)

### 3. 标定结果（`result/calibration/`）

每帧生成：
- `XXXXXX_calib_result.txt`: 标定结果
  ```
  # EdgeCalib v2.0 Calibration Result
  0.0123 0.0234 0.0345
  0.050 -0.280 1.750
  # Score: 12345.67
  ```
  - 第2行：旋转向量（angle-axis，rad）
  - 第3行：平移向量（meters）

### 4. 可视化结果（`result/visualization/`）

每帧生成：
- `XXXXXX_result.png`: 投影可视化图
  - 黄色点：LiDAR点云投影
  - 绿色线：铁轨线特征投影
  - 红色线：立柱线特征投影

---

## 🔧 高级用法

### 1. 只处理指定帧

修改 `config.yaml`：
```yaml
frames:
  mode: "select"
  frame_ids: [10, 20, 30]  # 只处理第10, 20, 30帧
```

### 2. 跳过已完成的阶段

```bash
# 如果SAM特征已提取，只运行后续阶段
python run_pipeline.py --skip-sam

# 只运行可视化
python run_pipeline.py --stage visual
```

### 3. 调整NDT融合窗口

```yaml
frames:
  frame_ids: [5, 10, 15]
  fusion_window: 5  # 每次融合5帧（当前帧+前4帧）
```

### 4. 禁用时序平滑

```yaml
calibration:
  temporal_validation:
    enabled: false  # 禁用时序平滑（不推荐）
```

---

## 🐛 故障排查

### 问题1: NDT配准失败

**症状**：`[Warning] NDT did not converge`

**解决**：
1. 检查帧间运动是否过大
2. 调整NDT参数：
   ```yaml
   ndt:
     resolution: 1.5        # 增大网格分辨率
     max_iterations: 50     # 增加迭代次数
   ```

### 问题2: 点云特征太少

**症状**：`[Warning] No valid points loaded`

**解决**：
1. 检查降采样参数：
   ```yaml
   lidar:
     voxel_size: 0.05  # 减小体素大小（更密集）
   ```
2. 检查融合窗口设置

### 问题3: 标定结果跳变

**症状**：相邻帧标定结果差异大

**解决**：
1. 启用时序平滑：
   ```yaml
   calibration:
     temporal_validation:
       enabled: true
       smoothness_threshold: 0.03  # 更严格的阈值
   ```

### 问题4: 可视化结果为空

**检查**：
1. 确认特征文件存在：
   ```bash
   ls result/lidar_features/000000_*.txt
   ls result/sam_features/000000_*.txt
   ```
2. 确认标定结果存在：
   ```bash
   ls result/calibration/000000_calib_result.txt
   ```

---

## 📈 性能优化建议

### 1. 多线程处理（未来改进）
当前版本按帧串行处理，可通过修改 `run_pipeline.py` 实现并行处理。

### 2. 粗搜索加速
```yaml
calibration:
  coarse:
    iterations: 500  # 减少迭代次数（可能影响精度）
```

### 3. 点云降采样
```yaml
lidar:
  voxel_size: 0.15  # 增大体素（更快但可能影响精度）
```

---

## 🎯 典型工作流程

### 场景1：首次使用

```bash
# 1. 编辑配置，选择少量帧测试
vim config.yaml  # 设置 frame_ids: [0]

# 2. 运行完整流程
python run_pipeline.py

# 3. 检查结果
ls result/visualization/
```

### 场景2：批量处理

```bash
# 1. 修改配置处理多帧
vim config.yaml  # 设置 frame_ids: [0, 5, 10, 15, 20]

# 2. 运行
python run_pipeline.py

# 3. 分析结果
python analyze_results.py  # (自行编写)
```

### 场景3：参数调优

```bash
# 1. 修改NDT参数
vim config.yaml

# 2. 只重新运行LiDAR提取和标定
python run_pipeline.py --skip-sam

# 3. 对比可视化结果
```

---

## 📚 参考资料

- **README2.0.md**: 完整方案描述
- **README.md**: 原版EdgeCalib文档
- **config.yaml**: 配置文件模板（含详细注释）

---

## 💡 常见问题

**Q: 如何选择fusion_window大小？**
A: 一般3-5帧。场景运动慢用更大值，运动快用更小值。

**Q: 时空权重weight的意义？**
A: weight接近1.0表示静态物体（铁轨、路灯），接近0.0表示动态物体（车辆）。优化时会优先使用高权重点。

**Q: 能否使用其他语义分割模型替代SAM？**
A: 可以。修改 `python/sam_extractor.py` 更换模型即可，保持输出格式一致。

**Q: 如何评估标定精度？**
A: 查看可视化结果中点云和线特征的对齐程度，或计算重投影误差。

---

## 📧 技术支持

如有问题，请检查：
1. 配置文件是否正确
2. 数据路径是否存在
3. C++程序是否编译成功
4. Python依赖是否安装完整

---

**EdgeCalib v2.0 - 让标定更简单、更鲁棒！** 🚀

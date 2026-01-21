# 基于几何与语义融合的 LiDAR-相机在线标定方案设计文档

## 1. 方案背景与适用场景

本方案专针对 **KITTI RAW 数据集** 所示的复杂城市道路场景设计。该场景具有以下显著特征：

* **强几何特征**：存在长距离的有轨电车铁轨、空中架空电缆、路灯杆。
* **强语义特征**：存在车辆、树木、建筑物。
* **面临挑战**：
    * 电缆和杆状物在单帧 LiDAR 中点云稀疏。
    * 树木边缘不规则易产生噪声。

> **核心策略**：
> 本方案融合了 **Line-based** 的几何捕捉能力、**EdgeCalib/Calib-Anything** 的大模型语义理解能力，以及 **CalibRefine** 的由粗到精策略，实现 **零训练（Zero-training）、无标靶（Targetless）** 的在线自动标定。

---

## 2. 详细实施步骤

### 阶段一：图像端——构建“混合边缘势场” (Hybrid Edge Field Generation)

**目标**：结合几何直线与语义轮廓，生成用于引导优化的全局梯度场。

1.  **几何线特征提取 (Geometric Line Extraction)**
    * **算法**：LSD (Line Segment Detector)。
    * **操作**：将图像转为灰度图，提取直线段。
    * **目的**：捕捉 SAM 容易忽略的细微结构（**铁轨、架空电缆**），这些是约束旋转角度（尤其是 Pitch 和 Yaw）的关键特征。

2.  **语义边缘提取 (Semantic Edge Extraction)**
    * **算法**：Segment Anything Model (SAM)。
    * **操作**：使用 Grid Prompt 模式分割全图，提取车辆和建筑物的 Mask 边界。
    * **目的**：提供鲁棒的物体轮廓，过滤掉树叶内部的纹理噪声。

3.  **边缘势场构建 (Distance Transform)**
    * **算法**：距离变换 (Distance Transform)。
    * **操作**：将 LSD 直线与 SAM 边缘合并为二值图 $E_{img}$。计算每个像素到最近边缘的欧氏距离，生成梯度图 $G$。
    * **输出**：一张灰度图，像素越亮（或越暗）代表距离特征线越近，作为优化时的“引力场”。

### 阶段二：点云端——时空增强与特征提取 (Spatiotemporal Enhancement)

**目标**：解决细小物体点云稀疏问题，提取可匹配的边缘与属性。

1.  **多帧点云融合 (Multi-frame Accumulation)**
    * **算法**：NDT (Normal Distributions Transform)。
    * **操作**：维护 3 帧滑动窗口，将 $t-1$ 和 $t-2$ 帧点云通过 NDT 配准转换到当前帧 $t$ 坐标系下（Three-in-one）。
    * **目的**：将单帧中断续的电缆点云连成线，增加铁轨和杆状物的点密度。

2.  **边缘特征提取 (Edge Extraction)**
    * **算法**：深度不连续性检测 (Depth Discontinuity)。
    * **操作**：投影点云为深度图，计算邻域深度跳变，提取边缘点 $P_{edge}$。
    * **分类**：将特征点分为水平特征（约束横向误差）和垂直特征（约束纵向误差）。

3.  **属性计算 (Attribute Estimation)**
    * **算法**：PCA 法向量估计。
    * **操作**：计算非边缘点的法向量 $N_p$ 和反射强度 $I_p$。

### 阶段三：由粗到精的迭代优化 (Coarse-to-Fine Optimization)

**目标**：求解外参 $T = R|t$，使点云特征与图像特征对齐。

1.  **粗标定 (Coarse Search)**
    * **策略**：暴力网格搜索 (Brute Force Search)。
    * **范围**：仅针对旋转角度（Roll, Pitch, Yaw），步长 $0.5^\circ$ - $1^\circ$。
    * **目标函数**：最大化投影点落在 SAM Mask 内部的一致性得分（参考 Calib-Anything 的逻辑，即 Mask 内点云强度方差最小）。

2.  **精细优化 (Fine Optimization)**
    * **策略**：Levenberg-Marquardt (LM) 非线性优化。
    * **Loss 函数设计**：
        $$
        Loss = \sum_{i} W_{frame}^{(i)} \cdot ( \alpha \cdot E_{geo} + \beta \cdot E_{sem} )
        $$
        * **$E_{geo}$ (几何误差)**：点云边缘投影到图像势场 $G$ 中的距离值（Line-based/EdgeCalib 逻辑）。
        * **$E_{sem}$ (语义一致性)**：投影到同一 Mask 内点云的法向量与强度差异（Calib-Anything 逻辑）。
    * **多帧加权 $W_{frame}$**：根据 **位置一致性**（特征是否稳定存在）和 **投影一致性**（投影误差是否在多帧间跳变）动态调整权重，降低动态物体（如行驶电车）的权重。

### 阶段四：结果验证 (Verification)

* **置信度检查 (CCuP)**
    * **算法**：时空平滑性检查。
    * **操作**：如果当前帧解算的参数与历史参数突变（不满足时间平滑性），则丢弃该帧结果，沿用上一帧参数。

---

## 3. 总结：方案流程图

1.  **预处理**：图像 $\rightarrow$ (SAM边缘 + LSD直线) $\rightarrow$ 距离变换场。
2.  **预处理**：点云 $\rightarrow$ NDT多帧叠加 $\rightarrow$ 提取边缘/法向量。
3.  **粗搜**：网格搜索旋转角 $\rightarrow$ 找强度一致性最优值。
4.  **精修**：LM算法 $\rightarrow$ 最小化 (边缘距离 + 法向不一致性) $\times$ 多帧权重。
5.  **输出**：通过 CCuP 检查则输出，否则沿用旧值。

### 关键流程说明

* **并行处理**：图像和点云的处理是并行的。图像端侧重于构建“势场”（Target），点云端侧重于构建“探针”（Source）。
* **LSD 的关键作用**：在流程图中，Algo_LSD 被明确包含，这是为了专门捕捉场景中的 **铁轨和电缆**。如果没有这一步，仅靠 SAM 可能会丢失这些高精度的几何约束。
* **NDT 的必要性**：在 Algo_NDT 步骤中，通过融合多帧数据，使得原本在单帧中看不清的细电缆变成连续的线条，从而能与图像中的 LSD 特征匹配。
* **混合 Loss**：在优化阶段，Loss 不仅仅计算距离（EdgeCalib 逻辑），还加入了 Cal_Sem（Calib-Anything 逻辑），确保投影不仅边缘重合，而且落在车身 Mask 内的点具有相同的反射率和法向量朝向。
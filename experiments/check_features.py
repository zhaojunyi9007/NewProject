import cv2
import numpy as np

# 1. 加载 SAM 提取的纯边缘图 (黑底白边)
sam_edge_img = cv2.imread("result/sam_features/0000000089_edge_map.png")

# 2. 加载优化出的最终外参 (从你的日志中复制的)
r_vec = np.array([1.20155, -1.19339, 1.17018])
t_vec = np.array([-0.00406977, -0.0763162, -0.271781])
R, _ = cv2.Rodrigues(r_vec)

# 3. 加载 KITTI 相机矩阵 (你日志里的)
P_rect = np.array([
    [721.5377, 0., 609.5593, 44.85728],
    [0., 721.5377, 172.854, 0.2163791],
    [0., 0., 1., 0.002745884]
])
R_rect = np.array([
    [0.9999239, 0.00983776, -0.00744505],
    [-0.0098698, 0.9999421, -0.00427846],
    [0.00740253, 0.00435161, 0.9999631]
])

# 4. 加载 LiDAR 专门提取的 edge_points (4列格式)
pts_3d = []
with open("result/lidar_features/0000000089_edge_points.txt", "r") as f:
    for line in f:
        if not line.strip() or line.strip().startswith('#'):
            continue
        pts_3d.append(list(map(float, line.split()[:3])))
pts_3d = np.array(pts_3d)

# 5. 投影运算
pts_c = (R @ pts_3d.T).T + t_vec
pts_rect = (R_rect @ pts_c.T).T
pts_rect = pts_rect[pts_rect[:, 2] > 0.1] # 过滤背后的点
pts_rect_homo = np.hstack([pts_rect, np.ones((pts_rect.shape[0], 1))])
uv_homo = (P_rect @ pts_rect_homo.T).T

u = (uv_homo[:, 0] / uv_homo[:, 2]).astype(np.int32)
v = (uv_homo[:, 1] / uv_homo[:, 2]).astype(np.int32)

# 6. 把雷达边缘点画在 SAM 边缘图上 (用红色和黄色)
for ui, vi in zip(u, v):
    if 0 <= ui < sam_edge_img.shape[1] and 0 <= vi < sam_edge_img.shape[0]:
        cv2.circle(sam_edge_img, (ui, vi), 1, (0, 0, 255), -1)

cv2.imwrite("feature_check_result.png", sam_edge_img)
print("特征透视图已生成：feature_check_result.png")
import cv2
import numpy as np

# 1. 加载原图
img = cv2.imread("/gz-data/dataset/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000089.png")  # 请确保路径正确

# 2. 加载真值外参
r_vec = np.array([1.20135, -1.19308, 1.21013])
t_vec = np.array([-0.00406977, -0.0763162, -0.271781])
R, _ = cv2.Rodrigues(r_vec)

# 3. 加载 KITTI 相机矩阵
P_rect = np.array([[721.5377, 0., 609.5593, 44.85728], [0., 721.5377, 172.854, 0.2163791], [0., 0., 1., 0.002745884]])
R_rect = np.array([[0.9999239, 0.00983776, -0.00744505], [-0.0098698, 0.9999421, -0.00427846], [0.00740253, 0.00435161, 0.9999631]])

# 4. 绘制 SAM 提取的 2D 蓝线 (目标)
with open("result/sam_features/0000000089_lines_2d.txt", "r") as f:
    for line in f:
        if line.startswith("#"): continue
        u1, v1, u2, v2, _ = map(float, line.split())
        cv2.line(img, (int(u1), int(v1)), (int(u2), int(v2)), (255, 0, 0), 2) # 蓝色

# 5. 投影 LiDAR 提取的 3D 绿线/红线 (源)
with open("result/lidar_features/0000000089_lines_3d.txt", "r") as f:
    for line in f:
        if line.startswith("#"): continue
        x1, y1, z1, x2, y2, z2, l_type = map(float, line.split())
        p1_c = R @ np.array([x1, y1, z1]) + t_vec
        p2_c = R @ np.array([x2, y2, z2]) + t_vec
        
        p1_rect = R_rect @ p1_c
        p2_rect = R_rect @ p2_c
        if p1_rect[2] < 0.1 or p2_rect[2] < 0.1: continue # 简单过滤背后点
            
        uv1 = P_rect @ np.hstack([p1_rect, 1.0])
        uv2 = P_rect @ np.hstack([p2_rect, 1.0])
        
        color = (0, 255, 0) if l_type == 0 else (0, 0, 255) # 铁轨绿，立柱红
        cv2.line(img, (int(uv1[0]/uv1[2]), int(uv1[1]/uv1[2])), 
                      (int(uv2[0]/uv2[2]), int(uv2[1]/uv2[2])), color, 4)

cv2.imwrite("line_check_result.png", img)
print("线特征透视图已生成：line_check_result.png")
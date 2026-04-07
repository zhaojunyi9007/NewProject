#ifndef BEV_MATCHER_H
#define BEV_MATCHER_H

#include "include/bev_builder.h"
#include <string>

// Phase 4：LiDAR-BEV 与图像 pseudo-BEV（rail 通道）粗对齐，仅优化 yaw + tx + ty。

struct BEVOptimizeConfig {
    double yaw_min_deg = -6.0;
    double yaw_max_deg = 6.0;
    double yaw_step_deg = 1.0;
    double tx_min_m = -2.0;
    double tx_max_m = 2.0;
    double ty_min_m = -2.0;
    double ty_max_m = 2.0;
    double trans_step_m = 0.5;
};

struct ScoreBreakdown {
    double rail_score = 0.0;
    double pole_score = 0.0;
    double total = 0.0;
};

struct PoseDeltaBev {
    double yaw_rad = 0.0;
    double tx_m = 0.0;
    double ty_m = 0.0;
};

// 使用 rail_probability 栅格：图像 BEV 会 resize 到与 LiDAR 相同尺寸后做旋转+平移搜索，得分=逐像素乘积和。
bool EstimateBEVDelta(
    const BEVChannels& lidar_bev,
    const BEVChannels& image_bev,
    const BEVOptimizeConfig& cfg,
    PoseDeltaBev* out_delta,
    ScoreBreakdown* debug_score);

// 生成调试图：左=LiDAR rail，右=对齐后图像 rail，下=叠加（伪彩）
bool SaveBEVDebugImages(
    const std::string& path_overlay,
    const std::string& path_match,
    const BEVChannels& lidar_bev,
    const BEVChannels& image_bev,
    const PoseDeltaBev& delta);

#endif

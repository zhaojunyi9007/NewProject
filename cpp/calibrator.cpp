#include "include/optimizer_cost_functions.h"
#include "include/optimizer_data_loader.h"
#include "include/optimizer_scoring.h"
#include "include/common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

static CalibHistory g_calib_history;

bool GetEnvBool(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value) return default_value;
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return !(s == "0" || s == "false" || s == "off" || s == "no");
}

double GetEnvDouble(const char* name, double default_value) {
    const char* value = std::getenv(name);
    if (!value) return default_value;
    char* end = nullptr;
    double parsed = std::strtod(value, &end);
    if (end == value) return default_value;
    return parsed;
}

// ========================================
// 主程序
// ========================================
int main(int argc, char** argv) {
    if (argc != 10 && argc != 11 && argc != 12) {
        std::cerr << "Usage: ./optimizer <lidar_feature_base> <sam_feature_base> <calib_file> "
                  << "<init_rx> <init_ry> <init_rz> <init_tx> <init_ty> <init_tz> [output_file] [history_file]" << std::endl;
        
        return -1;
    }
    
    std::string lidar_base = argv[1];
    std::string sam_base = argv[2];
    std::string calib_file = argv[3];
    double r_curr[3] = {std::atof(argv[4]), std::atof(argv[5]), std::atof(argv[6])};
    double t_curr[3] = {std::atof(argv[7]), std::atof(argv[8]), std::atof(argv[9])};
    std::string output_file;
    std::string history_file;
    if (argc == 11) {
        output_file = argv[10];
    } else if (argc == 12) {
        output_file = argv[10];
        history_file = argv[11];
    }

    std::cout << "=== EdgeCalib v2.0 - Two-Stage Calibration ===" << std::endl;
    std::cout << "LiDAR features: " << lidar_base << std::endl;
    std::cout << "SAM features: " << sam_base << std::endl;
    std::cout << "Calib file: " << (calib_file.empty() ? "(default)" : calib_file) << std::endl;
    std::cout << "Initial R: [" << r_curr[0] << ", " << r_curr[1] << ", " << r_curr[2] << "]" << std::endl;
    std::cout << "Initial T: [" << t_curr[0] << ", " << t_curr[1] << ", " << t_curr[2] << "]" << std::endl;
    if (!history_file.empty()) {
        std::cout << "History file: " << history_file << std::endl;
    }

    if (!history_file.empty()) {
        if (LoadCalibHistory(history_file, g_calib_history)) {
            std::cout << "  Loaded history with " << g_calib_history.rotation_history.size()
                      << " entries." << std::endl;
        } else {
            std::cout << "  No existing history loaded." << std::endl;
        }
    }

    // ========================================
    // 1. 加载数据
    // ========================================
    std::cout << "\n[Stage 1] Loading data..." << std::endl;

    // 加载边缘吸引场
    cv::Mat edge_dist = cv::imread(sam_base + "_edge_dist.png", cv::IMREAD_UNCHANGED);
    cv::Mat edge_weight = cv::imread(sam_base + "_edge_weight.png", cv::IMREAD_UNCHANGED);
    
    // 加载语义图（备用）
    cv::Mat semantic_map = cv::imread(sam_base + "_semantic_map.png", cv::IMREAD_UNCHANGED);
    if (semantic_map.empty()) {
        semantic_map = cv::imread(sam_base + "_mask_ids.png", cv::IMREAD_UNCHANGED);
    }
    
    if (edge_dist.empty() && semantic_map.empty()) {
        std::cerr << "[Error] No edge attraction field or semantic map found!" << std::endl;
        return -1;
    }
    
    int W = edge_dist.empty() ? semantic_map.cols : edge_dist.cols;
    int H = edge_dist.empty() ? semantic_map.rows : edge_dist.rows;
    
    if (!edge_dist.empty()) {
        std::cout << "  Loaded edge attraction field: " << W << "x" << H << std::endl;
    } else {
        std::cout << "  Loaded semantic map: " << W << "x" << H << std::endl;
    }
    
    // 加载点特征（含weight）
    std::vector<PointFeature> points;
    if (!IOUtils::LoadPointFeatures(lidar_base + "_points.txt", points)) {
        std::cerr << "[Error] Failed to load point features" << std::endl;
        return -1;
    }
    std::cout << "  Loaded " << points.size() << " point features" << std::endl;

    // 加载专门用于边缘匹配的 edge_points
    std::vector<PointFeature> edge_points = LoadEdgePointsCustom(lidar_base + "_edge_points.txt");
    if (edge_points.empty()) {
        std::cerr << "[Warning] Still failed to load edge points!" << std::endl;
        edge_points = points; // 降级
    } else {
        std::cout << "  Loaded " << edge_points.size() << " edge points (Custom Loader)" << std::endl;
    }
    
    // 加载线特征
    std::vector<Line3D> lines3d;
    IOUtils::LoadLines3D(lidar_base + "_lines_3d.txt", lines3d);
    std::cout << "  Loaded " << lines3d.size() << " 3D line features" << std::endl;
    
    std::vector<Line2D> lines2d;
    IOUtils::LoadLines2D(sam_base + "_lines_2d.txt", lines2d);
    std::cout << "  Loaded " << lines2d.size() << " 2D line features" << std::endl;

    // 加载内参
    Eigen::Matrix3d K; 
    Eigen::Matrix3d R_rect;
    Eigen::Matrix<double, 3, 4> P_rect;
    //LoadCalib(calib_file, K, R_rect);
    bool calib_used_default = false;
    LoadCalib(calib_file, K, R_rect, P_rect, &calib_used_default);

    // 最小调试日志（中文）：
    // 目的：快速确认投影几何主链路的尺寸/内参/外参约定，避免再次出现“看起来发散但原因不明”的问题。
    std::cout << "\n[Debug-Min] Geometry chain summary:" << std::endl;
    std::cout << "  Projection image size (W x H): " << W << " x " << H << std::endl;
    std::cout << "  edge_dist size: "
              << (edge_dist.empty() ? std::string("empty") :
                  (std::to_string(edge_dist.cols) + " x " + std::to_string(edge_dist.rows)))
              << std::endl;
    std::cout << "  semantic_map size: "
              << (semantic_map.empty() ? std::string("empty") :
                  (std::to_string(semantic_map.cols) + " x " + std::to_string(semantic_map.rows)))
              << std::endl;
    std::cout << "  Camera intrinsics K:\n" << K << std::endl;
    std::cout << "  Extrinsic convention used in optimizer: p_cam = R * p_lidar + t (LiDAR -> Camera)" << std::endl;
    if (calib_used_default) {
        std::cout << "  [Debug-Min] calib file unavailable/parse failed, using default intrinsics." << std::endl;
    } else {
        std::cout << "  [Debug-Min] calib loaded from: " << calib_file << std::endl;
    }

    Eigen::Vector3d t_vec_debug(t_curr[0], t_curr[1], t_curr[2]);
    Eigen::Matrix3d R_mat_debug;
    ceres::AngleAxisToRotationMatrix(r_curr, R_mat_debug.data());
    auto proj_stats = CountProjectionStats(points, R_rect, P_rect, R_mat_debug, t_vec_debug, W, H);
    const int projected_success = proj_stats.in_bounds + proj_stats.out_of_bounds;
    std::cout << "  Point stats (initial pose): raw=" << proj_stats.total
              << ", projected_success(z>0)=" << projected_success
              << ", in_image=" << proj_stats.in_bounds << std::endl;

    // ========================================
    // Debug信息输出
    // ========================================

    if (std::getenv("EDGECALIB_DEBUG")) {
        Eigen::Vector3d t_vec(t_curr[0], t_curr[1], t_curr[2]);
        Eigen::Matrix3d R_mat;
        ceres::AngleAxisToRotationMatrix(r_curr, R_mat.data());
        auto stats = CountProjectionStats(points, R_rect, P_rect, R_mat, t_vec, W, H);
        const double in_bounds_ratio = stats.total > 0
                                           ? static_cast<double>(stats.in_bounds) / static_cast<double>(stats.total)
                                           : 0.0;
        const double behind_ratio = stats.total > 0
                                        ? static_cast<double>(stats.behind) / static_cast<double>(stats.total)
                                        : 0.0;
        const double out_of_bounds_ratio = stats.total > 0
                                               ? static_cast<double>(stats.out_of_bounds) / static_cast<double>(stats.total)
                                               : 0.0;
        std::cout << "[Debug] Projection stats (initial pose): total=" << stats.total
                  << ", labeled=" << stats.labeled
                  << ", unlabeled=" << stats.unlabeled
                  << ", in_bounds=" << stats.in_bounds
                  << ", behind=" << stats.behind
                  << ", out_of_bounds=" << stats.out_of_bounds
                  << std::endl;

        std::cout << "[Debug] Projection ratios: in_bounds=" << in_bounds_ratio
                  << ", behind=" << behind_ratio
                  << ", out_of_bounds=" << out_of_bounds_ratio << std::endl;
        if (stats.total > 0 && stats.in_bounds == 0) {
            std::cout << "[Debug][Check] No points project inside the image. "
                         "Initial extrinsic may be far off or intrinsics mismatch."
                      << std::endl;
        } else if (stats.total > 0 && in_bounds_ratio < 0.01) {
            std::cout << "[Debug][Check] Very low in-bounds ratio (<1%). "
                         "Initial extrinsic may be poor."
                      << std::endl;
        }
        if (behind_ratio > 0.5) {
            std::cout << "[Debug][Check] More than 50% of points are behind the camera. "
                         "Check rotation direction or coordinate conventions."
                      << std::endl;
        }
        if (out_of_bounds_ratio > 0.9) {
            std::cout << "[Debug][Check] Most points are out of image bounds. "
                         "Check intrinsics or image size."
                      << std::endl;
        }
        double r_norm = std::sqrt(r_curr[0] * r_curr[0] + r_curr[1] * r_curr[1] + r_curr[2] * r_curr[2]);
        double t_norm = std::sqrt(t_curr[0] * t_curr[0] + t_curr[1] * t_curr[1] + t_curr[2] * t_curr[2]);
        if (r_norm > 1.0) {
            std::cout << "[Debug][Check] Initial rotation angle-axis norm > 1 rad ("
                      << r_norm << "). This may be a large initial rotation." << std::endl;
        }
        if (t_norm > 10.0) {
            std::cout << "[Debug][Check] Initial translation norm > 10m ("
                      << t_norm << "). This may be far from expected." << std::endl;
        }
        std::cout << "[Debug] Intrinsics K:\n" << K << std::endl;
        if (calib_used_default) {
            std::cout << "[Debug][Check] Using default intrinsics (calib file missing or load failed)."
                      << std::endl;
        } else {
            std::cout << "[Debug] Loaded intrinsics from: " << calib_file << std::endl;
        }
        if (!semantic_map.empty()) {
            std::cout << "[Debug] Semantic map max label: " << GetMaxLabel(semantic_map)
                      << ", type=" << semantic_map.type() << std::endl;
        }
        if (!edge_dist.empty()) {
            std::cout << "[Debug] Edge dist type: " << edge_dist.type()
                      << ", size=" << edge_dist.cols << "x" << edge_dist.rows << std::endl;
        }
        if (!edge_weight.empty()) {
            std::cout << "[Debug] Edge weight type: " << edge_weight.type()
                      << ", size=" << edge_weight.cols << "x" << edge_weight.rows << std::endl;
        }
        if (!edge_dist.empty() && !semantic_map.empty()) {
            if (edge_dist.cols != semantic_map.cols || edge_dist.rows != semantic_map.rows) {
                std::cout << "[Debug][Check] Edge dist map size (" << edge_dist.cols << "x" << edge_dist.rows
                          << ") differs from semantic map size (" << semantic_map.cols << "x"
                          << semantic_map.rows << ")." << std::endl;
            }
        }
        if ((W != 0 && H != 0) && (!edge_dist.empty() || !semantic_map.empty())) {
            std::cout << "[Debug] Projection image size W/H: " << W << "x" << H << std::endl;
        }
    }

    std::vector<PointFeature> points_for_opt = points;
    if (std::getenv("EDGECALIB_PREFILTER")) {
        Eigen::Vector3d t_vec(t_curr[0], t_curr[1], t_curr[2]);
        Eigen::Matrix3d R_mat;
        ceres::AngleAxisToRotationMatrix(r_curr, R_mat.data());
        auto filtered = FilterPointsInView(points, R_rect, P_rect, R_mat, t_vec, W, H);
        if (!filtered.empty()) {
            std::cout << "[Info] Prefiltered points in view: " << filtered.size()
                      << " / " << points.size() << std::endl;
            points_for_opt = std::move(filtered);
        } else {
            std::cout << "[Warning] Prefiltering removed all points; using original set."
                      << std::endl;
        }
    }

    // ========================================
    // 2. 粗标定 (Coarse Search)
    // ========================================
    std::cout << "\n[Stage 2] Coarse Calibration (Grid Search)..." << std::endl;
    
    // 根据README2.0.md第三阶段要求：优先使用语义方差评分（Calib-Anything逻辑）
    bool use_edge = !edge_dist.empty();
    bool use_semantic = !semantic_map.empty();
    
    if (use_edge) {
        std::cout << "  Strategy: Edge Attraction Field (EdgeCalib)" << std::endl;
        use_semantic = false; // 如果有边缘场，粗搜阶段禁用语义搜索
    } else if (use_semantic) {
        std::cout << "  Strategy: Mask Intensity Variance Minimization (Calib-Anything)" << std::endl;
    } else {
        std::cerr << "[Error] No scoring method available!" << std::endl;
        return -1;
    }

   // 先计算初始位置的得分，以此为基准，避免直接接受错误边界    
    double best_r[3] = {r_curr[0], r_curr[1], r_curr[2]};
    double best_t[3] = {t_curr[0], t_curr[1], t_curr[2]};
    Eigen::Vector3d init_t_vec(t_curr[0], t_curr[1], t_curr[2]);
    Eigen::Matrix3d init_R_mat;
    ceres::AngleAxisToRotationMatrix(r_curr, init_R_mat.data());
    
    double best_score = -1e8;
    if (use_edge) {
        // 【修改 3.1】：将 points_for_opt 替换为 edge_points
        best_score = EdgeAttractionScore(edge_points, edge_dist, edge_weight, R_rect, P_rect, W, H, init_R_mat, init_t_vec);
    } else if (use_semantic) {
        best_score = MaskIntensityVarianceScore(points_for_opt, semantic_map, R_rect, P_rect, W, H, init_R_mat, init_t_vec);
    }

    std::cout << "  Initial Pose Score: " << best_score << std::endl;
    if (best_score <= -1e5) {
        std::cout << "  [Warning] Initial score is very low (-1e6). Points might not project into valid areas or visible count < 50." << std::endl;
    }
    
    const double angle_range = 0.15;  // radians, ±8.6 degrees
    const double angle_step = 0.01;   // radians, ~0.57 degrees
    const double coarse_ty_range = GetEnvDouble("EDGECALIB_COARSE_TY_RANGE", 0.0);
    const double coarse_tz_range = GetEnvDouble("EDGECALIB_COARSE_TZ_RANGE", 0.0);
    const double coarse_tx_range = GetEnvDouble("EDGECALIB_COARSE_TX_RANGE", 0.0);
    const double coarse_ty_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_TY_STEP", 0.05));
    const double coarse_tz_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_TZ_STEP", 0.05));
    const double coarse_tx_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_TX_STEP", 0.05));
    // 中文说明：
    // 目的：做“最小范围”的 tx 消融验证，检查是否因 tx 未搜索而困在错误 basin。
    // 默认 range=0，行为与原始主流程完全一致，不改变默认搜索空间。
    std::vector<double> tx_candidates{t_curr[0]};
    std::vector<double> ty_candidates{t_curr[1]};
    std::vector<double> tz_candidates{t_curr[2]};
    if (coarse_tx_range > 1e-9) {
        tx_candidates.clear();
        for (double tx = t_curr[0] - coarse_tx_range; tx <= t_curr[0] + coarse_tx_range + 1e-12; tx += coarse_tx_step) {
            tx_candidates.push_back(tx);
        }
    }
    if (coarse_ty_range > 1e-9) {
        ty_candidates.clear();
        for (double ty = t_curr[1] - coarse_ty_range; ty <= t_curr[1] + coarse_ty_range + 1e-12; ty += coarse_ty_step) {
            ty_candidates.push_back(ty);
        }
    }
    if (coarse_tz_range > 1e-9) {
        tz_candidates.clear();
        for (double tz = t_curr[2] - coarse_tz_range; tz <= t_curr[2] + coarse_tz_range + 1e-12; tz += coarse_tz_step) {
            tz_candidates.push_back(tz);
        }
    }

    int iter = 0;
    const int steps = static_cast<int>(std::floor((2.0 * angle_range) / angle_step)) + 1;
    const int total_iters = steps * steps * steps
                          * static_cast<int>(tx_candidates.size())
                          * static_cast<int>(ty_candidates.size())
                          * static_cast<int>(tz_candidates.size());
    double best_score_tx_fixed = -1e8;
    
    std::cout << "  Search space: " << steps << "^3 = " << total_iters << " candidates" << std::endl;
    std::cout << "  Angle range: ±" << (angle_range * 180.0 / M_PI) << " degrees" << std::endl;

    std::cout << "  TX candidates: " << tx_candidates.size()
              << ", TY candidates: " << ty_candidates.size()
              << ", TZ candidates: " << tz_candidates.size() << std::endl;

    for (double tx : tx_candidates) {
        for (double ty : ty_candidates) {
            for (double tz : tz_candidates) {
                for (double rx = r_curr[0] - angle_range; rx <= r_curr[0] + angle_range + 1e-9; rx += angle_step) {
                    for (double ry = r_curr[1] - angle_range; ry <= r_curr[1] + angle_range + 1e-9; ry += angle_step) {
                        for (double rz = r_curr[2] - angle_range; rz <= r_curr[2] + angle_range + 1e-9; rz += angle_step) {
                            double r_try[3] = {rx, ry, rz};
                            double t_try[3] = {tx, ty, tz};

                            Eigen::Vector3d t_vec(t_try[0], t_try[1], t_try[2]);
                            Eigen::Matrix3d R_mat;
                            ceres::AngleAxisToRotationMatrix(r_try, R_mat.data());

                            double score = -1e6;
                            if (use_edge) {
                                score = EdgeAttractionScore(edge_points, edge_dist, edge_weight, R_rect, P_rect, W, H, R_mat, t_vec);
                            } else if (use_semantic) {
                                score = MaskIntensityVarianceScore(points_for_opt, semantic_map, R_rect, P_rect, W, H, R_mat, t_vec);
                            }


                            // 用于与“原始tx固定版本”做最小对比（同一轮搜索内统计）
                            if (std::abs(tx - t_curr[0]) < 1e-12 && score > best_score_tx_fixed) {
                                best_score_tx_fixed = score;
                            }

                            // 【修复核心 2】：只有得分有效 (大于-1e5) 且优于当前最好成绩，才更新
                            if (score > best_score && score > -1e5) {
                                best_score = score;
                                std::copy(r_try, r_try + 3, best_r);
                                std::copy(t_try, t_try + 3, best_t);
                            }

                            iter++;
                            if (iter % 2000 == 0) {
                                std::cout << "  Progress: " << iter << "/" << total_iters
                                          << " (" << (iter * 100 / total_iters) << "%), Best Score: " << best_score << std::endl;
                            }


                        }
                    }
                }
            }
        }
    }
    
    std::cout << "  Coarse Search Complete. Best Score: " << best_score << std::endl;
    if (tx_candidates.size() > 1) {
        std::cout << "  Coarse Compare (tx fixed vs tx search): fixed_tx_best=" << best_score_tx_fixed
                  << ", tx_search_best=" << best_score
                  << ", delta=" << (best_score - best_score_tx_fixed) << std::endl;
    }
    std::cout << "  R: [" << best_r[0] << ", " << best_r[1] << ", " << best_r[2] << "]" << std::endl;
    std::cout << "  T: [" << best_t[0] << ", " << best_t[1] << ", " << best_t[2] << "]" << std::endl;
    
    std::copy(best_r, best_r+3, r_curr);
    std::copy(best_t, best_t+3, t_curr);

    // ========================================
    // 3. 精细优化 (Fine Refinement) - README2.0.md 第三阶段
    // ========================================
    if (!edge_dist.empty() || !semantic_map.empty()) {
        std::cout << "\n[Stage 3] Fine Calibration (Ceres LM Optimization)..." << std::endl;
        ceres::Problem problem;
        const bool optimize_translation = GetEnvBool("EDGECALIB_OPT_TRANSLATION", true);
        const bool use_line_constraints = GetEnvBool("EDGECALIB_USE_LINE_CONSTRAINT", true);
        const bool log_line_debug = GetEnvBool("EDGECALIB_LOG_LINE_DEBUG", false);
        const double line_match_threshold = GetEnvDouble("EDGECALIB_LINE_MATCH_THRESHOLD", 50.0);
        const bool line_soft_penalty = GetEnvBool("EDGECALIB_LINE_SOFT_PENALTY", false);
        const double line_soft_cap = GetEnvDouble("EDGECALIB_LINE_SOFT_CAP", 100.0);
        const double line_behind_penalty = GetEnvDouble("EDGECALIB_LINE_BEHIND_PENALTY", 0.0);
        const double line_unmatched_penalty = GetEnvDouble("EDGECALIB_LINE_UNMATCHED_PENALTY", 0.0);
        const double line_threshold_fail_penalty = GetEnvDouble("EDGECALIB_LINE_THRESHOLD_FAIL_PENALTY", 0.0);
        const double t_prior_weight = GetEnvDouble("EDGECALIB_T_PRIOR_WEIGHT", 20.0);
        const double w_consistency_env = GetEnvDouble("EDGECALIB_W_CONSISTENCY", 0.0);
        const Eigen::Vector3d t_init_prior(t_curr[0], t_curr[1], t_curr[2]);
        
        // 采样点云以提高效率（最多5000个点）
        std::vector<int> sample_indices;
        sample_indices.reserve(edge_points.size());
        const int stride = std::max<int>(1, static_cast<int>(edge_points.size() / 5000));
        for (size_t i = 0; i < edge_points.size(); i += stride) {
            sample_indices.push_back(static_cast<int>(i));
        }
        std::cout << "  Sampled " << sample_indices.size() << " points from " << edge_points.size() << " edge points" << std::endl;

        // 混合Loss权重：α * E_geo + β * E_sem (README2.0.md公式)
        const double w_edge = 1.0;         // α: 几何误差权重
        const double w_consistency = std::max(0.0, w_consistency_env);  // β: 语义一致性权重
        std::cout << "  Loss weights: w_edge=" << w_edge << ", w_consistency=" << w_consistency << std::endl;
        std::vector<LabelStats> label_stats;
        if (!semantic_map.empty()) {
            Eigen::Vector3d t_vec(t_curr[0], t_curr[1], t_curr[2]);
            Eigen::Matrix3d R_mat;
            ceres::AngleAxisToRotationMatrix(r_curr, R_mat.data());
            label_stats = ComputeLabelStats(points_for_opt, semantic_map, R_rect, P_rect, W, H, R_mat, t_vec);
        }

        int valid_points_added = 0;
        for (int idx : sample_indices) {
            const auto& pt = edge_points[idx];
            auto* single_cost = new SinglePointEdgeCost(
                pt,
                edge_dist.empty() ? nullptr : &edge_dist,
                R_rect,
                P_rect,
                W,
                H
            );
            
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SinglePointEdgeCost, 1, 3, 3>(single_cost),
                new ceres::HuberLoss(0.1), // 阈值缩小到0.1，防噪点
                r_curr,
                t_curr
            );
            valid_points_added++;
        }
        std::cout << "  Added " << valid_points_added << " individual residual blocks to Ceres." << std::endl;
        

        //if (!lines3d.empty() && !lines2d.empty()) {
        if (use_line_constraints && !lines3d.empty() && !lines2d.empty()) {
            std::cout << "  Adding line reprojection constraints: " << lines3d.size() << " lines" << std::endl;
            int line_blocks_added = 0;
            const LineConstraintSummary init_line_summary = SummarizeLineConstraints(
                lines3d, lines2d, R_rect, P_rect, r_curr, t_curr, line_match_threshold);
            for (const auto& l3d : lines3d) {
                auto* line_cost = new LineReprojectionError(
                    l3d,
                    lines2d,
                    R_rect,
                    P_rect,
                    line_match_threshold,
                    line_soft_penalty,
                    line_soft_cap,
                    line_behind_penalty,
                    line_unmatched_penalty,
                    line_threshold_fail_penalty);                    
                problem.AddResidualBlock(
                    new ceres::NumericDiffCostFunction<LineReprojectionError, ceres::CENTRAL, 1, 3, 3>(line_cost),
                    new ceres::HuberLoss(1.0),
                    r_curr,
                    t_curr);
                line_blocks_added++;
                if (log_line_debug) {
                    LineMatchStats stats = EvaluateLineMatchStats(l3d, lines2d, R_rect, P_rect, r_curr, t_curr, line_match_threshold);
                    const char* reason = GetLineStatusReason(stats, line_match_threshold);
                    std::cout << "    [LineDebug] type=" << l3d.type
                              << ", in_front=" << (stats.in_front ? "Y" : "N")
                              << ", found_type_match=" << (stats.found_type_match ? "Y" : "N")
                              << ", min_dist=" << stats.min_dist
                              << ", active=" << (stats.active ? "Y" : "N")
                              << ", reason=" << reason
                              << std::endl;
                }
            }

            std::cout << "  Line constraints added: " << line_blocks_added
                      << ", active at init: " << init_line_summary.active << std::endl;
            PrintLineConstraintSummary("init", init_line_summary, line_match_threshold);
                      
            std::cout << "  Line match threshold: " << line_match_threshold << " px" << std::endl;
            std::cout << "  Line soft penalty: " << (line_soft_penalty ? "enabled" : "disabled")
                      << " (cap=" << line_soft_cap << ")" << std::endl;
            std::cout << "  Line hard penalties (when soft disabled): behind=" << line_behind_penalty
                      << ", unmatched=" << line_unmatched_penalty
                      << ", threshold_fail=" << line_threshold_fail_penalty << std::endl;         
        } else if (!use_line_constraints) {
            std::cout << "  Line constraints: disabled by EDGECALIB_USE_LINE_CONSTRAINT" << std::endl;
        }

        if (t_prior_weight > 0.0) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<TranslationPriorCost, 3, 3>(
                    new TranslationPriorCost(t_init_prior, t_prior_weight)),
                nullptr,
                t_curr);
            std::cout << "  Translation prior: enabled (weight=" << t_prior_weight << ")" << std::endl;
        } else {
            std::cout << "  Translation prior: disabled" << std::endl;

        }
        

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 100;

        //problem.SetParameterBlockConstant(t_curr);
        //std::cout << "  Translation optimization: enabled" << std::endl;
        if (!optimize_translation) {
            problem.SetParameterBlockConstant(t_curr);
            std::cout << "  Translation optimization: disabled by EDGECALIB_OPT_TRANSLATION" << std::endl;
        } else {
            std::cout << "  Translation optimization: enabled" << std::endl;
        }
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        std::cout << "  " << summary.BriefReport() << std::endl;
        std::cout << "  Final R: [" << r_curr[0] << ", " << r_curr[1] << ", " << r_curr[2] << "]" << std::endl;
        std::cout << "  Final T: [" << t_curr[0] << ", " << t_curr[1] << ", " << t_curr[2] << "]" << std::endl;
        if (use_line_constraints && !lines3d.empty() && !lines2d.empty()) {
            const LineConstraintSummary final_line_summary = SummarizeLineConstraints(
                lines3d, lines2d, R_rect, P_rect, r_curr, t_curr, line_match_threshold);
            PrintLineConstraintSummary("final", final_line_summary, line_match_threshold);
        }
    } else {
        std::cout << "\n[Stage 3] Skipped (no edge field or semantic map for fine optimization)" << std::endl;
    }

    // ========================================
    // 4. 置信度检查与时序平滑 (README2.0 第三阶段 3.3)
    // ========================================
    std::cout << "\n[Stage 4] Temporal Validation & Smoothing..." << std::endl;
    
    Eigen::Vector3d r_result(r_curr[0], r_curr[1], r_curr[2]);
    Eigen::Vector3d t_result(t_curr[0], t_curr[1], t_curr[2]);
    
    const double SMOOTHNESS_THRESHOLD = 0.05;  // 平滑性阈值
    
    if (!g_calib_history.is_smooth(r_result, t_result, SMOOTHNESS_THRESHOLD)) {
        std::cout << "  [Warning] Result not smooth! Applying temporal smoothing..." << std::endl;
        
        auto smoothed = g_calib_history.get_smoothed();
        r_result = smoothed.first;
        t_result = smoothed.second;
        
        std::cout << "  Smoothed R: [" << r_result[0] << ", " << r_result[1] << ", " << r_result[2] << "]" << std::endl;
        std::cout << "  Smoothed T: [" << t_result[0] << ", " << t_result[1] << ", " << t_result[2] << "]" << std::endl;
    } else {
        std::cout << "  Result passes smoothness check." << std::endl;
    }
    
    // 更新历史记录
    g_calib_history.push(r_result, t_result, best_score);
    if (!history_file.empty()) {
        if (!SaveCalibHistory(history_file, g_calib_history)) {
            std::cerr << "[Warning] Failed to save calibration history to: " << history_file << std::endl;
        }
    }


    // ========================================
    // 5. 保存结果
    // ========================================
    std::cout << "\n[Stage 5] Saving results..." << std::endl;
    
    // 从lidar_base路径提取帧ID，保存到calib目录
    if (output_file.empty()) {
        output_file = lidar_base + "_calib_result.txt";
        
        // 替换lidar_features为calibration
        size_t pos = output_file.find("lidar_features");
        if (pos != std::string::npos) {
            output_file.replace(pos, 14, "calibration");
        }
    }
    
    std::ofstream result_file(output_file);
    if (result_file.is_open()) {
        result_file << "# EdgeCalib v2.0 Calibration Result\n";
        result_file << r_result[0] << " " << r_result[1] << " " << r_result[2] << "\n";
        result_file << t_result[0] << " " << t_result[1] << " " << t_result[2] << "\n";
        result_file << "# Score: " << best_score << "\n";
        result_file.close();
        std::cout << "  Result saved to: " << output_file << std::endl;
    } else {
        std::cerr << "[Error] Cannot write result file: " << output_file << std::endl;
    }

    std::cout << "\n=== Calibration Complete ===" << std::endl;
    std::cout << "Final Extrinsic Parameters:" << std::endl;
    std::cout << "  R (angle-axis): [" << r_result[0] << ", " << r_result[1] << ", " << r_result[2] << "]" << std::endl;
    std::cout << "  T (meters): [" << t_result[0] << ", " << t_result[1] << ", " << t_result[2] << "]" << std::endl;

    return 0;
}

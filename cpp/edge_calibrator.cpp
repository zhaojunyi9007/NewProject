#include "include/edge_calibrator.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "include/optimizer_cost_function.h"
#include "include/optimizer_scoring.h"
#include "include/optimizer_semantic_scoring.h"

namespace {
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

void PrintProjectionStats(const char* tag,
                          const std::vector<PointFeature>& points,
                          const Eigen::Matrix3d& R_rect,
                          const Eigen::Matrix<double, 3, 4>& P_rect,
                          const double* r,
                          const double* t,
                          int W, int H) {
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(r, R.data());
    Eigen::Vector3d t_vec(t[0], t[1], t[2]);
    ProjectionDebugStats stats = CountProjectionStats(points, R_rect, P_rect, R, t_vec, W, H);
    const int projected = stats.in_bounds + stats.out_of_bounds;
    const double total = std::max(1, stats.total);
    const double projected_ratio = static_cast<double>(projected) / total;
    const double in_bounds_ratio = static_cast<double>(stats.in_bounds) / total;
    const double behind_ratio = static_cast<double>(stats.behind) / total;
    const double oob_ratio = static_cast<double>(stats.out_of_bounds) / total;
    std::cout << "[Debug][Proj][" << tag << "] total=" << stats.total
              << ", projected(z>0)=" << projected
              << " (" << projected_ratio << ")"
              << ", in_image=" << stats.in_bounds
              << " (" << in_bounds_ratio << ")"
              << ", behind=" << stats.behind
              << " (" << behind_ratio << ")"
              << ", out_of_bounds=" << stats.out_of_bounds
              << " (" << oob_ratio << ")"
              << std::endl;
}
}  // namespace

EdgeCalibrator::EdgeCalibrator(const EdgeCalibratorConfig& config) : config_(config) {
    std::copy(config_.init_r, config_.init_r + 3, r_curr_);
    std::copy(config_.init_t, config_.init_t + 3, t_curr_);
    sem_cfg_.class_weights = config_.class_weights;
    sem_cfg_.pyramid_scales = config_.pyramid_scales;
}

bool EdgeCalibrator::LoadData() {
    std::cout << "[Stage 1] Loading data..." << std::endl;
    edge_dist_ = cv::imread(config_.sam_base + "_edge_dist.png", cv::IMREAD_UNCHANGED);
    edge_weight_ = cv::imread(config_.sam_base + "_edge_weight.png", cv::IMREAD_UNCHANGED);
    semantic_map_ = cv::imread(config_.sam_base + "_semantic_map.png", cv::IMREAD_UNCHANGED);
    if (semantic_map_.empty()) semantic_map_ = cv::imread(config_.sam_base + "_mask_ids.png", cv::IMREAD_UNCHANGED);
    if (edge_dist_.empty() && semantic_map_.empty()) return false;

    W_ = edge_dist_.empty() ? semantic_map_.cols : edge_dist_.cols;
    H_ = edge_dist_.empty() ? semantic_map_.rows : edge_dist_.rows;

    if (!IOUtils::LoadPointFeatures(config_.lidar_base + "_points.txt", points_)) return false;
    edge_points_ = LoadEdgePointsCustom(config_.lidar_base + "_edge_points.txt");
    if (edge_points_.empty()) edge_points_ = points_;
    IOUtils::LoadLines3D(config_.lidar_base + "_lines_3d.txt", lines3d_);
    IOUtils::LoadLines2D(config_.sam_base + "_lines_2d.txt", lines2d_);

    bool used_default = false;
    if (!LoadCalib(config_.calib_file, K_, R_rect_, P_rect_, &used_default)) return false;

    // Phase B5: semantic-probability inputs
    semantic_inputs_ready_ = false;
    if (!config_.semantic_probs_path.empty() && !config_.lidar_semantic_points_path.empty()) {
        if (LoadSemanticProbabilityMaps(config_.semantic_probs_path, semantic_probs_) &&
            LoadSemanticPoints(config_.lidar_semantic_points_path, semantic_points_) &&
            !semantic_probs_.empty() && !semantic_points_.empty()) {
            semantic_inputs_ready_ = true;
            std::cout << "[Info] Semantic inputs ready: probs=" << semantic_probs_.W << "x" << semantic_probs_.H
                      << "x" << semantic_probs_.C << ", points=" << semantic_points_.size() << std::endl;
        } else {
            std::cout << "[Warning] Semantic inputs specified but failed to load." << std::endl;
        }
    }

    if (!config_.history_file.empty()) {
        LoadCalibHistory(config_.history_file, history_);
    }
    std::cout << "[Info] Feature counts: points=" << points_.size()
              << ", edge_points=" << edge_points_.size()
              << ", lines2d=" << lines2d_.size()
              << ", lines3d=" << lines3d_.size() << std::endl;
    return true;
}

void EdgeCalibrator::ApplyPoseFromBEVIfProvided() {
    if (config_.init_pose_from_bev_path.empty()) {
        return;
    }
    double r_bev[3] = {0, 0, 0};
    double t_bev[3] = {0, 0, 0};
    if (LoadInitPoseFromBEV(config_.init_pose_from_bev_path, r_bev, t_bev)) {
        r_curr_[0] = r_bev[0];
        r_curr_[1] = r_bev[1];
        r_curr_[2] = r_bev[2];
        t_curr_[0] = t_bev[0];
        t_curr_[1] = t_bev[1];
        t_curr_[2] = t_bev[2];
        std::cout << "[Info] Applied init pose from BEV: r=[" << r_curr_[0] << "," << r_curr_[1] << "," << r_curr_[2]
                  << "], t=[" << t_curr_[0] << "," << t_curr_[1] << "," << t_curr_[2] << "]" << std::endl;
    }
}

void EdgeCalibrator::PerformSemanticCoarseOptimizationIfEnabled() {
    if (!semantic_inputs_ready_) {
        return;
    }
    std::cout << "[Stage 2] Semantic coarse optimization (dominant scoring)..." << std::endl;

    const double angle_range = GetEnvDouble("EDGECALIB_COARSE_ANGLE_RANGE", 0.15);
    const double angle_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_ANGLE_STEP", 0.05));
    double tx_range = GetEnvDouble("EDGECALIB_COARSE_TX_RANGE", 0.10);
    double ty_range = GetEnvDouble("EDGECALIB_COARSE_TY_RANGE", 0.30);
    double tz_range = GetEnvDouble("EDGECALIB_COARSE_TZ_RANGE", 0.30);
    double tx_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_TX_STEP", 0.05));
    double ty_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_TY_STEP", 0.15));
    double tz_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_TZ_STEP", 0.15));

    double best_r[3] = {r_curr_[0], r_curr_[1], r_curr_[2]};
    double best_t[3] = {t_curr_[0], t_curr_[1], t_curr_[2]};

    Eigen::Matrix3d R0;
    ceres::AngleAxisToRotationMatrix(r_curr_, R0.data());
    Eigen::Vector3d t0(t_curr_[0], t_curr_[1], t_curr_[2]);

    const double w_edge = config_.edge_weight;
    const double w_line = config_.line_weight;

    best_score_ = ComputeTotalCalibrationScoreSemanticDominant(edge_points_, edge_dist_, edge_weight_, lines3d_, lines2d_, semantic_points_,
                                                              semantic_probs_, R_rect_, P_rect_, W_, H_, R0, t0,
                                                              config_.semantic_js_weight, config_.histogram_weight,
                                                              w_edge, w_line, sem_cfg_, &last_score_breakdown_);
    PrintProjectionStats("initial", edge_points_, R_rect_, P_rect_, r_curr_, t_curr_, W_, H_);

    for (double tx = t_curr_[0] - tx_range; tx <= t_curr_[0] + tx_range + 1e-12; tx += tx_step) {
        for (double ty = t_curr_[1] - ty_range; ty <= t_curr_[1] + ty_range + 1e-12; ty += ty_step) {
            for (double tz = t_curr_[2] - tz_range; tz <= t_curr_[2] + tz_range + 1e-12; tz += tz_step) {
                for (double rx = r_curr_[0] - angle_range; rx <= r_curr_[0] + angle_range + 1e-9; rx += angle_step) {
                    for (double ry = r_curr_[1] - angle_range; ry <= r_curr_[1] + angle_range + 1e-9; ry += angle_step) {
                        for (double rz = r_curr_[2] - angle_range; rz <= r_curr_[2] + angle_range + 1e-9; rz += angle_step) {
                            double r_try[3] = {rx, ry, rz};
                            Eigen::Matrix3d R_try;
                            ceres::AngleAxisToRotationMatrix(r_try, R_try.data());
                            Eigen::Vector3d t_try(tx, ty, tz);
                            TotalScoreBreakdown bd;
                            const double score =
                                ComputeTotalCalibrationScoreSemanticDominant(edge_points_, edge_dist_, edge_weight_, lines3d_, lines2d_, semantic_points_,
                                                                             semantic_probs_, R_rect_, P_rect_, W_, H_,
                                                                             R_try, t_try, config_.semantic_js_weight,
                                                                             config_.histogram_weight, w_edge, w_line,
                                                                             sem_cfg_, &bd);
                            if (score > best_score_) {
                                best_score_ = score;
                                std::copy(r_try, r_try + 3, best_r);
                                best_t[0] = tx;
                                best_t[1] = ty;
                                best_t[2] = tz;
                                last_score_breakdown_ = bd;
                            }
                        }
                    }
                }
            }
        }
    }

    std::copy(best_r, best_r + 3, r_curr_);
    std::copy(best_t, best_t + 3, t_curr_);
    PrintProjectionStats("after_semantic_coarse", edge_points_, R_rect_, P_rect_, r_curr_, t_curr_, W_, H_);
}

void EdgeCalibrator::PerformSemanticFineOptimizationIfEnabled() {
    if (!semantic_inputs_ready_) {
        return;
    }
    std::cout << "[Stage 2b] Semantic fine optimization (local search)..." << std::endl;

    const double a_range = GetEnvDouble("EDGECALIB_SEM_FINE_ANGLE_RANGE", 0.04);
    const double a_step = std::max(1e-6, GetEnvDouble("EDGECALIB_SEM_FINE_ANGLE_STEP", 0.02));
    const double t_range = GetEnvDouble("EDGECALIB_SEM_FINE_TRANS_RANGE", 0.08);
    const double t_step = std::max(1e-6, GetEnvDouble("EDGECALIB_SEM_FINE_TRANS_STEP", 0.04));

    const double w_edge = config_.edge_weight;
    const double w_line = config_.line_weight;

    double best_r[3] = {r_curr_[0], r_curr_[1], r_curr_[2]};
    double best_t[3] = {t_curr_[0], t_curr_[1], t_curr_[2]};
    double local_best = best_score_;

    for (double tx = t_curr_[0] - t_range; tx <= t_curr_[0] + t_range + 1e-12; tx += t_step) {
        for (double ty = t_curr_[1] - t_range; ty <= t_curr_[1] + t_range + 1e-12; ty += t_step) {
            for (double tz = t_curr_[2] - t_range; tz <= t_curr_[2] + t_range + 1e-12; tz += t_step) {
                for (double rx = r_curr_[0] - a_range; rx <= r_curr_[0] + a_range + 1e-9; rx += a_step) {
                    for (double ry = r_curr_[1] - a_range; ry <= r_curr_[1] + a_range + 1e-9; ry += a_step) {
                        for (double rz = r_curr_[2] - a_range; rz <= r_curr_[2] + a_range + 1e-9; rz += a_step) {
                            double r_try[3] = {rx, ry, rz};
                            Eigen::Matrix3d R_try;
                            ceres::AngleAxisToRotationMatrix(r_try, R_try.data());
                            Eigen::Vector3d t_try(tx, ty, tz);
                            TotalScoreBreakdown bd;
                            const double score =
                                ComputeTotalCalibrationScoreSemanticDominant(edge_points_, edge_dist_, edge_weight_, lines3d_, lines2d_, semantic_points_,
                                                                             semantic_probs_, R_rect_, P_rect_, W_, H_,
                                                                             R_try, t_try, config_.semantic_js_weight,
                                                                             config_.histogram_weight, w_edge, w_line,
                                                                             sem_cfg_, &bd);
                            if (score > local_best) {
                                local_best = score;
                                std::copy(r_try, r_try + 3, best_r);
                                best_t[0] = tx;
                                best_t[1] = ty;
                                best_t[2] = tz;
                                last_score_breakdown_ = bd;
                            }
                        }
                    }
                }
            }
        }
    }
    best_score_ = local_best;
    std::copy(best_r, best_r + 3, r_curr_);
    std::copy(best_t, best_t + 3, t_curr_);
    PrintProjectionStats("after_semantic_fine", edge_points_, R_rect_, P_rect_, r_curr_, t_curr_, W_, H_);
}

void EdgeCalibrator::PerformCoarseSearch() {
    std::cout << "[Stage 2] Coarse Calibration..." << std::endl;
    // Phase B5: apply BEV init pose first if provided.
    ApplyPoseFromBEVIfProvided();
    // Phase B5: semantic coarse optimization becomes the primary driver when inputs exist.
    PerformSemanticCoarseOptimizationIfEnabled();
    if (semantic_inputs_ready_) {
        return;
    }
    const bool use_edge = !edge_dist_.empty();
    const bool use_semantic = !semantic_map_.empty() && !use_edge;

    double best_r[3] = {r_curr_[0], r_curr_[1], r_curr_[2]};
    double best_t[3] = {t_curr_[0], t_curr_[1], t_curr_[2]};

    Eigen::Matrix3d init_R;
    ceres::AngleAxisToRotationMatrix(r_curr_, init_R.data());
    Eigen::Vector3d init_t(t_curr_[0], t_curr_[1], t_curr_[2]);

    best_score_ = use_edge
        ? EdgeAttractionScore(edge_points_, edge_dist_, edge_weight_, R_rect_, P_rect_, W_, H_, init_R, init_t)
        : MaskIntensityVarianceScore(points_, semantic_map_, R_rect_, P_rect_, W_, H_, init_R, init_t);
    PrintProjectionStats("initial", edge_points_, R_rect_, P_rect_, r_curr_, t_curr_, W_, H_);

    const double angle_range = GetEnvDouble("EDGECALIB_COARSE_ANGLE_RANGE", 0.15);
    const double angle_step  = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_ANGLE_STEP", 0.05));
    double tx_range = GetEnvDouble("EDGECALIB_COARSE_TX_RANGE", 0.0);
    double ty_range = GetEnvDouble("EDGECALIB_COARSE_TY_RANGE", 0.0);
    double tz_range = GetEnvDouble("EDGECALIB_COARSE_TZ_RANGE", 0.0);
    double tx_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_TX_STEP", 0.05));
    double ty_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_TY_STEP", 0.15));
    double tz_step = std::max(1e-6, GetEnvDouble("EDGECALIB_COARSE_TZ_STEP", 0.15));

    // Per-dimension fallback: each axis independently gets a minimum search
    // range so the coarse search can always escape poor translation basins.
    if (tx_range <= 1e-9) tx_range = 0.10;
    if (ty_range <= 1e-9) ty_range = 0.30;
    if (tz_range <= 1e-9) tz_range = 0.30;

    std::vector<double> tx_candidates;
    std::vector<double> ty_candidates;
    std::vector<double> tz_candidates;
    for (double tx = t_curr_[0] - tx_range; tx <= t_curr_[0] + tx_range + 1e-12; tx += tx_step) tx_candidates.push_back(tx);
    for (double ty = t_curr_[1] - ty_range; ty <= t_curr_[1] + ty_range + 1e-12; ty += ty_step) ty_candidates.push_back(ty);
    for (double tz = t_curr_[2] - tz_range; tz <= t_curr_[2] + tz_range + 1e-12; tz += tz_step) tz_candidates.push_back(tz);
    if (tx_candidates.empty()) tx_candidates.push_back(t_curr_[0]);
    if (ty_candidates.empty()) ty_candidates.push_back(t_curr_[1]);
    if (tz_candidates.empty()) tz_candidates.push_back(t_curr_[2]);

    std::cout << "[Debug][Coarse] Translation search candidates: tx=" << tx_candidates.size()
              << ", ty=" << ty_candidates.size() << ", tz=" << tz_candidates.size() << std::endl;

    for (double tx : tx_candidates) {
        for (double ty : ty_candidates) {
            for (double tz : tz_candidates) {
                for (double rx = r_curr_[0] - angle_range; rx <= r_curr_[0] + angle_range + 1e-9; rx += angle_step) {
                    for (double ry = r_curr_[1] - angle_range; ry <= r_curr_[1] + angle_range + 1e-9; ry += angle_step) {
                        for (double rz = r_curr_[2] - angle_range; rz <= r_curr_[2] + angle_range + 1e-9; rz += angle_step) {
                            double r_try[3] = {rx, ry, rz};
                            Eigen::Matrix3d R_try;
                            ceres::AngleAxisToRotationMatrix(r_try, R_try.data());
                            Eigen::Vector3d t_try(tx, ty, tz);
                            double score = use_edge
                                ? EdgeAttractionScore(edge_points_, edge_dist_, edge_weight_, R_rect_, P_rect_, W_, H_, R_try, t_try)
                                : (use_semantic ? MaskIntensityVarianceScore(points_, semantic_map_, R_rect_, P_rect_, W_, H_, R_try, t_try) : -1e8);
                            if (score > best_score_) {
                                best_score_ = score;
                                std::copy(r_try, r_try + 3, best_r);
                                best_t[0] = tx;
                                best_t[1] = ty;
                                best_t[2] = tz;
                            }
                        }
                    }
                }
            }
        }
    }

    std::copy(best_r, best_r + 3, r_curr_);
    std::copy(best_t, best_t + 3, t_curr_);
    PrintProjectionStats("after_coarse", edge_points_, R_rect_, P_rect_, r_curr_, t_curr_, W_, H_);
}

void EdgeCalibrator::PerformGeometricRegularizedRefinement() {
    if (edge_dist_.empty() && semantic_map_.empty()) return;

    std::cout << "[Stage 3] Fine Calibration..." << std::endl;
    PrintProjectionStats("before_fine", edge_points_, R_rect_, P_rect_, r_curr_, t_curr_, W_, H_);
    ceres::Problem problem;
    Eigen::Matrix3d r_mat;
    ceres::AngleAxisToRotationMatrix(r_curr_, r_mat.data());
    Eigen::Vector3d t_vec(t_curr_[0], t_curr_[1], t_curr_[2]);
    ProjectionDebugStats stats = CountProjectionStats(edge_points_, R_rect_, P_rect_, r_mat, t_vec, W_, H_);
    const double in_image_ratio = stats.total > 0 ? static_cast<double>(stats.in_bounds) / stats.total : 0.0;
    const double behind_ratio = stats.total > 0 ? static_cast<double>(stats.behind) / stats.total : 0.0;

    // Use only in-view / near-view points in fine stage to improve gradient quality.
    // Make the thresholds tunable for quick experiments.
    const double fine_margin_switch = GetEnvDouble("EDGECALIB_FINE_MARGIN_SWITCH_INIMAGE", 0.03);
    const int fine_margin_low = static_cast<int>(GetEnvDouble("EDGECALIB_FINE_MARGIN_LOW", 240.0));
    const int fine_margin_high = static_cast<int>(GetEnvDouble("EDGECALIB_FINE_MARGIN_HIGH", 120.0));
    const int fine_margin = (in_image_ratio < fine_margin_switch) ? fine_margin_low : fine_margin_high;
    std::vector<PointFeature> fine_points = FilterPointsInView(edge_points_, R_rect_, P_rect_, r_mat, t_vec, W_, H_, fine_margin);
    if (fine_points.size() < 200) {
        fine_points = edge_points_;
        std::cout << "[Debug][Fine] Filtered points too few, fallback to all edge points: "
                  << fine_points.size() << std::endl;
    } else {
        std::cout << "[Debug][Fine] Using filtered edge points: " << fine_points.size()
                  << " / " << edge_points_.size() << " (margin=" << fine_margin << ")" << std::endl;
    }

    // Compute gating stats on the actual point set used in fine optimization.
    // This avoids disabling/enabling the translation prior based on points that do not
    // contribute gradients in the fine stage.
    ProjectionDebugStats fine_stats = CountProjectionStats(fine_points, R_rect_, P_rect_, r_mat, t_vec, W_, H_);
    const double fine_in_image_ratio =
        fine_stats.total > 0 ? static_cast<double>(fine_stats.in_bounds) / fine_stats.total : 0.0;
    const double fine_behind_ratio =
        fine_stats.total > 0 ? static_cast<double>(fine_stats.behind) / fine_stats.total : 0.0;
    const double fine_oob_ratio =
        fine_stats.total > 0 ? static_cast<double>(fine_stats.out_of_bounds) / fine_stats.total : 0.0;
    std::cout << "[Debug][Fine] Gating stats on fine_points: total=" << fine_stats.total
              << ", in_image_ratio=" << fine_in_image_ratio
              << ", behind_ratio=" << fine_behind_ratio
              << ", oob_ratio=" << fine_oob_ratio << std::endl;

    const int stride = std::max<int>(1, static_cast<int>(fine_points.size() / 5000));
    for (size_t i = 0; i < fine_points.size(); i += stride) {
        auto* cost = new SinglePointEdgeCost(fine_points[i], edge_dist_.empty() ? nullptr : &edge_dist_, R_rect_, P_rect_, W_, H_);
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SinglePointEdgeCost, 1, 3, 3>(cost),
                                 new ceres::HuberLoss(0.1), r_curr_, t_curr_);
    }

    // 1. 【修改】：激活线特征的 AutoDiff（自动微分）
    if (GetEnvBool("EDGECALIB_USE_LINE_CONSTRAINT", false) && !lines2d_.empty() && !lines3d_.empty()) {
        for (const auto& l3d : lines3d_) {
            auto* line_cost = new LineReprojectionError(l3d, lines2d_, R_rect_, P_rect_);
            // 注意这里改成了 AutoDiffCostFunction
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LineReprojectionError, 1, 3, 3>(line_cost),
                                     new ceres::HuberLoss(1.0), r_curr_, t_curr_);
        }
    }

    // 2. 【新增】：恢复遗漏的 Translation Prior (防止由于单目深度不可观测导致平移量飞掉)
    double t_prior_weight = GetEnvDouble("EDGECALIB_T_PRIOR_WEIGHT", 5.0);
    const double t_prior_disable_inimage = GetEnvDouble("EDGECALIB_T_PRIOR_DISABLE_INIMAGE_THRESH", 0.03);
    const double t_prior_weaken_inimage = GetEnvDouble("EDGECALIB_T_PRIOR_WEAKEN_INIMAGE_THRESH", 0.08);
    const double t_prior_weaken_factor = GetEnvDouble("EDGECALIB_T_PRIOR_WEAKEN_FACTOR", 0.1);

    // Gating for stability using the actual fine point set:
    // - If most fine points are out-of-bounds, we are likely in a bad basin: disable prior for basin escape.
    // - If OOB is moderate, keep a weak prior to prevent translation drifting too far.
    // - Additionally, if fine in-image ratio is extremely low, disable prior to allow escape.
    const double t_prior_disable_oob = GetEnvDouble("EDGECALIB_T_PRIOR_DISABLE_OOB_THRESH", 0.65);
    const double t_prior_weaken_oob = GetEnvDouble("EDGECALIB_T_PRIOR_WEAKEN_OOB_THRESH", 0.35);
    const double t_prior_weaken_oob_factor = GetEnvDouble("EDGECALIB_T_PRIOR_WEAKEN_OOB_FACTOR", 0.1);

    if (fine_in_image_ratio < t_prior_disable_inimage) {
        t_prior_weight = 0.0;
        std::cout << "[Debug][Fine] In-image ratio too low on fine_points (" << fine_in_image_ratio
                  << "), disable translation prior for basin escape (th=" << t_prior_disable_inimage << ")." << std::endl;
    } else if (fine_oob_ratio > t_prior_disable_oob) {
        t_prior_weight = 0.0;
        std::cout << "[Debug][Fine] OOB ratio high on fine_points (" << fine_oob_ratio
                  << "), disable translation prior for basin escape (th=" << t_prior_disable_oob << ")." << std::endl;
    } else if (fine_oob_ratio > t_prior_weaken_oob) {
        t_prior_weight *= t_prior_weaken_oob_factor;
        std::cout << "[Debug][Fine] OOB ratio moderate on fine_points (" << fine_oob_ratio
                  << "), weaken translation prior by factor " << t_prior_weaken_oob_factor
                  << " (th=" << t_prior_weaken_oob << "). New weight=" << t_prior_weight << std::endl;
    } else if (fine_in_image_ratio < t_prior_weaken_inimage) {
        t_prior_weight *= t_prior_weaken_factor;
        std::cout << "[Debug][Fine] Low in-image ratio on fine_points (" << fine_in_image_ratio
                  << "), weaken translation prior by factor " << t_prior_weaken_factor
                  << " (th=" << t_prior_weaken_inimage << "). New weight=" << t_prior_weight << std::endl;
    }
    if (t_prior_weight > 0.0) {
        // 使用 config 里的初始平移量作为先验锚点
        Eigen::Vector3d t_init_prior(config_.init_t[0], config_.init_t[1], config_.init_t[2]);
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TranslationPriorCost, 3, 3>(
                new TranslationPriorCost(t_init_prior, t_prior_weight)),
            nullptr, t_curr_);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;

    if (!GetEnvBool("EDGECALIB_OPT_TRANSLATION", true)) {
        problem.SetParameterBlockConstant(t_curr_);
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    PrintProjectionStats("after_fine", edge_points_, R_rect_, P_rect_, r_curr_, t_curr_, W_, H_);
}

void EdgeCalibrator::PerformFineOptimization() {
    // Phase B5: semantic fine optimization (local search) then geometric regularized refinement.
    PerformSemanticFineOptimizationIfEnabled();
    PerformGeometricRegularizedRefinement();
}

void EdgeCalibrator::ApplyTemporalSmoothing() {
    std::cout << "[Stage 4] Temporal smoothing..." << std::endl;
    r_result_ = Eigen::Vector3d(r_curr_[0], r_curr_[1], r_curr_[2]);
    t_result_ = Eigen::Vector3d(t_curr_[0], t_curr_[1], t_curr_[2]);

    if (!history_.is_smooth(r_result_, t_result_, 0.05)) {
        auto smoothed = history_.get_smoothed();
        r_result_ = smoothed.first;
        t_result_ = smoothed.second;
    }

    // Refresh unified breakdown at the final exported pose.
    // This avoids leaving breakdown as all-zero when semantic branch is not active.
    {
        const double w_edge = config_.edge_weight;
        const double w_line = config_.line_weight;
        TotalScoreBreakdown bd;
        const Eigen::Vector3d t_eval = t_result_;
        double r_eval_aa[3] = {r_result_.x(), r_result_.y(), r_result_.z()};
        Eigen::Matrix3d R_eval;
        ceres::AngleAxisToRotationMatrix(r_eval_aa, R_eval.data());

        // semantic_points_/semantic_probs_ may be empty in legacy runs; semantic terms then stay 0.
        (void)ComputeTotalCalibrationScoreSemanticDominant(
            edge_points_, edge_dist_, edge_weight_, lines3d_, lines2d_, semantic_points_, semantic_probs_,
            R_rect_, P_rect_, W_, H_, R_eval, t_eval,
            config_.semantic_js_weight, config_.histogram_weight, w_edge, w_line, sem_cfg_, &bd);

        double rail_sum = 0.0, vert_sum = 0.0;
        int rail_cnt = 0, vert_cnt = 0;
        for (const auto& l : lines3d_) {
            const double c = std::max(0.0f, std::min(1.0f, l.confidence));
            if (l.class_id == SEM_RAIL_LIKE || l.type == 0) {
                rail_sum += c;
                rail_cnt++;
            }
            if (l.class_id == SEM_VERTICAL_STRUCTURE || l.type == 1) {
                vert_sum += c;
                vert_cnt++;
            }
        }
        bd.rail_confidence = (rail_cnt > 0) ? (rail_sum / static_cast<double>(rail_cnt)) : 0.0;
        bd.vertical_structure_confidence = (vert_cnt > 0) ? (vert_sum / static_cast<double>(vert_cnt)) : 0.0;
        last_score_breakdown_ = bd;
    }

    history_.push(r_result_, t_result_, best_score_);
    if (!config_.history_file.empty()) {
        SaveCalibHistory(config_.history_file, history_);
    }
}

bool EdgeCalibrator::SaveResult() const {
    std::string output = config_.output_file.empty() ? (config_.lidar_base + "_calib_result.txt") : config_.output_file;
    std::ofstream result_file(output);
    if (!result_file.is_open()) return false;

    // Phase A3: stable key-value output for downstream parsing.
    // Always write the same keys.
    result_file << "r: " << r_result_[0] << " " << r_result_[1] << " " << r_result_[2] << "\n";
    result_file << "t: " << t_result_[0] << " " << t_result_[1] << " " << t_result_[2] << "\n";
    result_file << "Score: " << best_score_ << "\n";
    // Write cached semantic/regularizer terms (may be 0.0 when semantic inputs are absent).
    result_file << "semantic_js_divergence: " << last_score_breakdown_.semantic_js_divergence << "\n";
    result_file << "semantic_hist_similarity: " << last_score_breakdown_.semantic_hist_similarity << "\n";
    result_file << "edge_term_norm: " << last_score_breakdown_.edge_score_norm << "\n";
    result_file << "line_term_norm: " << last_score_breakdown_.line_score_norm << "\n";
    // Phase C5: unified confidences inferred from extracted line confidences.
    result_file << "rail_confidence: " << last_score_breakdown_.rail_confidence << "\n";
    result_file << "vertical_structure_confidence: " << last_score_breakdown_.vertical_structure_confidence << "\n";
    return true;
}

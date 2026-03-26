#include "include/edge_calibrator.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "include/optimizer_cost_function.h"
#include "include/optimizer_scoring.h"

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
}  // namespace

EdgeCalibrator::EdgeCalibrator(const EdgeCalibratorConfig& config) : config_(config) {
    std::copy(config_.init_r, config_.init_r + 3, r_curr_);
    std::copy(config_.init_t, config_.init_t + 3, t_curr_);
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

    if (!config_.history_file.empty()) {
        LoadCalibHistory(config_.history_file, history_);
    }
    return true;
}

void EdgeCalibrator::PerformCoarseSearch() {
    std::cout << "[Stage 2] Coarse Calibration..." << std::endl;
    const bool use_edge = !edge_dist_.empty();
    const bool use_semantic = !semantic_map_.empty() && !use_edge;

    double best_r[3] = {r_curr_[0], r_curr_[1], r_curr_[2]};
    double best_t[3] = {t_curr_[0], t_curr_[1], t_curr_[2]};

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> init_R_row;
    ceres::AngleAxisToRotationMatrix(r_curr_, init_R_row.data());
    Eigen::Matrix3d init_R = init_R_row; // Eigen 会自动正确地将 Row-Major 转换为 Col-Major
    Eigen::Vector3d init_t(t_curr_[0], t_curr_[1], t_curr_[2]);

    best_score_ = use_edge
        ? EdgeAttractionScore(edge_points_, edge_dist_, edge_weight_, R_rect_, P_rect_, W_, H_, init_R, init_t)
        : MaskIntensityVarianceScore(points_, semantic_map_, R_rect_, P_rect_, W_, H_, init_R, init_t);

    const double angle_range = 0.15;
    const double angle_step = 0.01;
    for (double rx = r_curr_[0] - angle_range; rx <= r_curr_[0] + angle_range + 1e-9; rx += angle_step) {
        for (double ry = r_curr_[1] - angle_range; ry <= r_curr_[1] + angle_range + 1e-9; ry += angle_step) {
            for (double rz = r_curr_[2] - angle_range; rz <= r_curr_[2] + angle_range + 1e-9; rz += angle_step) {
                double r_try[3] = {rx, ry, rz};
                Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_try_row;
                ceres::AngleAxisToRotationMatrix(r_try, R_try_row.data());
                Eigen::Matrix3d R_try = R_try_row;
                Eigen::Vector3d t_try(t_curr_[0], t_curr_[1], t_curr_[2]);
                double score = use_edge
                    ? EdgeAttractionScore(edge_points_, edge_dist_, edge_weight_, R_rect_, P_rect_, W_, H_, R_try, t_try)
                    : (use_semantic ? MaskIntensityVarianceScore(points_, semantic_map_, R_rect_, P_rect_, W_, H_, R_try, t_try) : -1e8);
                if (score > best_score_) {
                    best_score_ = score;
                    std::copy(r_try, r_try + 3, best_r);
                }
            }
        }
    }

    std::copy(best_r, best_r + 3, r_curr_);
    std::copy(best_t, best_t + 3, t_curr_);
}

void EdgeCalibrator::PerformFineOptimization() {
    if (edge_dist_.empty() && semantic_map_.empty()) return;

    std::cout << "[Stage 3] Fine Calibration..." << std::endl;
    ceres::Problem problem;
    const int stride = std::max<int>(1, static_cast<int>(edge_points_.size() / 5000));
    for (size_t i = 0; i < edge_points_.size(); i += stride) {
        auto* cost = new SinglePointEdgeCost(edge_points_[i], edge_dist_.empty() ? nullptr : &edge_dist_, R_rect_, P_rect_, W_, H_);
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SinglePointEdgeCost, 1, 3, 3>(cost),
                                 new ceres::HuberLoss(0.1), r_curr_, t_curr_);
    }

    // 1. 【修改】：激活线特征的 AutoDiff（自动微分）
    if (GetEnvBool("EDGECALIB_USE_LINE_CONSTRAINT", true) && !lines2d_.empty() && !lines3d_.empty()) {
        for (const auto& l3d : lines3d_) {
            auto* line_cost = new LineReprojectionError(l3d, lines2d_, R_rect_, P_rect_);
            // 注意这里改成了 AutoDiffCostFunction
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LineReprojectionError, 1, 3, 3>(line_cost),
                                     new ceres::HuberLoss(1.0), r_curr_, t_curr_);
        }
    }

    // 2. 【新增】：恢复遗漏的 Translation Prior (防止由于单目深度不可观测导致平移量飞掉)
    double t_prior_weight = GetEnvDouble("EDGECALIB_T_PRIOR_WEIGHT", 20.0);
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

    history_.push(r_result_, t_result_, best_score_);
    if (!config_.history_file.empty()) {
        SaveCalibHistory(config_.history_file, history_);
    }
}

bool EdgeCalibrator::SaveResult() const {
    std::string output = config_.output_file.empty() ? (config_.lidar_base + "_calib_result.txt") : config_.output_file;
    std::ofstream result_file(output);
    if (!result_file.is_open()) return false;

    result_file << "# EdgeCalib v2.0 Calibration Result\n";
    result_file << r_result_[0] << " " << r_result_[1] << " " << r_result_[2] << "\n";
    result_file << t_result_[0] << " " << t_result_[1] << " " << t_result_[2] << "\n";
    result_file << "# Score: " << best_score_ << "\n";
    return true;
}

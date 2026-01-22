#include "include/common.h"
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <map>
#include <algorithm>
#include <deque>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// ========================================
// 全局历史记录（用于时序平滑）
// ========================================
struct CalibHistory {
    std::deque<Eigen::Vector3d> rotation_history;
    std::deque<Eigen::Vector3d> translation_history;
    std::deque<double> score_history;
    int max_history = 3;
    
    void push(const Eigen::Vector3d& r, const Eigen::Vector3d& t, double score) {
        rotation_history.push_back(r);
        translation_history.push_back(t);
        score_history.push_back(score);
        
        if (rotation_history.size() > static_cast<size_t>(max_history)) {
            rotation_history.pop_front();
            translation_history.pop_front();
            score_history.pop_front();
        }
    }
    
    bool is_smooth(const Eigen::Vector3d& r_new, const Eigen::Vector3d& t_new, double threshold = 0.05) const {
        if (rotation_history.empty()) return true;
        
        Eigen::Vector3d r_last = rotation_history.back();
        Eigen::Vector3d t_last = translation_history.back();
        
        double r_diff = (r_new - r_last).norm();
        double t_diff = (t_new - t_last).norm();
        
        return (r_diff < threshold && t_diff < threshold);
    }
    
    std::pair<Eigen::Vector3d, Eigen::Vector3d> get_smoothed() const {
        if (rotation_history.empty()) {
            return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
        }
        
        Eigen::Vector3d r_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d t_mean = Eigen::Vector3d::Zero();
        
        for (size_t i = 0; i < rotation_history.size(); ++i) {
            r_mean += rotation_history[i];
            t_mean += translation_history[i];
        }
        
        r_mean /= rotation_history.size();
        t_mean /= translation_history.size();
        
        return {r_mean, t_mean};
    }
};

static CalibHistory g_calib_history;

bool LoadCalibHistory(const std::string& history_file, CalibHistory& history) {
    if (history_file.empty()) {
        return false;
    }
    std::ifstream file(history_file);
    if (!file.is_open()) {
        return false;
    }

    history.rotation_history.clear();
    history.translation_history.clear();
    history.score_history.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        double rx, ry, rz, tx, ty, tz, score;
        if (ss >> rx >> ry >> rz >> tx >> ty >> tz >> score) {
            history.rotation_history.emplace_back(rx, ry, rz);
            history.translation_history.emplace_back(tx, ty, tz);
            history.score_history.push_back(score);
        }
    }
    return !history.rotation_history.empty();
}

bool SaveCalibHistory(const std::string& history_file, const CalibHistory& history) {
    if (history_file.empty()) {
        return false;
    }
    std::ofstream file(history_file);
    if (!file.is_open()) {
        return false;
    }
    file << "# rx ry rz tx ty tz score\n";
    for (size_t i = 0; i < history.rotation_history.size(); ++i) {
        const auto& r = history.rotation_history[i];
        const auto& t = history.translation_history[i];
        double score = 0.0;
        if (i < history.score_history.size()) {
            score = history.score_history[i];
        }
        file << r.x() << " " << r.y() << " " << r.z() << " "
             << t.x() << " " << t.y() << " " << t.z() << " " << score << "\n";
    }
    return true;
}

struct LabelStats {
    double mean_intensity = 0.0;
    double var_intensity = 0.0;
    Eigen::Vector3d mean_normal = Eigen::Vector3d::Zero();
    int count = 0;

    bool valid() const { return count > 10; }
};

// ========================================
// 工具函数
// ========================================
int GetMaxLabel(const cv::Mat& semantic_map) {
    if (semantic_map.empty()) {
        return 0;
    }
    if (semantic_map.type() == CV_16U || semantic_map.type() == CV_32S) {
        double min_v = 0.0;
        double max_v = 0.0;
        cv::minMaxLoc(semantic_map, &min_v, &max_v);
        return static_cast<int>(max_v);
    }
    if (semantic_map.type() == CV_8U) {
        return 255;
    }
    return 0;
}
// 统一的语义图标签读取
inline int GetSemanticLabel(const cv::Mat& semantic_map, int u, int v) {
    if (semantic_map.empty()) return 0;
    if (semantic_map.type() == CV_16U) return semantic_map.at<ushort>(v, u);
    else if (semantic_map.type() == CV_8U) return semantic_map.at<uchar>(v, u);
    else if (semantic_map.type() == CV_32S) return semantic_map.at<int>(v, u);
    return 0;
}

// 统一的距离图值读取
inline float GetDistanceValue(const cv::Mat& dist_map, int u, int v) {
    if (dist_map.empty()) return 1.0f;
    if (dist_map.type() == CV_16U) return static_cast<float>(dist_map.at<ushort>(v, u)) / 65535.0f;
    else if (dist_map.type() == CV_32F) return dist_map.at<float>(v, u);
    else if (dist_map.type() == CV_64F) return static_cast<float>(dist_map.at<double>(v, u));
    return 1.0f;
}

bool LoadCalib(const std::string& calib_file, Eigen::Matrix3d& K, Eigen::Matrix3d& R_rect) {
    if (!calib_file.empty() && IOUtils::LoadKittiCalib(calib_file, K)) {
        R_rect = Eigen::Matrix3d::Identity();
        return true;
    }
    
    // 如果文件为空或加载失败，使用默认值
    std::cerr << "[Warning] Using default calibration parameters" << std::endl;
    K << 721.5, 0, 609.5, 0, 721.5, 172.8, 0, 0, 1;
    R_rect = Eigen::Matrix3d::Identity(); 
    return true; 
}

// Project声明
bool Project(const Eigen::Vector3d& p_lidar, const Eigen::Matrix3d& K, 
            const Eigen::Matrix3d& R, const Eigen::Vector3d& t, 
            int& u, int& v, int W, int H) {
    Eigen::Vector3d p_cam = R * p_lidar + t;
    if (p_cam.z() < 0.1) return false;
    Eigen::Vector3d uv = K * p_cam;
    u = static_cast<int>(uv.x() / uv.z());
    v = static_cast<int>(uv.y() / uv.z());
    return (u >= 0 && u < W && v >= 0 && v < H);
}

//Debug
//-------------------------------------------------------
struct ProjectionDebugStats {
    int total = 0;
    int behind = 0;
    int out_of_bounds = 0;
    int in_bounds = 0;
    int labeled = 0;
    int unlabeled = 0;
};

ProjectionDebugStats CountProjectionStats(const std::vector<PointFeature>& points,
                                          const Eigen::Matrix3d& K,
                                          const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& t,
                                          int W, int H) {
    ProjectionDebugStats stats;
    for (const auto& pt : points) {
        stats.total++;
        if (pt.label == 0) {
            stats.unlabeled++;
        } else {
            stats.labeled++;
        }
        Eigen::Vector3d p_cam = R * pt.p + t;
        if (p_cam.z() < 0.1) {
            stats.behind++;
            continue;
        }
        Eigen::Vector3d uv = K * p_cam;
        int u = static_cast<int>(uv.x() / uv.z());
        int v = static_cast<int>(uv.y() / uv.z());
        if (u >= 0 && u < W && v >= 0 && v < H) {
            stats.in_bounds++;
        } else {
            stats.out_of_bounds++;
        }
    }
    return stats;
}

//-------------------------------------------------------

std::vector<LabelStats> ComputeLabelStats(const std::vector<PointFeature>& points,
                                        const cv::Mat& semantic_map,
                                        const Eigen::Matrix3d& K,
                                        int W, int H,
                                        const Eigen::Matrix3d& R,
                                        const Eigen::Vector3d& t) {
    int max_label = GetMaxLabel(semantic_map);
    if (max_label <= 0) return std::vector<LabelStats>();

    std::vector<double> sum_intensity(max_label + 1, 0.0);
    std::vector<double> sum_intensity_sq(max_label + 1, 0.0);
    std::vector<Eigen::Vector3d> sum_normal(max_label + 1, Eigen::Vector3d::Zero());
    std::vector<int> counts(max_label + 1, 0);

    for (const auto& pt : points) {
        int u, v;
        if (!Project(pt.p, K, R, t, u, v, W, H)) continue;

        int img_label = GetSemanticLabel(semantic_map, u, v);
        if (img_label <= 0 || img_label > max_label) continue;
        sum_intensity[img_label] += pt.intensity;
        sum_intensity_sq[img_label] += pt.intensity * pt.intensity;
        Eigen::Vector3d n = pt.normal;
        if (n.norm() > 1e-6) {
        n.normalize();
        }

        sum_normal[img_label] += n;
        counts[img_label] += 1;
    }
    

    std::vector<LabelStats> stats(max_label + 1);
    for (int i = 1; i <= max_label; ++i) {
        if (counts[i] == 0) continue;
        stats[i].count = counts[i];
        stats[i].mean_intensity = sum_intensity[i] / counts[i];
        double mean_sq = sum_intensity_sq[i] / counts[i];
        stats[i].var_intensity = std::max(0.0, mean_sq - stats[i].mean_intensity * stats[i].mean_intensity);
        Eigen::Vector3d mean_n = sum_normal[i];
        if (mean_n.norm() > 1e-6) {
            mean_n.normalize();
        }
        stats[i].mean_normal = mean_n;
    }

    return stats;
}




// ========================================
// 语义兼容性打分
// ========================================
double GetSemanticCompatibility(int pc_label, int img_label) {
    if (pc_label == 0 || img_label == 0) return 0.0;
    
    if (pc_label == img_label) return 1.0;
    
    // 冲突惩罚
    if (pc_label == 2 && img_label == 1) return -1.0; // Veg vs Road
    if (pc_label == 1 && img_label == 2) return -1.0; // Road vs Veg
    if (pc_label == 1 && img_label == 3) return -1.0; // Road vs Struct
    if (pc_label == 3 && img_label == 1) return -1.0; // Struct vs Road
    
    return 0.0;
}

double MaskIntensityVarianceScore(const std::vector<PointFeature>& points,
                                    const cv::Mat& semantic_map,
                                    const Eigen::Matrix3d& K, int W, int H,
                                    const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    if (semantic_map.empty()) return -1e6;

    int max_label = GetMaxLabel(semantic_map);
    if (max_label <= 0) return -1e6;

    std::vector<double> sum_intensity(max_label + 1, 0.0);
    std::vector<double> sum_intensity_sq(max_label + 1, 0.0);
    std::vector<int> counts(max_label + 1, 0);
    
    for (const auto& pt : points) {        
        int u, v;
        if (!Project(pt.p, K, R, t, u, v, W, H)) continue;

        int img_label = GetSemanticLabel(semantic_map, u, v);

        if (img_label <= 0 || img_label > max_label) continue;
        sum_intensity[img_label] += pt.intensity;
        sum_intensity_sq[img_label] += pt.intensity * pt.intensity;
        counts[img_label] += 1;
    }

    double total_variance = 0.0;
    int valid_labels = 0;
    for (int i = 1; i <= max_label; ++i) {
        if (counts[i] < 10) continue;
        double mean = sum_intensity[i] / counts[i];
        double mean_sq = sum_intensity_sq[i] / counts[i];
        double variance = std::max(0.0, mean_sq - mean * mean);
        total_variance += variance;
        valid_labels++;
    }

    if (valid_labels == 0) return -1e6;
    return -total_variance;

}


// ========================================
// 边缘吸引场打分 (README2.0 第一阶段 2.3)
// ========================================
double EdgeAttractionScore(const std::vector<PointFeature>& points,
                           const cv::Mat& dist_map,
                           const cv::Mat& weight_map,
                           const Eigen::Matrix3d& K, int W, int H,
                           const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    if (dist_map.empty()) return -1e6;
    double total_score = 0.0;
    int visible_count = 0;

    for (const auto& pt : points) {
        if (pt.label == 0) continue;  // 跳过未标记点
        
        int u, v;
        if (!Project(pt.p, K, R, t, u, v, W, H)) {
            continue;
        }

        // 读取距离场值
        float dist_value = GetDistanceValue(dist_map, u, v);

        // 读取边缘权重值
        float edge_weight = GetDistanceValue(weight_map, u, v);

        dist_value = std::min(std::max(dist_value, 0.0f), 1.0f);
        edge_weight = std::min(std::max(edge_weight, 0.0f), 1.0f);

        // 综合打分: 吸引场得分 × 边缘权重 × 时空权重
        // 距离越近(dist_value越小)，得分越高
        double score = (1.0f - dist_value) * edge_weight * pt.weight;
        total_score += score;
        visible_count++;
    }

    if (visible_count < 50) return -1e6;
    return total_score;
}

// ========================================
// Ceres优化：3D-2D线匹配
// ========================================
struct LineReprojectionError {
    LineReprojectionError(const Line3D& l3d, const std::vector<Line2D>& l2ds, 
                          const double fx, const double fy, const double cx, const double cy)
        : l3d_(l3d), l2ds_(l2ds), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const r, const T* const t, T* residual) const {
        // 1. Transform 3D Line Endpoints
        T p1[3], p2[3];
        T p1_raw[3] = {T(l3d_.p1.x()), T(l3d_.p1.y()), T(l3d_.p1.z())};
        T p2_raw[3] = {T(l3d_.p2.x()), T(l3d_.p2.y()), T(l3d_.p2.z())};
        
        ceres::AngleAxisRotatePoint(r, p1_raw, p1);
        ceres::AngleAxisRotatePoint(r, p2_raw, p2);
        p1[0] += t[0]; p1[1] += t[1]; p1[2] += t[2];
        p2[0] += t[0]; p2[1] += t[1]; p2[2] += t[2];

        // 2. Project to Image
        if (p1[2] < T(0.1) || p2[2] < T(0.1)) {
            residual[0] = T(0.0); return true; // Behind camera
        }
        T u1 = (p1[0] * T(fx_) / p1[2]) + T(cx_);
        T v1 = (p1[1] * T(fy_) / p1[2]) + T(cy_);
        T u2 = (p2[0] * T(fx_) / p2[2]) + T(cx_);
        T v2 = (p2[1] * T(fy_) / p2[2]) + T(cy_);

        // 3. Data Association & Error Calculation
        T min_dist = T(1000.0);
        bool found = false;

        for (const auto& l2d : l2ds_) {
            if (l2d.type != l3d_.type) continue; 
            
            double lx1 = l2d.p1.x(), ly1 = l2d.p1.y();
            double lx2 = l2d.p2.x(), ly2 = l2d.p2.y();
            
            double A = ly1 - ly2;
            double B = lx2 - lx1;
            double C = lx1 * ly2 - lx2 * ly1;
            double norm = std::sqrt(A*A + B*B);
            
            if (norm < 1e-6) continue;
            
            T d1 = ceres::abs(T(A)*u1 + T(B)*v1 + T(C)) / T(norm);
            T d2 = ceres::abs(T(A)*u2 + T(B)*v2 + T(C)) / T(norm);
            
            T avg_dist = (d1 + d2) / T(2.0);
            if (avg_dist < min_dist) {
                min_dist = avg_dist;
                found = true;
            }
        }

        if (found && min_dist < T(50.0)) {
            residual[0] = min_dist;
        } else {
            residual[0] = T(0.0);
        }
        return true;
    }

    Line3D l3d_;
    std::vector<Line2D> l2ds_;
    double fx_, fy_, cx_, cy_;
};

// ========================================
// Ceres优化：边缘吸引场 + 语义一致性 (README2.0 第三阶段 3.2)
// ========================================
struct EdgeConsistencyCost {
    EdgeConsistencyCost(const std::vector<PointFeature>* points,
                        const std::vector<int>* indices,
                        const cv::Mat* dist_map,
                        const cv::Mat* semantic_map,
                        const std::vector<LabelStats>* label_stats,
                        const Eigen::Matrix3d& K,
                        int W, int H,
                        double w_edge,
                        double w_consistency)
        : points_(points),
          indices_(indices),
          dist_map_(dist_map),
          semantic_map_(semantic_map),
          label_stats_(label_stats),
          K_(K),
          W_(W),
          H_(H),
          w_edge_(w_edge),
          w_consistency_(w_consistency) {}

    bool operator()(const double* const r, const double* const t, double* residual) const {
        double total_error = 0.0;
        int visible_count = 0;

        double rot[9];
        ceres::AngleAxisToRotationMatrix(r, rot);

        for (int idx : *indices_) {
            const auto& pt = (*points_)[idx];
            if (pt.label == 0) continue;

            Eigen::Vector3d p_cam;
            p_cam.x() = rot[0] * pt.p.x() + rot[1] * pt.p.y() + rot[2] * pt.p.z() + t[0];
            p_cam.y() = rot[3] * pt.p.x() + rot[4] * pt.p.y() + rot[5] * pt.p.z() + t[1];
            p_cam.z() = rot[6] * pt.p.x() + rot[7] * pt.p.y() + rot[8] * pt.p.z() + t[2];

            if (p_cam.z() < 0.1) continue;

            Eigen::Vector3d uv = K_ * p_cam;
            int u = static_cast<int>(uv.x() / uv.z());
            int v = static_cast<int>(uv.y() / uv.z());
            if (u < 0 || u >= W_ || v < 0 || v >= H_) continue;

            double edge_error = 0.0;
            if (dist_map_ && !dist_map_->empty()) {
                edge_error = static_cast<double>(GetDistanceValue(*dist_map_, u, v));
                edge_error = std::min(std::max(edge_error, 0.0), 1.0);
            }

            double consistency_error = 0.0;
            if (semantic_map_ && !semantic_map_->empty()) {
                int img_label = GetSemanticLabel(*semantic_map_, u, v);

                if (pt.label != 0 && img_label != pt.label) {
                    consistency_error += 1.0;
                }

                if (label_stats_ && img_label >= 0 && img_label < static_cast<int>(label_stats_->size())) {
                    const auto& stats = (*label_stats_)[img_label];
                    if (stats.valid()) {
                        double std_intensity = std::sqrt(stats.var_intensity) + 1e-3;
                        double intensity_error = std::abs(pt.intensity - stats.mean_intensity) / std_intensity;
                        Eigen::Vector3d n = pt.normal;
                        if (n.norm() > 1e-6) {
                            n.normalize();
                        }
                        double normal_error = 1.0 - std::max(-1.0, std::min(1.0, n.dot(stats.mean_normal)));
                        consistency_error += intensity_error + normal_error;
                    }
                }
            }

            double weight = pt.weight;
            total_error += weight * (w_edge_ * edge_error + w_consistency_ * consistency_error);
            visible_count++;
        }

        if (visible_count < 50) {
            residual[0] = 1e3;
        } else {
            residual[0] = total_error / static_cast<double>(visible_count);
        }
        return true;
    }

    const std::vector<PointFeature>* points_;
    const std::vector<int>* indices_;
    const cv::Mat* dist_map_;
    const cv::Mat* semantic_map_;
    const std::vector<LabelStats>* label_stats_;
    Eigen::Matrix3d K_;
    int W_;
    int H_;
    double w_edge_;
    double w_consistency_;
};


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
    LoadCalib(calib_file, K, R_rect);

    // ========================================
    // Debug信息输出
    // ========================================

    if (std::getenv("EDGECALIB_DEBUG")) {
        Eigen::Vector3d t_vec(t_curr[0], t_curr[1], t_curr[2]);
        Eigen::Matrix3d R_mat;
        ceres::AngleAxisToRotationMatrix(r_curr, R_mat.data());
        auto stats = CountProjectionStats(points, K, R_mat, t_vec, W, H);
        std::cout << "[Debug] Projection stats (initial pose): total=" << stats.total
                  << ", labeled=" << stats.labeled
                  << ", unlabeled=" << stats.unlabeled
                  << ", in_bounds=" << stats.in_bounds
                  << ", behind=" << stats.behind
                  << ", out_of_bounds=" << stats.out_of_bounds
                  << std::endl;
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
    }

    // ========================================
    // 2. 粗标定 (Coarse Search)
    // ========================================
    std::cout << "\n[Stage 2] Coarse Calibration (Grid Search)..." << std::endl;
    
    // 根据README2.0.md第三阶段要求：优先使用语义方差评分（Calib-Anything逻辑）
    bool use_semantic = !semantic_map.empty();
    bool use_edge = !edge_dist.empty();
    
    if (use_semantic) {
        std::cout << "  Strategy: Mask Intensity Variance Minimization (Calib-Anything)" << std::endl;
    } else if (use_edge) {
        std::cout << "  Strategy: Edge Attraction Field (EdgeCalib)" << std::endl;
    } else {
        std::cerr << "[Error] No scoring method available!" << std::endl;
        return -1;
    }
    
    double best_score = -1e9;
    double best_r[3], best_t[3];
    
    const double angle_range = 0.15;  // radians, ±8.6 degrees
    const double angle_step = 0.01;   // radians, ~0.57 degrees
    int iter = 0;
    const int steps = static_cast<int>(std::floor((2.0 * angle_range) / angle_step)) + 1;
    const int total_iters = steps * steps * steps;
    
    std::cout << "  Search space: " << steps << "^3 = " << total_iters << " candidates" << std::endl;
    std::cout << "  Angle range: ±" << (angle_range * 180.0 / M_PI) << " degrees" << std::endl;

    for (double rx = r_curr[0] - angle_range; rx <= r_curr[0] + angle_range + 1e-9; rx += angle_step) {
        for (double ry = r_curr[1] - angle_range; ry <= r_curr[1] + angle_range + 1e-9; ry += angle_step) {
            for (double rz = r_curr[2] - angle_range; rz <= r_curr[2] + angle_range + 1e-9; rz += angle_step) {
                double r_try[3] = {rx, ry, rz};
                double t_try[3] = {t_curr[0], t_curr[1], t_curr[2]};

                Eigen::Vector3d t_vec(t_try[0], t_try[1], t_try[2]);
                Eigen::Matrix3d R_mat;
                ceres::AngleAxisToRotationMatrix(r_try, R_mat.data());

                double score = 0.0;
                if (use_semantic) {
                    // README2.0.md要求：最大化Mask内强度一致性（方差最小）
                    score = MaskIntensityVarianceScore(points, semantic_map, K, W, H, R_mat, t_vec);
                } else if (use_edge) {
                    // 备用方案：边缘吸引场
                    score = EdgeAttractionScore(points, edge_dist, edge_weight, K, W, H, R_mat, t_vec);
                }else{
                    score = -1e6;
                }

                if (score > best_score) {
                    best_score = score;
                    std::copy(r_try, r_try + 3, best_r);
                    std::copy(t_try, t_try + 3, best_t);
                }

                iter++;
                if (iter % 200 == 0) {
                    std::cout << "  Progress: " << iter << "/" << total_iters
                              << " (" << (iter * 100 / total_iters) << "%), Best Score: " << best_score << std::endl;
                }
            }
        }
    }
    
    std::cout << "  Coarse Search Complete. Best Score: " << best_score << std::endl;
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
        
        // 采样点云以提高效率（最多5000个点）
        std::vector<int> sample_indices;
        sample_indices.reserve(points.size());
        const int stride = std::max<int>(1, static_cast<int>(points.size() / 5000));
        for (size_t i = 0; i < points.size(); i += stride) {
            sample_indices.push_back(static_cast<int>(i));
        }
        std::cout << "  Sampled " << sample_indices.size() << " points from " << points.size() << " total points" << std::endl;

        // 混合Loss权重：α * E_geo + β * E_sem (README2.0.md公式)
        const double w_edge = 1.0;         // α: 几何误差权重
        const double w_consistency = 1.0;  // β: 语义一致性权重
        std::cout << "  Loss weights: w_edge=" << w_edge << ", w_consistency=" << w_consistency << std::endl;
        std::vector<LabelStats> label_stats;
        if (!semantic_map.empty()) {
            Eigen::Vector3d t_vec(t_curr[0], t_curr[1], t_curr[2]);
            Eigen::Matrix3d R_mat;
            ceres::AngleAxisToRotationMatrix(r_curr, R_mat.data());
            label_stats = ComputeLabelStats(points, semantic_map, K, W, H, R_mat, t_vec);
        }
        auto* cost = new EdgeConsistencyCost(
            &points,
            &sample_indices,
            edge_dist.empty() ? nullptr : &edge_dist,
            semantic_map.empty() ? nullptr : &semantic_map,
            label_stats.empty() ? nullptr : &label_stats,
            K,
            W,
            H,
            w_edge,
            w_consistency);

        problem.AddResidualBlock(
            new ceres::NumericDiffCostFunction<EdgeConsistencyCost, ceres::CENTRAL, 1, 3, 3>(cost),
            new ceres::HuberLoss(1.0),
            r_curr,
            t_curr);

        if (!lines3d.empty() && !lines2d.empty()) {
            std::cout << "  Adding line reprojection constraints: " << lines3d.size() << " lines" << std::endl;
            for (const auto& l3d : lines3d) {
                auto* line_cost = new LineReprojectionError(
                    l3d,
                    lines2d,
                    K(0, 0),
                    K(1, 1),
                    K(0, 2),
                    K(1, 2));
                problem.AddResidualBlock(
                    new ceres::NumericDiffCostFunction<LineReprojectionError, ceres::CENTRAL, 1, 3, 3>(line_cost),
                    new ceres::HuberLoss(1.0),
                    r_curr,
                    t_curr);
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 100;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        std::cout << "  " << summary.BriefReport() << std::endl;
        std::cout << "  Final R: [" << r_curr[0] << ", " << r_curr[1] << ", " << r_curr[2] << "]" << std::endl;
        std::cout << "  Final T: [" << t_curr[0] << ", " << t_curr[1] << ", " << t_curr[2] << "]" << std::endl;
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

#include "include/optimizer_data_loader.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

bool ParseCSVToDoubles(const std::string& csv, std::vector<double>& out) {
    out.clear();
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        char* end = nullptr;
        const double v = std::strtod(item.c_str(), &end);
        if (end != item.c_str()) {
            out.push_back(v);
        }
    }
    return !out.empty();
}

bool LoadSemanticProbabilityMaps(const std::string& path, SemanticProbMaps& out) {
    return LoadSemanticProbabilityMapsBinary(path, out);
}

bool LoadInitPoseFromBEV(const std::string& path, double r_out[3], double t_out[3]) {
    if (path.empty()) {
        return false;
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[LoadInitPoseFromBEV] Cannot open " << path << std::endl;
        return false;
    }
    std::vector<double> nums;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string part;
        while (ss >> part) {
            char* end = nullptr;
            double v = std::strtod(part.c_str(), &end);
            if (end != part.c_str()) nums.push_back(v);
        }
    }
    if (nums.size() < 6) {
        std::cerr << "[LoadInitPoseFromBEV] Need 6 numbers in " << path << std::endl;
        return false;
    }
    r_out[0] = nums[0];
    r_out[1] = nums[1];
    r_out[2] = nums[2];
    t_out[0] = nums[3];
    t_out[1] = nums[4];
    t_out[2] = nums[5];
    return true;
}

void CalibHistory::push(const Eigen::Vector3d& r, const Eigen::Vector3d& t, double score) {
    rotation_history.push_back(r);
    translation_history.push_back(t);
    score_history.push_back(score);

    if (rotation_history.size() > static_cast<size_t>(max_history)) {
        rotation_history.pop_front();
        translation_history.pop_front();
        score_history.pop_front();
    }
}

bool CalibHistory::is_smooth(const Eigen::Vector3d& r_new, const Eigen::Vector3d& t_new, double threshold) const {
    if (rotation_history.empty()) return true;

    Eigen::Vector3d r_last = rotation_history.back();
    Eigen::Vector3d t_last = translation_history.back();

    double r_diff = (r_new - r_last).norm();
    double t_diff = (t_new - t_last).norm();

    return (r_diff < threshold && t_diff < threshold);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> CalibHistory::get_smoothed() const {
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

std::vector<PointFeature> LoadEdgePointsCustom(const std::string& filename) {
    std::vector<PointFeature> pts;
    std::ifstream fin(filename);
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double x, y, z, val;
        if (ss >> x >> y >> z >> val) {
            PointFeature pt;
            pt.p = Eigen::Vector3d(x, y, z);
            pt.intensity = val;
            pt.weight = 1.0;
            pt.label = 1;
            pts.push_back(pt);
        }
    }
    return pts;
}

bool LoadCalib(const std::string& calib_file,
               Eigen::Matrix3d& K,
               Eigen::Matrix3d& R_rect,
               Eigen::Matrix<double, 3, 4>& P_rect,
               bool* used_default) {
    if (used_default) {
        *used_default = false;
    }

    const char* fmt_env = std::getenv("EDGECALIB_DATASET_FORMAT");
    std::string ds_fmt = fmt_env ? std::string(fmt_env) : "";
    for (auto& c : ds_fmt) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    // Explicit dataset: avoid trying KITTI parser on OSDaR23 files (misleading errors).
    if (ds_fmt == "osdar23" || ds_fmt == "osdar") {
        if (!calib_file.empty()) {
            const char* cam = std::getenv("EDGECALIB_OSDAR_CAMERA");
            std::string cam_folder = cam ? std::string(cam) : "rgb_center";
            Eigen::Matrix4d T_lidar_to_cam;
            if (IOUtils::LoadOSDaRCalib(calib_file, cam_folder, K, T_lidar_to_cam)) {
                R_rect = Eigen::Matrix3d::Identity();
                P_rect << K(0, 0), K(0, 1), K(0, 2), 0.0,
                          K(1, 0), K(1, 1), K(1, 2), 0.0,
                          K(2, 0), K(2, 1), K(2, 2), 0.0;
                return true;
            }
        }
        std::cerr << "[Warning] OSDaR calib load failed; using default calibration parameters" << std::endl;
    } else {
        if (!calib_file.empty() && IOUtils::LoadKittiCalib(calib_file, K, R_rect, P_rect)) {
            return true;
        }

        // Legacy fallback: try OSDaR23 if env points to OSDaR camera (or unset format on mixed runs).
        if (!calib_file.empty()) {
            const char* cam = std::getenv("EDGECALIB_OSDAR_CAMERA");
            std::string cam_folder = cam ? std::string(cam) : "rgb_center";
            Eigen::Matrix4d T_lidar_to_cam;
            if (IOUtils::LoadOSDaRCalib(calib_file, cam_folder, K, T_lidar_to_cam)) {
                R_rect = Eigen::Matrix3d::Identity();
                P_rect << K(0, 0), K(0, 1), K(0, 2), 0.0,
                          K(1, 0), K(1, 1), K(1, 2), 0.0,
                          K(2, 0), K(2, 1), K(2, 2), 0.0;
                return true;
            }
        }
    }

    std::cerr << "[Warning] Using default calibration parameters" << std::endl;
    K << 721.5, 0, 609.5, 0, 721.5, 172.8, 0, 0, 1;

    R_rect = Eigen::Matrix3d::Identity();
    P_rect << K(0, 0), K(0, 1), K(0, 2), 0.0,
              K(1, 0), K(1, 1), K(1, 2), 0.0,
              K(2, 0), K(2, 1), K(2, 2), 0.0;
    if (used_default) {
        *used_default = true;
    }
    return true;
}
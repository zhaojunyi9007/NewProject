// Phase 4：读取 LiDAR / 图像 BEV 二进制，粗估计 yaw、tx、ty 增量并写出位姿与调试图。
#include "include/bev_builder.h"
#include "include/bev_matcher.h"
#include <opencv2/calib3d.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <eigen3/Eigen/Core>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double GetEnvD(const char* name, double def) {
    const char* v = std::getenv(name);
    if (!v) return def;
    char* end = nullptr;
    double x = std::strtod(v, &end);
    if (end == v) return def;
    return x;
}

static bool LoadInitPoseTxt(const std::string& path, cv::Mat& rvec, cv::Mat& tvec) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[bev_initializer] Cannot open " << path << std::endl;
        return false;
    }
    std::vector<double> nums;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        double x;
        while (ss >> x) {
            nums.push_back(x);
        }
    }
    if (nums.size() < 6) {
        std::cerr << "[bev_initializer] Need 6 numbers (rvec x3, tvec x3) in " << path << std::endl;
        return false;
    }
    rvec = (cv::Mat_<double>(3, 1) << nums[0], nums[1], nums[2]);
    tvec = (cv::Mat_<double>(3, 1) << nums[3], nums[4], nums[5]);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: bev_initializer <lidar_bev.bin> <image_rail_bev.bin> <init_pose.txt> <out_dir> "
                     "<frame_id>\n";
        return -1;
    }
    const std::string lidar_path = argv[1];
    const std::string image_path = argv[2];
    const std::string init_pose_path = argv[3];
    const std::string out_dir = argv[4];
    const std::string frame_id = argv[5];

    BEVChannels lidar_bev;
    BEVChannels image_bev;
    if (!LoadBEVChannelsBin(lidar_path, &lidar_bev)) {
        return -1;
    }
    if (!LoadBEVChannelsBin(image_path, &image_bev)) {
        return -1;
    }

    cv::Mat rvec_init, tvec_init;
    if (!LoadInitPoseTxt(init_pose_path, rvec_init, tvec_init)) {
        return -1;
    }

    BEVOptimizeConfig cfg;
    cfg.yaw_min_deg = GetEnvD("EDGECALIB_BEV_YAW_MIN_DEG", cfg.yaw_min_deg);
    cfg.yaw_max_deg = GetEnvD("EDGECALIB_BEV_YAW_MAX_DEG", cfg.yaw_max_deg);
    cfg.yaw_step_deg = GetEnvD("EDGECALIB_BEV_YAW_STEP_DEG", cfg.yaw_step_deg);
    cfg.tx_min_m = GetEnvD("EDGECALIB_BEV_TX_MIN_M", cfg.tx_min_m);
    cfg.tx_max_m = GetEnvD("EDGECALIB_BEV_TX_MAX_M", cfg.tx_max_m);
    cfg.ty_min_m = GetEnvD("EDGECALIB_BEV_TY_MIN_M", cfg.ty_min_m);
    cfg.ty_max_m = GetEnvD("EDGECALIB_BEV_TY_MAX_M", cfg.ty_max_m);
    cfg.trans_step_m = GetEnvD("EDGECALIB_BEV_TRANS_STEP_M", cfg.trans_step_m);

    PoseDeltaBev delta;
    ScoreBreakdown score;
    if (!EstimateBEVDelta(lidar_bev, image_bev, cfg, &delta, &score)) {
        return -1;
    }

    cv::Mat R0;
    cv::Rodrigues(rvec_init, R0);
    Eigen::Matrix3d R_init;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_init(i, j) = R0.at<double>(i, j);
        }
    }
    const double c = std::cos(delta.yaw_rad);
    const double s = std::sin(delta.yaw_rad);
    Eigen::Matrix3d Rz;
    Rz << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
    const Eigen::Matrix3d R_new = Rz * R_init;

    cv::Mat R_new_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_new_cv.at<double>(i, j) = R_new(i, j);
        }
    }
    cv::Mat rvec_new;
    cv::Rodrigues(R_new_cv, rvec_new);

    Eigen::Vector3d t_in(tvec_init.at<double>(0), tvec_init.at<double>(1), tvec_init.at<double>(2));
    const Eigen::Vector3d t_new = t_in + Eigen::Vector3d(delta.tx_m, delta.ty_m, 0.0);

    std::string mkdir = "mkdir -p " + out_dir;
    std::system(mkdir.c_str());

    const std::string p_delta = out_dir + "/pose_delta.txt";
    const std::string p_after = out_dir + "/pose_after_bev.txt";
    const std::string p_json = out_dir + "/debug_bev_score.json";
    const std::string p_ov = out_dir + "/debug_bev_overlay.png";
    const std::string p_mt = out_dir + "/debug_bev_match.png";

    {
        std::ofstream f(p_delta);
        f << "# yaw_rad tx_m ty_m (LiDAR frame: Z up, delta applied as Rz * R_init, t += [tx,ty,0])\n";
        f << delta.yaw_rad << " " << delta.tx_m << " " << delta.ty_m << "\n";
    }
    {
        std::ofstream f(p_after);
        f << "# rvec (3) after BEV coarse init\n";
        f << rvec_new.at<double>(0) << " " << rvec_new.at<double>(1) << " " << rvec_new.at<double>(2) << "\n";
        f << "# tvec (3)\n";
        f << t_new.x() << " " << t_new.y() << " " << t_new.z() << "\n";
    }
    {
        std::ofstream f(p_json);
        f << "{\"frame_id\":\"" << frame_id << "\",\"yaw_rad\":" << delta.yaw_rad << ",\"tx_m\":" << delta.tx_m
          << ",\"ty_m\":" << delta.ty_m << ",\"rail_score\":" << score.rail_score << ",\"total\":" << score.total
          << "}\n";
    }

    SaveBEVDebugImages(p_ov, p_mt, lidar_bev, image_bev, delta);

    std::cout << "[bev_initializer] Wrote " << p_delta << " " << p_after << std::endl;
    return 0;
}

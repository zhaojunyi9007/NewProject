#include "include/common.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace IOUtils {

    // 加载 KITTI 格式的 bin 文件 (保持不变，用于extractors)
    bool LoadKittiBin(const std::string& bin_file, std::vector<PointFeature>& points) {
        std::ifstream file(bin_file, std::ios::binary);
        if (!file.good()) {
            std::cerr << "[Error] Cannot open bin file: " << bin_file << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::beg);
        points.clear();

        while (file.good() && !file.eof()) {
            float data[4]; // x, y, z, intensity
            file.read(reinterpret_cast<char*>(data), sizeof(float) * 4);
            
            // 添加安全检查
            if (file.gcount() < static_cast<std::streamsize>(sizeof(float) * 4)) break;
            
            PointFeature pt;
            pt.p = Eigen::Vector3d(data[0], data[1], data[2]);
            pt.intensity = data[3];
            pt.normal = Eigen::Vector3d(0, 0, 0); // 原始bin没有法向
            pt.label = 0; // 默认为 0
            points.push_back(pt);
        }
        
        if (points.empty()) {
            std::cerr << "[Warning] No points loaded from: " << bin_file << std::endl;
            return false;
        }
        
        return true;
    }

    // [New] 加载 lidar_extractor 生成的点特征文件 (x y z intensity nx ny nz label weight)
    bool LoadPointFeatures(const std::string& filepath, std::vector<PointFeature>& points) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[Error] Cannot open point file: " << filepath << std::endl;
            return false;
        }

        points.clear();
        std::string line;
        int line_num = 0;
        
        while (std::getline(file, line)) {
            line_num++;
            if (line.empty() || line[0] == '#') continue; // 跳过空行和注释
            
            std::stringstream ss(line);
            double x, y, z, i, nx, ny, nz;
            int label = 0;
            double weight = 1.0;
            
            if (ss >> x >> y >> z >> i >> nx >> ny >> nz) {
                // 尝试读取 label 和 weight，如果不存在则使用默认值
                if (!(ss >> label)) label = 0;
                if (!(ss >> weight)) weight = 1.0;
                
                // 添加数据有效性检查
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    std::cerr << "[Warning] Invalid point at line " << line_num << ", skipping" << std::endl;
                    continue;
                }
                
                PointFeature pt;
                pt.p = Eigen::Vector3d(x, y, z);
                pt.intensity = static_cast<float>(i);
                pt.normal = Eigen::Vector3d(nx, ny, nz);
                pt.label = label;
                pt.weight = weight;  // 设置时空权重
                points.push_back(pt);
            } else {
                std::cerr << "[Warning] Failed to parse line " << line_num << ": " << line << std::endl;
            }
        }
        
        file.close();        
        if (points.empty()) {
            std::cerr << "[Warning] No valid points loaded from: " << filepath << std::endl;
            return false;
        }
        
        return true;
    }

    // 加载 lidar_extractor 生成的 3D 线特征 (x1 y1 z1 x2 y2 z2 type)
    bool LoadLines3D(const std::string& filepath, std::vector<Line3D>& lines) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[Error] Cannot open 3D line file: " << filepath << std::endl;
            return false;
        }

        lines.clear();
        std::string line;
        int line_num = 0;
        
        while (std::getline(file, line)) {
            line_num++;
            if (line.empty() || line[0] == '#') continue;
            
            std::stringstream ss(line);
            double x1, y1, z1, x2, y2, z2;
            int type;
            
            if (ss >> x1 >> y1 >> z1 >> x2 >> y2 >> z2 >> type) {
                // 添加数据有效性检查
                if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(z1) ||
                    !std::isfinite(x2) || !std::isfinite(y2) || !std::isfinite(z2)) {
                    std::cerr << "[Warning] Invalid line at line " << line_num << ", skipping" << std::endl;
                    continue;
                }
                
                // 检查线段长度是否合理
                Eigen::Vector3d p1(x1, y1, z1);
                Eigen::Vector3d p2(x2, y2, z2);
                double length = (p2 - p1).norm();
                
                if (length < 0.1) { // 线段太短
                    std::cerr << "[Warning] Line too short at line " << line_num << ", skipping" << std::endl;
                    continue;
                }
                
                Line3D l;
                l.p1 = p1;
                l.p2 = p2;
                l.type = type;
                lines.push_back(l);
            } else {
                std::cerr << "[Warning] Failed to parse line " << line_num << ": " << line << std::endl;
            }
        }
        
        file.close();
        
        if (lines.empty()) {
            std::cerr << "[Warning] No valid 3D lines loaded from: " << filepath << std::endl;
            return false;
        }
        
        return true;
    }

    // 加载 sam_extractor 生成的 2D 线特征 (u1 v1 u2 v2 type)
    bool LoadLines2D(const std::string& filepath, std::vector<Line2D>& lines) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[Error] Cannot open 2D line file: " << filepath << std::endl;
            return false;
        }

        lines.clear();
        std::string line;
        int line_num = 0;
        
        while (std::getline(file, line)) {
            line_num++;
            if (line.empty() || line[0] == '#') continue;
            
            std::stringstream ss(line);
            double u1, v1, u2, v2;
            int type;
            
            if (ss >> u1 >> v1 >> u2 >> v2 >> type) {
                // 添加数据有效性检查
                if (!std::isfinite(u1) || !std::isfinite(v1) || 
                    !std::isfinite(u2) || !std::isfinite(v2)) {
                    std::cerr << "[Warning] Invalid line at line " << line_num << ", skipping" << std::endl;
                    continue;
                }
                
                // 检查线段长度是否合理
                Eigen::Vector2d p1(u1, v1);
                Eigen::Vector2d p2(u2, v2);
                double length = (p2 - p1).norm();
                
                if (length < 5.0) { // 2D线段太短 (至少5像素)
                    std::cerr << "[Warning] 2D line too short at line " << line_num << ", skipping" << std::endl;
                    continue;
                }
                
                Line2D l;
                l.p1 = p1;
                l.p2 = p2;
                l.type = type;
                lines.push_back(l);
            } else {
                std::cerr << "[Warning] Failed to parse line " << line_num << ": " << line << std::endl;
            }
        }
        
        file.close();
        
        if (lines.empty()) {
            std::cerr << "[Warning] No valid 2D lines loaded from: " << filepath << std::endl;
            return false;
        }
        
        return true;
    }
    
    // 加载 KITTI 标定文件 (读取 P2/P_rect_02 和 R0_rect/R_rect_00)
    bool LoadKittiCalib(const std::string& calib_file,
                        Eigen::Matrix3d& K,
                        Eigen::Matrix3d& R_rect,
                        Eigen::Matrix<double, 3, 4>& P_rect) {
        std::ifstream file(calib_file);
        if (!file.is_open()) {
            std::cerr << "[Error] Cannot open calibration file: " << calib_file << std::endl;
            return false;
        }
        
        std::string line;
        bool found_p = false;
        bool found_r = false;
        
        std::string key;
        while (std::getline(file, line)) {
            // 查找投影矩阵行 (KITTI object: P2, KITTI raw: P_rect_02,左彩色相机的投影矩阵)
            key.clear();
            if (line.find("P2:") == 0) {
                //std::stringstream ss(line.substr(3)); // 跳过 "P2:"
                key = "P2:";
            } else if (line.find("P_rect_02:") == 0) {
                key = "P_rect_02:";
            }

            if (!key.empty()) {
                std::stringstream ss(line.substr(key.size())); // 跳过 key
                double p[12];
                bool parse_success = true;
                
                for(int i=0; i<12; ++i) {
                    if (!(ss >> p[i])) {
                        parse_success = false;
                        break;
                    }
                }
                
                if (parse_success) {
                    P_rect << p[0], p[1], p[2], p[3],
                              p[4], p[5], p[6], p[7],
                              p[8], p[9], p[10], p[11];
                    // K from projection matrix P2 (3x4)
                    // P2 = K * [R|t], usually top-left 3x3 is K if rectified
                    K << p[0], p[1], p[2],
                         p[4], p[5], p[6],
                         p[8], p[9], p[10];
                    found_p = true;
                } else {
                    std::cerr << "[Error] Failed to parse P2 line: " << line << std::endl;
                    return false;
                }
                continue;
            }

            if (line.find("R0_rect:") == 0 || line.find("R_rect_00:") == 0) {
                std::stringstream ss(line.substr(line.find(":") + 1));
                double r[9];
                bool parse_success = true;
                for (int i = 0; i < 9; ++i) {
                    if (!(ss >> r[i])) {
                        parse_success = false;
                        break;
                    }
                }
                if (!parse_success) {
                    std::cerr << "[Error] Failed to parse R_rect line: " << line << std::endl;
                    return false;
                }
                R_rect << r[0], r[1], r[2],
                          r[3], r[4], r[5],
                          r[6], r[7], r[8];
                found_r = true;

            }
        }
        
        file.close();
        
        if (!found_p) {
            std::cerr << "[Error] P2 matrix not found in calibration file" << std::endl;
            return false;
        }
         if (!found_r) {
            R_rect = Eigen::Matrix3d::Identity();
        }
        
        return true;
    }

    static std::vector<double> _extract_numbers(const std::string& s) {
        std::string buf;
        buf.reserve(s.size());
        for (char c : s) {
            if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E') {
                buf.push_back(c);
            } else {
                buf.push_back(' ');
            }
        }
        std::stringstream ss(buf);
        std::vector<double> out;
        double v;
        while (ss >> v) out.push_back(v);
        return out;
    }

    /// Read numbers from `first_segment`, then keep reading lines from `file` until `need_count`
    /// values are collected (row-major). Returns false on EOF before enough numbers.
    static bool _collect_numbers_multiline(std::ifstream& file,
                                         const std::string& first_segment,
                                         size_t need_count,
                                         std::vector<double>& out,
                                         const char* context_label) {
        out.clear();
        auto append_line = [&](const std::string& s) {
            std::vector<double> nums = _extract_numbers(s);
            for (double x : nums) {
                if (out.size() >= need_count) {
                    return;
                }
                out.push_back(x);
            }
        };
        append_line(first_segment);
        std::string extra_line;
        while (out.size() < need_count) {
            if (!std::getline(file, extra_line)) {
                std::cerr << "[Error] OSDaR " << context_label << " truncated: need " << need_count
                          << " numbers, got " << out.size() << " (EOF)" << std::endl;
                return false;
            }
            append_line(extra_line);
        }
        if (out.size() < need_count) {
            std::cerr << "[Error] OSDaR " << context_label << " truncated: need " << need_count
                      << " numbers, got " << out.size() << std::endl;
            return false;
        }
        return true;
    }

    bool LoadOSDaRCalib(const std::string& calib_file,
                        const std::string& camera_folder,
                        Eigen::Matrix3d& K,
                        Eigen::Matrix4d& T_lidar_to_cam) {
        std::ifstream file(calib_file);
        if (!file.is_open()) {
            std::cerr << "[Error] Cannot open OSDaR calibration file: " << calib_file << std::endl;
            return false;
        }

        const std::string want = camera_folder.empty() ? "rgb_center" : camera_folder;
        bool in_cam = false;
        bool cam_match = false;
        bool got_K = false;
        bool got_T_cam_to_parent = false;
        Eigen::Matrix4d T_cam_to_parent = Eigen::Matrix4d::Identity();

        std::string line;
        while (std::getline(file, line)) {
            const std::string trimmed = [&] {
                std::string t = line;
                t.erase(t.begin(), std::find_if(t.begin(), t.end(), [](unsigned char ch) { return !std::isspace(ch); }));
                t.erase(std::find_if(t.rbegin(), t.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), t.end());
                return t;
            }();

            if (trimmed == "CAMERA") {
                in_cam = true;
                cam_match = false;
                continue;
            }
            if (!in_cam) continue;

            if (trimmed.rfind("data_folder:", 0) == 0) {
                std::string folder = trimmed.substr(std::string("data_folder:").size());
                folder.erase(folder.begin(), std::find_if(folder.begin(), folder.end(), [](unsigned char ch) { return !std::isspace(ch); }));
                cam_match = (folder == want);
                continue;
            }
            if (!cam_match) continue;

            if (trimmed.rfind("camera_matrix:", 0) == 0) {
                std::string after_colon = trimmed.substr(std::string("camera_matrix:").size());
                std::vector<double> nums;
                if (!_collect_numbers_multiline(file, after_colon, 9, nums, "camera_matrix")) {
                    return false;
                }
                K << nums[0], nums[1], nums[2],
                     nums[3], nums[4], nums[5],
                     nums[6], nums[7], nums[8];
                got_K = true;
                continue;
            }

            if (trimmed.rfind("homogeneous transform:", 0) == 0) {
                std::string after_colon = trimmed.substr(std::string("homogeneous transform:").size());
                std::vector<double> nums;
                if (!_collect_numbers_multiline(file, after_colon, 16, nums, "homogeneous transform")) {
                    return false;
                }
                T_cam_to_parent << nums[0],  nums[1],  nums[2],  nums[3],
                                   nums[4],  nums[5],  nums[6],  nums[7],
                                   nums[8],  nums[9],  nums[10], nums[11],
                                   nums[12], nums[13], nums[14], nums[15];
                got_T_cam_to_parent = true;
                continue;
            }

            // Stop after we have both.
            if (got_K && got_T_cam_to_parent) {
                break;
            }
        }

        if (!got_K || !got_T_cam_to_parent) {
            std::cerr << "[Error] OSDaR calib missing K/T for camera_folder=" << want << " in " << calib_file << std::endl;
            return false;
        }

        // OSDaR23 reference/body frame: X=driving direction, Y=left, Z=up (calibration.txt header).
        // Camera optical frame (OpenCV/pinhole K convention): X=right, Y=down, Z=into-scene.
        // Step 1: lidar-frame -> camera body-frame: T_lidar_to_body = inv(T_cam_to_parent)
        // Step 2: camera body-frame -> camera optical-frame: multiply by R_body_to_optical (pure rotation)
        //   optical-X = -body-Y,  optical-Y = -body-Z,  optical-Z = body-X
        Eigen::Matrix4d T_lidar_to_body = T_cam_to_parent.inverse();

        Eigen::Matrix4d T_body_to_optical = Eigen::Matrix4d::Identity();
        T_body_to_optical(0, 0) =  0.0;  T_body_to_optical(0, 1) = -1.0;  T_body_to_optical(0, 2) =  0.0;
        T_body_to_optical(1, 0) =  0.0;  T_body_to_optical(1, 1) =  0.0;  T_body_to_optical(1, 2) = -1.0;
        T_body_to_optical(2, 0) =  1.0;  T_body_to_optical(2, 1) =  0.0;  T_body_to_optical(2, 2) =  0.0;

        T_lidar_to_cam = T_body_to_optical * T_lidar_to_body;

        std::cerr << "[OSDaR23] T_cam_to_parent t= ["
                  << T_cam_to_parent(0,3) << ", " << T_cam_to_parent(1,3) << ", " << T_cam_to_parent(2,3) << "]\n"
                  << "[OSDaR23] T_lidar_to_optical t= ["
                  << T_lidar_to_cam(0,3) << ", " << T_lidar_to_cam(1,3) << ", " << T_lidar_to_cam(2,3) << "]\n";
        return true;
    }
}

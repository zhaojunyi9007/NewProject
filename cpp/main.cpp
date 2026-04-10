#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "include/edge_calibrator.h"

int main(int argc, char** argv) {
    auto print_usage = []() {
        std::cerr
            << "Usage (legacy):\n"
            << "  ./optimizer <lidar_feature_base> <sam_feature_base> <calib_file> "
               "<init_rx> <init_ry> <init_rz> <init_tx> <init_ty> <init_tz> [output_file] [history_file]\n\n"
            << "Usage (Phase B2):\n"
            << "  ./optimizer --lidar_feature_base P --sam_feature_base P --calib_file P "
               "--init_rx X --init_ry X --init_rz X --init_tx X --init_ty X --init_tz X "
               "[--output_file P] [--history_file P]\n"
            << "            [--semantic_probs P] [--lidar_semantic_points P] [--init_pose_from_bev P]\n"
            << "            [--semantic_js_weight X] [--histogram_weight X] [--edge_weight X] [--rail_weight X]\n"
            << "            [--class_weights \"w0,w1,...\"] [--pyramid_scales \"1.0,0.5,0.25\"]\n"
            << "            [--mode full_calib|refine_only] [--max_delta_deg X] [--max_delta_m X]\n";
    };

    auto split_csv_doubles = [](const std::string& s, std::vector<double>* out) {
        out->clear();
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (item.empty()) continue;
            char* end = nullptr;
            double v = std::strtod(item.c_str(), &end);
            if (end != item.c_str()) out->push_back(v);
        }
    };

    EdgeCalibratorConfig config;

    // Legacy positional mode (keep compatibility).
    if (argc >= 2 && argv[1][0] != '-') {
        if (argc != 10 && argc != 11 && argc != 12) {
            print_usage();
            return -1;
        }
        config.lidar_base = argv[1];
        config.sam_base = argv[2];
        config.calib_file = argv[3];
        config.init_r[0] = std::atof(argv[4]);
        config.init_r[1] = std::atof(argv[5]);
        config.init_r[2] = std::atof(argv[6]);
        config.init_t[0] = std::atof(argv[7]);
        config.init_t[1] = std::atof(argv[8]);
        config.init_t[2] = std::atof(argv[9]);
        if (argc >= 11) config.output_file = argv[10];
        if (argc == 12) config.history_file = argv[11];
    } else {
        // Phase B2 flag mode.
        auto need = [&](int i) -> bool { return (i + 1) < argc; };
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--lidar_feature_base" && need(i)) config.lidar_base = argv[++i];
            else if (a == "--sam_feature_base" && need(i)) config.sam_base = argv[++i];
            else if (a == "--calib_file" && need(i)) config.calib_file = argv[++i];
            else if (a == "--output_file" && need(i)) config.output_file = argv[++i];
            else if (a == "--history_file" && need(i)) config.history_file = argv[++i];
            else if (a == "--init_rx" && need(i)) config.init_r[0] = std::atof(argv[++i]);
            else if (a == "--init_ry" && need(i)) config.init_r[1] = std::atof(argv[++i]);
            else if (a == "--init_rz" && need(i)) config.init_r[2] = std::atof(argv[++i]);
            else if (a == "--init_tx" && need(i)) config.init_t[0] = std::atof(argv[++i]);
            else if (a == "--init_ty" && need(i)) config.init_t[1] = std::atof(argv[++i]);
            else if (a == "--init_tz" && need(i)) config.init_t[2] = std::atof(argv[++i]);

            else if (a == "--semantic_probs" && need(i)) config.semantic_probs_path = argv[++i];
            else if (a == "--lidar_semantic_points" && need(i)) config.lidar_semantic_points_path = argv[++i];
            else if (a == "--init_pose_from_bev" && need(i)) config.init_pose_from_bev_path = argv[++i];
            else if (a == "--semantic_js_weight" && need(i)) config.semantic_js_weight = std::atof(argv[++i]);
            else if (a == "--histogram_weight" && need(i)) config.histogram_weight = std::atof(argv[++i]);
            else if (a == "--edge_weight" && need(i)) config.edge_weight = std::atof(argv[++i]);
            else if (a == "--rail_weight" && need(i)) config.rail_weight = std::atof(argv[++i]);
            else if (a == "--class_weights" && need(i)) split_csv_doubles(argv[++i], &config.class_weights);
            else if (a == "--pyramid_scales" && need(i)) split_csv_doubles(argv[++i], &config.pyramid_scales);

            else if (a == "--mode" && need(i)) config.mode = argv[++i];
            else if (a == "--max_delta_deg" && need(i)) config.max_delta_deg = std::atof(argv[++i]);
            else if (a == "--max_delta_m" && need(i)) config.max_delta_m = std::atof(argv[++i]);
            else {
                std::cerr << "[optimizer] Unknown or incomplete arg: " << a << std::endl;
                print_usage();
                return -1;
            }
        }
        if (config.lidar_base.empty() || config.sam_base.empty()) {
            std::cerr << "[optimizer] --lidar_feature_base and --sam_feature_base are required." << std::endl;
            print_usage();
            return -1;
        }
    }

    EdgeCalibrator calibrator(config);
    if (!calibrator.LoadData()) return -1;
    calibrator.PerformCoarseSearch();
    calibrator.PerformFineOptimization();
    calibrator.ApplyTemporalSmoothing();
    return calibrator.SaveResult() ? 0 : -1;
}

#include <cstdlib>
#include <iostream>
#include <string>

#include "include/edge_calibrator.h"

int main(int argc, char** argv) {
    if (argc != 10 && argc != 11 && argc != 12) {
        std::cerr << "Usage: ./optimizer <lidar_feature_base> <sam_feature_base> <calib_file> "
                  << "<init_rx> <init_ry> <init_rz> <init_tx> <init_ty> <init_tz> [output_file] [history_file]"
                  << std::endl;
        return -1;
    }

    EdgeCalibratorConfig config;
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

    EdgeCalibrator calibrator(config);
    if (!calibrator.LoadData()) return -1;
    calibrator.PerformCoarseSearch();
    calibrator.PerformFineOptimization();
    calibrator.ApplyTemporalSmoothing();
    return calibrator.SaveResult() ? 0 : -1;
}

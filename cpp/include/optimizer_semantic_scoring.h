#pragma once

// Phase 5/7 placeholder (Phase A1 unblocks build):
// This header defines a minimal semantic-scoring interface.
// A full implementation will be added in Phase B (todo.md).

#include "include/common.h"

#include <string>
#include <vector>

#include <eigen3/Eigen/Core>

struct SemanticProbMaps {
    int W = 0;
    int H = 0;
    int C = 0;
    std::vector<float> data;  // H*W*C, row-major (v,u,c)

    bool empty() const { return data.empty() || W <= 0 || H <= 0 || C <= 0; }
};

struct SemanticScoreBreakdown {
    double semantic_js_divergence = 0.0;
    double semantic_hist_similarity = 0.0;
};

struct SemanticScoringConfig {
    std::vector<double> class_weights;
    std::vector<double> pyramid_scales;
};

// Binary format exported by tools/export_semantic_probs_bin.py
// magic(8)='EDGESEM1' + int32(W,H,C) + float32 data.
bool LoadSemanticProbabilityMapsBinary(const std::string& path, SemanticProbMaps& out);

// Load LiDAR semantic points from lidar_extractor: <lidar_base>_semantic_points.txt
// Format: x y z intensity nx ny nz label weight semantic_id
bool LoadSemanticPoints(const std::string& path, std::vector<SemanticPointRecord>& out);

// Phase B1: minimal semantic-probability scoring (higher is better).
// - Projects LiDAR semantic points to image.
// - Samples semantic probability vector at projected pixels.
// - Computes:
//   * semantic_js_divergence (lower is better)
//   * semantic_hist_similarity (higher is better)
double ComputeSemanticJSDivergence(const std::vector<SemanticPointRecord>& lidar_sem,
                                  const SemanticProbMaps& image_probs,
                                  const Eigen::Matrix3d& R_rect,
                                  const Eigen::Matrix<double, 3, 4>& P_rect,
                                  const Eigen::Matrix3d& R,
                                  const Eigen::Vector3d& t,
                                  const SemanticScoringConfig& cfg,
                                  SemanticScoreBreakdown* breakdown);

double ComputeSemanticHistogramConsistency(const std::vector<SemanticPointRecord>& lidar_sem,
                                          const SemanticProbMaps& image_probs,
                                          const Eigen::Matrix3d& R_rect,
                                          const Eigen::Matrix<double, 3, 4>& P_rect,
                                          const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& t,
                                          const SemanticScoringConfig& cfg,
                                          SemanticScoreBreakdown* breakdown);

double ComputeSemanticScore(const std::vector<SemanticPointRecord>& lidar_sem,
                            const SemanticProbMaps& image_probs,
                            const Eigen::Matrix3d& R_rect,
                            const Eigen::Matrix<double, 3, 4>& P_rect,
                            const Eigen::Matrix3d& R,
                            const Eigen::Vector3d& t,
                            const SemanticScoringConfig& cfg,
                            SemanticScoreBreakdown* breakdown);


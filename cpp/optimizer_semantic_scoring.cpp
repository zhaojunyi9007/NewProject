#include "include/optimizer_semantic_scoring.h"

#include "include/common.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

inline bool ProjectPoint(const Eigen::Vector3d& p_lidar,
                         const Eigen::Matrix3d& R_rect,
                         const Eigen::Matrix<double, 3, 4>& P_rect,
                         const Eigen::Matrix3d& R,
                         const Eigen::Vector3d& t,
                         int W,
                         int H,
                         int& u,
                         int& v) {
    Eigen::Vector3d p_cam = R * p_lidar + t;
    Eigen::Vector3d p_rect = R_rect * p_cam;
    if (p_rect.z() < 0.1) return false;
    Eigen::Vector4d p_rect_h;
    p_rect_h << p_rect.x(), p_rect.y(), p_rect.z(), 1.0;
    Eigen::Vector3d uv = P_rect * p_rect_h;
    const double uu = uv.x() / uv.z();
    const double vv = uv.y() / uv.z();
    u = static_cast<int>(std::floor(uu));
    v = static_cast<int>(std::floor(vv));
    return (u >= 0 && u < W && v >= 0 && v < H);
}

inline void NormalizeProb(std::vector<double>* p) {
    double s = 0.0;
    for (double x : *p) s += x;
    if (s < 1e-14) return;
    for (double& x : *p) x /= s;
}

inline void ApplyClassWeights(std::vector<double>* p, const std::vector<double>& w) {
    const int C = static_cast<int>(p->size());
    for (int c = 0; c < C; ++c) {
        const double wc = (c < static_cast<int>(w.size()) && w[c] > 0.0) ? w[c] : 1.0;
        (*p)[static_cast<size_t>(c)] *= wc;
    }
    NormalizeProb(p);
}

inline double JSD(const std::vector<double>& P, const std::vector<double>& Q) {
    const double eps = 1e-12;
    const int C = static_cast<int>(P.size());
    double js = 0.0;
    for (int c = 0; c < C; ++c) {
        const double p = P[c];
        const double q = Q[c];
        const double m = 0.5 * (p + q);
        js += 0.5 * p * std::log((p + eps) / (m + eps)) + 0.5 * q * std::log((q + eps) / (m + eps));
    }
    return js;
}

inline double L1(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) s += std::abs(a[i] - b[i]);
    return s;
}

inline void OneHot(int idx, int C, std::vector<double>* out) {
    out->assign(static_cast<size_t>(C), 0.0);
    if (idx >= 0 && idx < C) (*out)[static_cast<size_t>(idx)] = 1.0;
}

inline bool SampleSemanticProbNN(const SemanticProbMaps& m, int u, int v, std::vector<double>* out) {
    out->assign(static_cast<size_t>(m.C), 0.0);
    if (m.empty()) return false;
    if (u < 0 || u >= m.W || v < 0 || v >= m.H) return false;
    const size_t base = (static_cast<size_t>(v) * static_cast<size_t>(m.W) + static_cast<size_t>(u)) *
                        static_cast<size_t>(m.C);
    for (int c = 0; c < m.C; ++c) {
        (*out)[static_cast<size_t>(c)] = static_cast<double>(m.data[base + static_cast<size_t>(c)]);
    }
    return true;
}

}  // namespace

bool LoadSemanticProbabilityMapsBinary(const std::string& path, SemanticProbMaps& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[LoadSemanticProbabilityMapsBinary] Cannot open " << path << std::endl;
        return false;
    }
    char magic[8];
    f.read(magic, 8);
    if (f.gcount() != 8 || std::strncmp(magic, "EDGESEM1", 8) != 0) {
        std::cerr << "[LoadSemanticProbabilityMapsBinary] Bad magic in " << path << std::endl;
        return false;
    }
    int W = 0, H = 0, C = 0;
    f.read(reinterpret_cast<char*>(&W), sizeof(int));
    f.read(reinterpret_cast<char*>(&H), sizeof(int));
    f.read(reinterpret_cast<char*>(&C), sizeof(int));
    if (!f || W <= 0 || H <= 0 || C <= 0) {
        std::cerr << "[LoadSemanticProbabilityMapsBinary] Bad dims in " << path << std::endl;
        return false;
    }
    const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C);
    out.W = W;
    out.H = H;
    out.C = C;
    out.data.resize(n);
    f.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) {
        std::cerr << "[LoadSemanticProbabilityMapsBinary] Short read in " << path << std::endl;
        return false;
    }
    return true;
}

bool LoadSemanticPoints(const std::string& path, std::vector<SemanticPointRecord>& out) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[LoadSemanticPoints] Cannot open " << path << std::endl;
        return false;
    }
    out.clear();
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double x, y, z, i, nx, ny, nz;
        int label = 0;
        double weight = 1.0;
        int semantic_id = -1;
        if (!(ss >> x >> y >> z >> i >> nx >> ny >> nz >> label >> weight >> semantic_id)) {
            continue;
        }
        SemanticPointRecord r;
        r.p = Eigen::Vector3d(x, y, z);
        r.intensity = static_cast<float>(i);
        r.normal = Eigen::Vector3d(nx, ny, nz);
        r.label = label;
        r.weight = weight;
        r.semantic_id = semantic_id;
        out.push_back(r);
    }
    return !out.empty();
}

double ComputeSemanticJSDivergence(const std::vector<SemanticPointRecord>& lidar_sem,
                                  const SemanticProbMaps& image_probs,
                                  const Eigen::Matrix3d& R_rect,
                                  const Eigen::Matrix<double, 3, 4>& P_rect,
                                  const Eigen::Matrix3d& R,
                                  const Eigen::Vector3d& t,
                                  const SemanticScoringConfig& cfg,
                                  SemanticScoreBreakdown* breakdown) {
    if (image_probs.empty() || lidar_sem.empty()) {
        if (breakdown) {
            breakdown->semantic_js_divergence = 0.0;
        }
        return 0.0;
    }
    const int C = image_probs.C;
    std::vector<double> Pdist(static_cast<size_t>(C), 0.0);
    std::vector<double> Qdist(static_cast<size_t>(C), 0.0);
    std::vector<double> p_one;
    std::vector<double> q_uv;
    int used = 0;
    for (const auto& sp : lidar_sem) {
        int u = 0, v = 0;
        if (!ProjectPoint(sp.p, R_rect, P_rect, R, t, image_probs.W, image_probs.H, u, v)) continue;
        if (sp.semantic_id >= 0 && sp.semantic_id < C) {
            OneHot(sp.semantic_id, C, &p_one);
        } else {
            OneHot(sp.label, C, &p_one);
        }
        if (!SampleSemanticProbNN(image_probs, u, v, &q_uv)) continue;
        NormalizeProb(&q_uv);
        for (int c = 0; c < C; ++c) {
            Pdist[static_cast<size_t>(c)] += p_one[static_cast<size_t>(c)];
            Qdist[static_cast<size_t>(c)] += q_uv[static_cast<size_t>(c)];
        }
        used++;
    }
    if (used < 5) {
        if (breakdown) breakdown->semantic_js_divergence = 0.0;
        return 0.0;
    }
    NormalizeProb(&Pdist);
    NormalizeProb(&Qdist);
    ApplyClassWeights(&Pdist, cfg.class_weights);
    ApplyClassWeights(&Qdist, cfg.class_weights);

    const double js = JSD(Pdist, Qdist);
    if (breakdown) breakdown->semantic_js_divergence = js;
    return js;
}

double ComputeSemanticHistogramConsistency(const std::vector<SemanticPointRecord>& lidar_sem,
                                          const SemanticProbMaps& image_probs,
                                          const Eigen::Matrix3d& R_rect,
                                          const Eigen::Matrix<double, 3, 4>& P_rect,
                                          const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& t,
                                          const SemanticScoringConfig& cfg,
                                          SemanticScoreBreakdown* breakdown) {
    if (image_probs.empty() || lidar_sem.empty()) {
        if (breakdown) {
            breakdown->semantic_hist_similarity = 0.0;
        }
        return 0.0;
    }
    const int C = image_probs.C;
    std::vector<double> Pdist(static_cast<size_t>(C), 0.0);
    std::vector<double> Qdist(static_cast<size_t>(C), 0.0);
    std::vector<double> p_one;
    std::vector<double> q_uv;
    int used = 0;
    for (const auto& sp : lidar_sem) {
        int u = 0, v = 0;
        if (!ProjectPoint(sp.p, R_rect, P_rect, R, t, image_probs.W, image_probs.H, u, v)) continue;
        if (sp.semantic_id >= 0 && sp.semantic_id < C) {
            OneHot(sp.semantic_id, C, &p_one);
        } else {
            OneHot(sp.label, C, &p_one);
        }
        if (!SampleSemanticProbNN(image_probs, u, v, &q_uv)) continue;
        NormalizeProb(&q_uv);
        for (int c = 0; c < C; ++c) {
            Pdist[static_cast<size_t>(c)] += p_one[static_cast<size_t>(c)];
            Qdist[static_cast<size_t>(c)] += q_uv[static_cast<size_t>(c)];
        }
        used++;
    }
    if (used < 5) {
        if (breakdown) breakdown->semantic_hist_similarity = 0.0;
        return 0.0;
    }
    NormalizeProb(&Pdist);
    NormalizeProb(&Qdist);
    ApplyClassWeights(&Pdist, cfg.class_weights);
    ApplyClassWeights(&Qdist, cfg.class_weights);
    const double l1 = L1(Pdist, Qdist);
    const double sim = std::max(0.0, 1.0 - 0.5 * l1);  // [0,1]
    if (breakdown) breakdown->semantic_hist_similarity = sim;
    return sim;
}

double ComputeSemanticScore(const std::vector<SemanticPointRecord>& lidar_sem,
                            const SemanticProbMaps& image_probs,
                            const Eigen::Matrix3d& R_rect,
                            const Eigen::Matrix<double, 3, 4>& P_rect,
                            const Eigen::Matrix3d& R,
                            const Eigen::Vector3d& t,
                            const SemanticScoringConfig& cfg,
                            SemanticScoreBreakdown* breakdown) {
    SemanticScoreBreakdown bd;
    const double js = ComputeSemanticJSDivergence(lidar_sem, image_probs, R_rect, P_rect, R, t, cfg, &bd);
    const double hist = ComputeSemanticHistogramConsistency(lidar_sem, image_probs, R_rect, P_rect, R, t, cfg, &bd);
    if (breakdown) *breakdown = bd;
    // Higher is better: penalize divergence, reward histogram similarity.
    return (-js) + 0.5 * hist;
}


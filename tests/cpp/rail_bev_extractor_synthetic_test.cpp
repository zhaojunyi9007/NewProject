#include "include/rail_bev_extractor.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

BEVChannels MakeGrid() {
    BEVChannels bev;
    bev.nx = 500;
    bev.ny = 250;
    bev.xmin = 0.0;
    bev.ymin = -25.0;
    bev.resolution = 0.2;
    const int n = bev.nx * bev.ny;
    bev.density.assign(n, 0.0f);
    bev.max_height.assign(n, 0.0f);
    bev.min_height.assign(n, 0.0f);
    bev.mean_height.assign(n, 0.0f);
    bev.mean_intensity.assign(n, 0.0f);
    bev.verticality.assign(n, 0.0f);
    bev.rail_probability.assign(n, 0.0f);
    return bev;
}

void SetCell(BEVChannels* bev, int x, int y, float value) {
    if (!bev || x < 0 || x >= bev->nx || y < 0 || y >= bev->ny) return;
    bev->rail_probability[static_cast<size_t>(y * bev->nx + x)] = value;
}

void DrawMetricLine(BEVChannels* bev, double x0, double y0, double x1, double y1, double half_width_m, float value) {
    const int samples = static_cast<int>(std::ceil(std::hypot(x1 - x0, y1 - y0) / bev->resolution)) * 3;
    const int r = std::max(1, static_cast<int>(std::ceil(half_width_m / bev->resolution)));
    for (int i = 0; i <= samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(std::max(1, samples));
        const double x = x0 + (x1 - x0) * t;
        const double y = y0 + (y1 - y0) * t;
        const int ix = static_cast<int>(std::floor((x - bev->xmin) / bev->resolution));
        const int iy = static_cast<int>(std::floor((y - bev->ymin) / bev->resolution));
        for (int oy = -r; oy <= r; ++oy) {
            for (int ox = -r; ox <= r; ++ox) {
                SetCell(bev, ix + ox, iy + oy, value);
            }
        }
    }
}

RailPriorConfig TestConfig() {
    RailPriorConfig cfg;
    cfg.rail_prob_thresh = 0.15f;
    cfg.min_component_cells = 10.0f;
    cfg.smooth_ksize = 1;
    cfg.min_length_m = 8.0f;
    cfg.max_width_m = 1.2f;
    cfg.min_linearity = 8.0f;
    cfg.gauge_m = 1.435f;
    cfg.gauge_tolerance_m = 0.45f;
    cfg.max_parallel_angle_deg = 5.0f;
    cfg.min_pair_overlap_m = 5.0f;
    cfg.morph_open_kernel_cells = 1;
    cfg.hough_threshold = 8;
    cfg.hough_min_line_length_m = 8.0f;
    cfg.hough_max_line_gap_m = 1.2f;
    cfg.accepted_line_band_m = 0.35f;
    cfg.blob_max_width_m = 1.8f;
    cfg.blob_max_fill_ratio = 0.45f;
    cfg.blob_min_linearity = 4.0f;
    cfg.blob_max_area_cells = 3500;
    return cfg;
}

void Expect(bool cond, const std::string& message) {
    if (!cond) throw std::runtime_error(message);
}

float NonzeroRatio(const std::vector<float>& v) {
    int n = 0;
    for (float x : v) if (x > 1e-4f) ++n;
    return v.empty() ? 0.0f : static_cast<float>(n) / static_cast<float>(v.size());
}

void TestBlobNoiseRejected() {
    BEVChannels bev = MakeGrid();
    for (int y = 75; y < 175; ++y) for (int x = 80; x < 420; ++x) SetCell(&bev, x, y, 0.8f);
    RailBEVDebug dbg;
    float conf = -1.0f;
    bool branch = false;
    std::vector<float> refined;
    auto lines = ExtractMultiRailLinesFromBEV(bev, TestConfig(), 0.0, 6, &conf, &branch, &dbg, &refined);
    Expect(lines.empty(), "blob noise must not create rail lines");
    Expect(dbg.pair_count == 0, "blob noise pair_count must be zero");
    Expect(conf == 0.0f, "blob noise rail confidence must be zero");
    Expect(NonzeroRatio(refined) == 0.0f, "blob noise refined rail must be empty");
}

void TestParallelGaugePairAccepted() {
    BEVChannels bev = MakeGrid();
    DrawMetricLine(&bev, 12.0, -0.72, 42.0, -0.72, 0.12, 0.95f);
    DrawMetricLine(&bev, 12.0, 0.715, 42.0, 0.715, 0.12, 0.95f);
    RailBEVDebug dbg;
    float conf = 0.0f;
    bool branch = false;
    std::vector<float> refined;
    auto lines = ExtractMultiRailLinesFromBEV(bev, TestConfig(), 0.0, 6, &conf, &branch, &dbg, &refined);
    Expect(dbg.pair_count >= 1, "parallel gauge pair should be detected");
    Expect(lines.size() >= 2, "parallel gauge pair should emit at least two rail lines");
    Expect(conf > 0.0f, "parallel gauge pair should produce confidence");
    Expect(std::abs(dbg.best_pair_gauge_m - 1.435f) <= TestConfig().gauge_tolerance_m, "best pair gauge should be close to standard gauge");
    Expect(NonzeroRatio(refined) > 0.0f, "accepted pair should produce refined rail map");
}

void TestWrongGaugeRejected() {
    BEVChannels bev = MakeGrid();
    DrawMetricLine(&bev, 12.0, -1.5, 42.0, -1.5, 0.12, 0.95f);
    DrawMetricLine(&bev, 12.0, 1.5, 42.0, 1.5, 0.12, 0.95f);
    RailBEVDebug dbg;
    float conf = -1.0f;
    bool branch = false;
    std::vector<float> refined;
    auto lines = ExtractMultiRailLinesFromBEV(bev, TestConfig(), 0.0, 6, &conf, &branch, &dbg, &refined);
    Expect(lines.empty(), "wrong gauge must not emit rail lines");
    Expect(dbg.pair_count == 0, "wrong gauge pair_count must be zero");
    Expect(conf == 0.0f, "wrong gauge confidence must be zero");
}

void TestNoisyPairPreferredOverBlob() {
    BEVChannels bev = MakeGrid();
    for (int y = 20; y < 85; ++y) for (int x = 320; x < 480; ++x) SetCell(&bev, x, y, 0.75f);
    DrawMetricLine(&bev, 14.0, -0.72, 48.0, -0.72, 0.12, 0.95f);
    DrawMetricLine(&bev, 14.0, 0.715, 48.0, 0.715, 0.12, 0.95f);
    RailBEVDebug dbg;
    float conf = 0.0f;
    bool branch = false;
    std::vector<float> refined;
    auto lines = ExtractMultiRailLinesFromBEV(bev, TestConfig(), 0.0, 6, &conf, &branch, &dbg, &refined);
    Expect(dbg.rejected_blob_component_count >= 1, "blob should be rejected before pairing");
    Expect(dbg.pair_count >= 1, "valid pair should survive blob noise");
    Expect(lines.size() >= 2, "valid pair should emit rail lines despite blob noise");
    Expect(NonzeroRatio(refined) > 0.0f, "valid pair should produce refined rail map");
}

}  // namespace

int main() {
    try {
        TestBlobNoiseRejected();
        TestParallelGaugePairAccepted();
        TestWrongGaugeRejected();
        TestNoisyPairPreferredOverBlob();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "rail_bev_extractor_synthetic_test: PASS" << std::endl;
    return 0;
}

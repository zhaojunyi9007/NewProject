#include "include/bev_io.h"

#include <cstring>
#include <fstream>
#include <iostream>

bool SaveBEVChannelsRaw(const std::string& output_base, const BEVChannels& bev) {
    const int n = bev.nx * bev.ny;
    if (n <= 0) return false;

    const std::string bin_path = output_base + "_bev_channels.bin";
    std::ofstream ofs(bin_path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[BEV] Cannot write " << bin_path << std::endl;
        return false;
    }
    const int nch = 7;
    const char magic[8] = {'E', 'D', 'G', 'E', 'B', 'E', 'V', '1'};
    ofs.write(magic, 8);
    const int32_t dims[3] = {bev.nx, bev.ny, nch};
    ofs.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    const float meta[4] = {static_cast<float>(bev.xmin), static_cast<float>(bev.ymin), static_cast<float>(bev.resolution), 0.f};
    ofs.write(reinterpret_cast<const char*>(meta), sizeof(meta));

    auto write_ch = [&](const std::vector<float>& v) {
        ofs.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    };
    write_ch(bev.density);
    write_ch(bev.max_height);
    write_ch(bev.min_height);
    write_ch(bev.mean_height);
    write_ch(bev.mean_intensity);
    write_ch(bev.verticality);
    write_ch(bev.rail_probability);
    ofs.close();

    const std::string json_path = output_base + "_bev_meta.json";
    std::ofstream jf(json_path);
    if (!jf) return false;
    jf << "{\"nx\":" << bev.nx << ",\"ny\":" << bev.ny
       << ",\"xmin\":" << bev.xmin << ",\"ymin\":" << bev.ymin
       << ",\"resolution\":" << bev.resolution
       << ",\"channels\":[\"density\",\"max_height\",\"min_height\",\"mean_height\","
          "\"mean_intensity\",\"verticality\",\"rail_probability\"]"
       << ",\"binary\":\"" << bin_path << "\"}\n";
    jf.close();
    std::cout << "[BEV] Saved raw BEV: " << bin_path << std::endl;
    return true;
}

bool LoadBEVChannelsBin(const std::string& path, BEVChannels* out) {
    if (!out) return false;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "[BEV] Cannot open " << path << std::endl;
        return false;
    }
    char magic[8];
    ifs.read(magic, 8);
    if (ifs.gcount() != 8 || std::strncmp(magic, "EDGEBEV1", 8) != 0) {
        std::cerr << "[BEV] Bad magic in " << path << std::endl;
        return false;
    }
    int32_t nx = 0, ny = 0, nch = 0;
    ifs.read(reinterpret_cast<char*>(&nx), sizeof(nx));
    ifs.read(reinterpret_cast<char*>(&ny), sizeof(ny));
    ifs.read(reinterpret_cast<char*>(&nch), sizeof(nch));
    float xmin = 0.f, ymin = 0.f, res = 0.f, pad = 0.f;
    ifs.read(reinterpret_cast<char*>(&xmin), sizeof(xmin));
    ifs.read(reinterpret_cast<char*>(&ymin), sizeof(ymin));
    ifs.read(reinterpret_cast<char*>(&res), sizeof(res));
    ifs.read(reinterpret_cast<char*>(&pad), sizeof(pad));
    if (nx <= 0 || ny <= 0 || nch <= 0) {
        std::cerr << "[BEV] Invalid dims\n";
        return false;
    }
    const int n = nx * ny;
    const size_t total_floats = static_cast<size_t>(n) * static_cast<size_t>(nch);
    std::vector<float> all(total_floats);
    ifs.read(reinterpret_cast<char*>(all.data()), static_cast<std::streamsize>(total_floats * sizeof(float)));
    if (static_cast<size_t>(ifs.gcount()) != total_floats * sizeof(float)) {
        std::cerr << "[BEV] Short read\n";
        return false;
    }

    out->nx = nx;
    out->ny = ny;
    out->xmin = static_cast<double>(xmin);
    out->ymin = static_cast<double>(ymin);
    out->resolution = static_cast<double>(res);
    out->density.assign(static_cast<size_t>(n), 0.f);
    out->max_height.assign(static_cast<size_t>(n), 0.f);
    out->min_height.assign(static_cast<size_t>(n), 0.f);
    out->mean_height.assign(static_cast<size_t>(n), 0.f);
    out->mean_intensity.assign(static_cast<size_t>(n), 0.f);
    out->verticality.assign(static_cast<size_t>(n), 0.f);
    out->rail_probability.assign(static_cast<size_t>(n), 0.f);

    if (nch == 1) {
        out->rail_probability.assign(all.begin(), all.begin() + static_cast<size_t>(n));
        std::cout << "[BEV] Loaded " << path << " (" << nx << "x" << ny << " rail-only)\n";
        return true;
    }

    for (int c = 0; c < nch; ++c) {
        const size_t off = static_cast<size_t>(c) * static_cast<size_t>(n);
        for (int i = 0; i < n; ++i) {
            const float v = all[off + static_cast<size_t>(i)];
            switch (c) {
                case 0: out->density[static_cast<size_t>(i)] = v; break;
                case 1: out->max_height[static_cast<size_t>(i)] = v; break;
                case 2: out->min_height[static_cast<size_t>(i)] = v; break;
                case 3: out->mean_height[static_cast<size_t>(i)] = v; break;
                case 4: out->mean_intensity[static_cast<size_t>(i)] = v; break;
                case 5: out->verticality[static_cast<size_t>(i)] = v; break;
                case 6: out->rail_probability[static_cast<size_t>(i)] = v; break;
                default: break;
            }
        }
    }
    std::cout << "[BEV] Loaded " << path << " (" << nx << "x" << ny << " x " << nch << " ch)\n";
    return true;
}

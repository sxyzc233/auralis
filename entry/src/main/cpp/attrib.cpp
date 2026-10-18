#include "attrib.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace vh {
namespace {

// Shapley weights for |P| <= 3 path features
double weight(int path_len, int rest) {
    static const double f[5] = {1.0, 1.0, 2.0, 6.0, 24.0};
    return f[rest] * f[path_len - rest - 1] / f[path_len];
}

int popcount(unsigned x) {
    int c = 0;
    while (x) { c += (int)(x & 1u); x >>= 1; }
    return c;
}
}  // namespace

bool AttribModel::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    auto rd = [&](void* buf, size_t bytes) { return fread(buf, 1, bytes, f) == bytes; };

    char magic[4];
    if (!rd(magic, 4) || std::memcmp(magic, "VHA1", 4) != 0) { fclose(f); return false; }
    int hdr[3];
    if (!rd(hdr, sizeof(hdr))) { fclose(f); return false; }
    n_features_ = hdr[0]; n_trees_ = hdr[1]; n_families_ = hdr[2];
    float sc[3];
    if (!rd(sc, sizeof(sc))) { fclose(f); return false; }
    threshold_ = sc[0]; init_ = sc[1]; lr_ = sc[2];

    mean_.resize(n_features_); std_.resize(n_features_); feature_family_.resize(n_features_);
    if (!rd(mean_.data(), sizeof(float) * n_features_)
        || !rd(std_.data(), sizeof(float) * n_features_)
        || !rd(feature_family_.data(), sizeof(int) * n_features_)) { fclose(f); return false; }

    trees_.resize(n_trees_);
    for (int i = 0; i < n_trees_; ++i) {
        AttribTree& t = trees_[i];
        if (!rd(&t.n, sizeof(int))) { fclose(f); return false; }
        t.feature.resize(t.n); t.threshold.resize(t.n);
        t.left.resize(t.n); t.right.resize(t.n);
        t.value.resize(t.n); t.cover.resize(t.n);
        if (!rd(t.feature.data(), sizeof(int) * t.n)
            || !rd(t.threshold.data(), sizeof(float) * t.n)
            || !rd(t.left.data(), sizeof(int) * t.n)
            || !rd(t.right.data(), sizeof(int) * t.n)
            || !rd(t.value.data(), sizeof(float) * t.n)
            || !rd(t.cover.data(), sizeof(float) * t.n)) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

float AttribModel::subtreeValue(const AttribTree& t, int node, const double* xs,
                                unsigned fixed_mask, const int* path, int path_len) const {
    const int f = t.feature[node];
    if (f == -2) return t.value[node];
    bool fixed = false;
    for (int i = 0; i < path_len; ++i)
        if (path[i] == f && (fixed_mask >> i & 1u)) { fixed = true; break; }
    if (fixed) {
        const int child = (xs[f] <= (double)t.threshold[node]) ? t.left[node] : t.right[node];
        return subtreeValue(t, child, xs, fixed_mask, path, path_len);
    }
    const float wl = t.cover[t.left[node]], wr = t.cover[t.right[node]];
    const float vl = subtreeValue(t, t.left[node], xs, fixed_mask, path, path_len);
    const float vr = subtreeValue(t, t.right[node], xs, fixed_mask, path, path_len);
    const float den = wl + wr;
    return den > 0.f ? (wl * vl + wr * vr) / den : 0.5f * (vl + vr);
}

void AttribModel::score(const float* feats, float* out_families,
                        float* out_family_shap, float* out_feature_shap) const {
    std::vector<double> xs(n_features_);
    for (int i = 0; i < n_features_; ++i) xs[i] = ((double)feats[i] - mean_[i]) / std_[i];

    std::vector<double> phi(n_features_, 0.0);
    int path[4];
    for (const AttribTree& t : trees_) {
        // unique path features
        int path_len = 0;
        int node = 0;
        while (t.feature[node] != -2) {
            const int f = t.feature[node];
            bool seen = false;
            for (int i = 0; i < path_len; ++i) if (path[i] == f) seen = true;
            if (!seen) path[path_len++] = f;
            node = (xs[f] <= (double)t.threshold[node]) ? t.left[node] : t.right[node];
        }
        auto F = [&](unsigned mask) { return (double)subtreeValue(t, 0, xs.data(), mask, path, path_len); };
        for (int j = 0; j < path_len; ++j) {
            const unsigned jbit = 1u << j;
            double tot = 0.0;
            const unsigned full = (path_len >= 32) ? 0u : (1u << path_len);
            for (unsigned mask = 0; mask < full; ++mask) {
                if (mask & jbit) continue;
                const int rest = popcount(mask);
                tot += weight(path_len, rest) * (F(mask | jbit) - F(mask));
            }
            phi[path[j]] += tot;
        }
    }

    std::vector<double> fam((size_t)n_families_, 0.0);
    for (int i = 0; i < n_features_; ++i) {
        const double v = phi[i] * (double)lr_;
        if (out_feature_shap) out_feature_shap[i] = (float)v;
        const int fi = feature_family_[i];
        if (fi >= 0 && fi < n_families_) fam[fi] += v;
    }
    for (int i = 0; i < n_families_; ++i) {
        if (out_family_shap) out_family_shap[i] = (float)fam[i];
        double s = 5.0 * (1.0 - fam[i] / 0.5);       // same mapping as the Python service
        if (s < 1.0) s = 1.0;
        if (s > 10.0) s = 10.0;
        out_families[i] = (float)std::lround(s);
    }
}

void AttribModel::judge(const float* feats, float* out_prob, int* out_label) const {
    std::vector<float> x(n_features_);
    for (int i = 0; i < n_features_; ++i) x[i] = (feats[i] - mean_[i]) / std_[i];

    float tot = 0.f;
    for (const AttribTree& t : trees_) {
        int n = 0;
        while (t.feature[n] != -2)
            n = (x[t.feature[n]] <= t.threshold[n]) ? t.left[n] : t.right[n];
        tot += t.value[n];
    }
    const float raw = init_ + lr_ * tot;
    const float p = 1.f / (1.f + std::exp(-raw));
    *out_prob = p;
    *out_label = (p > threshold_) ? 1 : 0;
}

}  // namespace vh

// ---------------------------------------------------------------------------
// Standalone validation harness (not compiled into the Android library):
//   attrib_test <attrib.bin> <features.bin> <out.bin>
//   features.bin = N x 55 float32 ; out.bin = N x (8 families + 55 shap) float32
// ---------------------------------------------------------------------------
#ifdef ATTRIB_TEST
int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: attrib_test <attrib.bin> <features.bin> <out.bin>\n"); return 1; }
    vh::AttribModel m;
    if (!m.load(argv[1])) { printf("load failed\n"); return 1; }
    const int nf = m.numFeatures();

    FILE* fi = fopen(argv[2], "rb");
    if (!fi) { printf("cannot open features\n"); return 1; }
    fseek(fi, 0, SEEK_END); long sz = ftell(fi); fseek(fi, 0, SEEK_SET);
    const int n = (int)(sz / (sizeof(float) * nf));
    std::vector<float> feats((size_t)n * nf);
    if (fread(feats.data(), sizeof(float), feats.size(), fi) != feats.size()) { printf("short read\n"); return 1; }
    fclose(fi);

    std::vector<float> out((size_t)n * (8 + 55));
    std::vector<float> fam(8), fshap(8), pshap(nf);
    for (int i = 0; i < n; ++i) {
        m.score(&feats[(size_t)i * nf], fam.data(), fshap.data(), pshap.data());
        for (int k = 0; k < 8; ++k) out[(size_t)i * 63 + k] = fam[k];
        for (int k = 0; k < 55; ++k) out[(size_t)i * 63 + 8 + k] = pshap[k];
    }
    FILE* fo = fopen(argv[3], "wb");
    fwrite(out.data(), sizeof(float), out.size(), fo);
    fclose(fo);
    printf("scored %d samples -> %s\n", n, argv[3]);
    return 0;
}
#endif

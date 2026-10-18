// Family attribution (exact tree-SHAP, grouped by semantic family) for the on-device engine.
// Reads the compact binary packed by work/gen_attrib_bin.py.
#pragma once

#include <string>
#include <vector>

namespace vh {

constexpr int kNumFamilies = 8;

struct AttribTree {
    int n = 0;
    std::vector<int> feature;
    std::vector<float> threshold;
    std::vector<int> left, right;
    std::vector<float> value, cover;
};

class AttribModel {
public:
    bool load(const std::string& path);

    // feats: 55 raw feature values in the GBDT's order.
    // out_families: kNumFamilies health scores (1-10, higher = healthier), display order.
    // out_family_shap / out_feature_shap: optional raw SHAP (logit units).
    void score(const float* feats, float* out_families,
               float* out_family_shap = nullptr, float* out_feature_shap = nullptr) const;

    // Health judgement: raw = init + lr*sum(leaf values), p = sigmoid(raw), label = p > threshold.
    // Accumulated in float32 so it matches the exported ONNX graph bit-for-bit.
    void judge(const float* feats, float* out_prob, int* out_label) const;

    int numFeatures() const { return n_features_; }
    float threshold() const { return threshold_; }

private:
    float subtreeValue(const AttribTree& t, int node, const double* xs,
                       unsigned fixed_mask, const int* path, int path_len) const;

    int n_features_ = 0, n_trees_ = 0, n_families_ = 0;
    float threshold_ = 0.f, init_ = 0.f, lr_ = 0.f;
    std::vector<float> mean_, std_;
    std::vector<int> feature_family_;
    std::vector<AttribTree> trees_;
};

}  // namespace vh

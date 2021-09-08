//
// Created by James Noeckel on 4/14/20.
//

#pragma once
#include <vector>
#include <string>
#include <ostream>
#include "construction/FeatureExtractor.h"

class Solution {
public:
    explicit Solution(FeatureExtractor &features);
    bool Load(const std::string &filename);
    void serialize(std::ostream &o) const;
    std::vector<int> part_ids;
    std::vector<int> depths;
    std::vector<int> shapes;
    int num_samples;
private:
    FeatureExtractor &features_;
};

std::ostream &operator<<(std::ostream &o, const Solution &sol);

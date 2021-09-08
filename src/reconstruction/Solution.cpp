//
// Created by James Noeckel on 4/14/20.
//

#include "Solution.h"
#include <fstream>
#include "utils/sorted_data_structures.hpp"

Solution::Solution(FeatureExtractor &features_) : features_(features_) {}

bool Solution::Load(const std::string &filename) {
    std::ifstream is(filename);
    if (!is) {
        std::cerr << "solution file not found" << std::endl;
        return false;
    }
    std::string input_line;
    if (!std::getline(is, input_line) || input_line.empty()) {
        std::cerr << "empty line found instead of solution" << std::endl;
        return false;
    } else {
        std::istringstream line_is(input_line);
        if (!(line_is >> num_samples)) {
            std::cerr << "expected sample count on first line" << std::endl;
            return false;
        }
    }
    while (std::getline(is, input_line) && !input_line.empty()) {
        int ind, depth, shape;
        std::istringstream line_is(input_line);
        if (!(line_is >> ind >> depth >> shape)) {
            std::cerr << "failed to parse solution file" << std::endl;
            return false;
        }
        part_ids.push_back(ind);
        depths.push_back(depth);
        shapes.push_back(shape);
    }
    return true;
}

void Solution::serialize(std::ostream &o) const {
    o << "<solution>" << std::endl;
    for (int i = 0; i < part_ids.size(); i++) {
        if (!features_.planes[part_ids[i]].hasShape()) continue;
        o << "<part id=\"" << part_ids[i] << "\" depth=\"" << features_.depths[part_ids[i]][depths[i]] << "\">" << std::endl;
        o << features_.planes[part_ids[i]] << std::endl;
        o << "</part>" << std::endl;
    }
    for (int i=0; i < part_ids.size() - 1; i++) {
        if (!features_.planes[part_ids[i]].hasShape()) continue;
        for (int j=i+1; j < part_ids.size(); j++) {
            if (!features_.planes[part_ids[j]].hasShape()) continue;
            if (features_.adjacency[part_ids[i]].find(part_ids[j]) != features_.adjacency[part_ids[i]].end()) {
                o << "<connection id1=\"" << part_ids[i] << "\" id2=\"" << part_ids[j] << "\"/>" << std::endl;
            }
        }
        for (const auto &pair : features_.conditional_connections[i]) {
            if (depths[i] >= pair.first && pair.second.first > i && sorted_find(part_ids, pair.second.first) != part_ids.end() && features_.planes[pair.second.first].hasShape()) {
                o << "<connection id1=\"" << part_ids[i] << "\" id2=\"" << pair.second.first << "\"/>" << std::endl;
            }
        }
    }
    o << "</solution>" << std::endl;
}

std::ostream &operator<<(std::ostream &o, const Solution &sol) {
    sol.serialize(o);
    return o;
}
//
// Created by James Noeckel on 4/7/20.
//

#pragma once
#include <vector>
#include <numeric>
#include <algorithm>

template <class ForwardIt, class Comp>
struct Compare {
    explicit Compare(ForwardIt begin, Comp comp) : begin_(begin), comp_(comp) {
    }
    bool operator()(size_t a, size_t b) {
        return comp_(*(begin_ + a), *(begin_ + b));
    }
    ForwardIt begin_;
    Comp comp_;
};

/**
 * @return vector of top k indices in descending order
 */
template <class ForwardIt, class Comp=std::less<>>
std::vector<size_t> top_k_indices(ForwardIt begin, ForwardIt end, size_t k, Comp comp=Comp()) {
    std::vector<size_t> indices(std::distance(begin, end));
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), Compare<ForwardIt, Comp>(begin, comp));
    std::vector<size_t> out_indices(std::min(k, indices.size()));
    for (size_t i=0; i<out_indices.size(); i++) {
        out_indices[i] = indices[indices.size()-(1+i)];
    }
    return out_indices;
}
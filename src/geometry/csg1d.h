//
// Created by James Noeckel on 12/9/20.
//

#pragma once
#include <vector>

/**
 * Intersect two sorted non-self-intersecting lists of intervals
 * @tparam T
 * @param intervals1
 * @param intervals2
 * @param intersection if false, take the union
 * @return
 */
template <typename T>
std::vector<std::pair<T, T>> csg1d(const std::vector<std::pair<T, T>> &intervals1, const std::vector<std::pair<T, T>> &intervals2, bool intersection=true);

template <typename T>
std::vector<std::pair<T, T>> csg1d(const std::vector<std::pair<T, T>> &intervals1, const std::vector<std::pair<T, T>> &intervals2, bool intersection) {
    //true: entering
    std::vector<std::pair<T, bool>> crossings;
    for (auto intervals : {&intervals1, &intervals2}) {
        for (const auto &interval : *intervals) {
            crossings.emplace_back(interval.first, true);
            crossings.emplace_back(interval.second, false);
        }
    }
    int minValue = intersection ? 2 : 1;
    std::sort(crossings.begin(), crossings.end(), [](const auto &a, const auto &b) {return a.first < b.first;});
    int stack = 0;
    std::vector<T> intersections;
    for (const auto &crossing : crossings) {
        if (crossing.second) {
            ++stack;
            if (stack == minValue) {
                intersections.push_back(crossing.first);
            }
        } else {
            if (stack == minValue) {
                intersections.push_back(crossing.first);
            }
            --stack;
        }
    }
    std::vector<std::pair<T, T>> intervals;
    for (size_t i=0; i<intersections.size(); i+=2) {
        intervals.emplace_back(intersections[i], intersections[i+1]);
    }
    return intervals;
}
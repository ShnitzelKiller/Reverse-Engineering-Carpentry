//
// Created by James Noeckel on 10/28/20.
//

#pragma once
#include <numeric>
#include <limits>

/**
 * Sets all elements in iterator range that are not peaks to 0, and writes new sequence to output iterator (can be the same as input)
 * If adjacent values are the same, the rightmost will be kept.
 */
template <class T, class ForwardIt, class OutputIterator>
int nonmax_suppression(ForwardIt first, ForwardIt last, OutputIterator out);


template <class T, class ForwardIt, class OutputIterator>
int nonmax_suppression(ForwardIt first, ForwardIt last, OutputIterator out) {
    T left = std::numeric_limits<T>::lowest();
    ForwardIt it_next = first;
    ++it_next;
    T val;
    int numPeaks = 0;
    for (ForwardIt it=first; it != last; ++it, ++it_next, left=val) {
        val = *it;
        if (it == first || it_next == last) {
            //suppress left endpoint
            *out++ = 0;
            continue;
        }
        if (left > val) {
            *out++ = 0;
            continue;
        }
        if (it_next != last) {
            T right = *it_next;
            if (right >= val) {
                *out++ = 0;
                continue;
            }
        }
        *out++ = val;
        ++numPeaks;
    }
    return numPeaks;
}
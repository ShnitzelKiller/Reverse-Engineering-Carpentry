//
// Created by James Noeckel on 9/10/20.
//

#pragma once
#include <vector>
#include <iostream>

template <class T>
std::ostream &operator<<(std::ostream &o, const std::vector<T> &vec) {
    o << '[';
    if (!vec.empty()) {
        for (size_t i=0; i<vec.size()-1; ++i) {
            o << vec[i] << ", ";
        }
        o << vec.back();
    }
    o << ']';
    return o;
}

/*template <class T>
void printvec(const std::vector<T> &vec) {
    std::cout << '[';
    for (size_t i=0; i<vec.size()-1; ++i) {
        std::cout << vec[i] << ", ";
    }
    std::cout << vec.back() << ']' << std::endl;
}*/

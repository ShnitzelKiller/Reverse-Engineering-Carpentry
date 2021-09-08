//
// Created by James Noeckel on 5/5/20.
//

#pragma once
#include <iostream>
#include <string>

bool assertPrint(bool condition, const std::string &message) {
    if (!condition) {
        std::cerr << message << std::endl;
    }
    return condition;
}

template <class T1, class T2>
bool isApprox(T1 a, T2 b, double eps=1e-6) {
    return std::fabs(a-b) <= eps;
}

template <class T1, class T2>
bool assertApproxEquals(T1 val1, T2 val2, const std::string &name, double eps=1e-6) {
    if (!isApprox(val1, val2, eps)) {
        std::cerr << name << " is " << val1 << ", should be " << val2 << std::endl;
        return false;
    }
    return true;
}

template <class T1, class T2>
bool assertEquals(T1 val1, T2 val2, const std::string &name) {
    if (val1 != val2) {
        std::cerr << name << " is " << val1 << ", should be " << val2 << std::endl;
        return false;
    }
    return true;
}

//
// Created by James Noeckel on 11/12/20.
//

#pragma once

template <typename T>
inline int getSign(T val) {
    if (val < 0) {
        return -1;
    } else if (val > 0) {
        return 1;
    } else {
        return 0;
    }
}
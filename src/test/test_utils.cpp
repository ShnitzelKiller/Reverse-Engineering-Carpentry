//
// Created by James Noeckel on 4/7/20.
//

#include <iostream>
#include "utils/top_k_indices.hpp"
#include "utils/sorted_data_structures.hpp"

int main(int argc, char **argv) {
    bool passed = true;
    {
        std::vector<int> sorted_vec;
        sorted_insert(sorted_vec, 0);
        sorted_insert(sorted_vec, 0);
        sorted_insert(sorted_vec, 0);
        sorted_insert(sorted_vec, 99);
        sorted_insert(sorted_vec, 99);
        sorted_insert(sorted_vec, 0);
        sorted_insert(sorted_vec, 3);
        passed = passed && sorted_vec.size() == 3 && sorted_vec[0] == 0 && sorted_vec[1] == 3 && sorted_vec[2] == 99
                && &(*sorted_find(sorted_vec, 99)) == &sorted_vec[2] && sorted_contains(sorted_vec, 3);
    }
    {
        std::vector<std::pair<int, int>> sorted_map;
        sorted_insert(sorted_map, 0, 9);
        sorted_insert(sorted_map, 4, 99);
        sorted_insert(sorted_map, 86, 999);
        sorted_insert(sorted_map, 4, 98);
        sorted_get(sorted_map, 86) = 998;
        for (const auto &pair : sorted_map) {
            std::cout << '(' << pair.first << ", " << pair.second << ") ";
        }
        std::cout << std::endl;
        passed = passed && sorted_map.size() == 3 && sorted_get(sorted_map, 4) == 99 &&
                sorted_get(sorted_map, 0) == 9 &&
                sorted_get(sorted_map, 86) == 998 &&
                sorted_find(sorted_map, 666) == sorted_map.end() &&
                sorted_find(sorted_map, 0)->second == 9;
    }
    {
        std::vector<int> values = {-1, 8, 4, 99};
        std::vector<size_t> indices = top_k_indices(values.begin(), values.end(), 3);
        passed = passed && indices.size() == 3 && indices[0] == 3 && indices[1] == 1 && indices[2] == 2;
        std::vector<size_t> indices2 = top_k_indices(values.begin(), values.end(), 2, std::greater<>());
        passed = passed && indices2.size() == 2 && indices2[0] == 0 && indices2[1] == 2;
    }
    return !passed;
}
//
// Created by James Noeckel on 2/13/20.
//

#include "utils/IntervalTree.h"
#include <iostream>

template <typename T, typename V>
void printIntervals(const IntervalTreeResult<T, V> &result) {
    for (auto it=result.begin(); it != result.end(); ++it) {
        std::cout << "([" << it->first.start << ", " << it->first.end << "], " << it->second << "), ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    bool success = true;
    {
        std::vector<std::pair<Interval<float>, int>> intervals;
        intervals.reserve(1000);
        for (int i = 0; i < 1000; i++) {
            intervals.emplace_back(Interval<float>(i - 1, i + 1), i);
        }

        //construction
        IntervalTree<float, int> tree(intervals.begin(), intervals.end());
        {
            auto result = tree.query(50.732);
            std::cout << "result after construction: ";
            printIntervals(result);
            success = success && result.size() == 2;
        }

        //query interval
        {
            auto result = tree.query(Interval<float>(49.99, 52));
            std::cout << "result of interval query: ";
            printIntervals(result);
            //success = success && result.size() == 2;
        }

        //insert
        {
            tree.insert(Interval<float>(50, 51), 0);
            tree.insert(Interval<float>(49, 52), 1);
            tree.insert(Interval<float>(10, 30), 2);
            auto result = tree.query(50.732);
            std::cout << "result after insertion: ";
            printIntervals(result);
            success = success && result.size() == 4;
        }

        //clear
        {
            tree.clear();
            auto result = tree.query(50.732);
            std::cout << "result size after clear: " << result.size() << std::endl;
            success = success && result.begin() == result.end();
        }

        //build
        {
            tree.build(intervals.begin(), intervals.end());
            auto result = tree.query(50.732);
            std::cout << "result after rebuild: ";
            printIntervals(result);
            success = success && result.size() == 2;
        }

        //many queries
        {
            std::cout << "some other queries: " << std::endl;
            for (int i = 5; i < 8; i++) {
                float point = i + 0.25f;
                auto result = tree.query(point);
                std::cout << "result at " << point << ": ";
                printIntervals(result);
                success = success && result.size() == 2;
            }
        }
    }
    //integer
    {
        std::cout << "integer: " << std::endl;
        std::vector<std::pair<Interval<int>, int>> intervals;
        intervals.emplace_back(Interval<int>(2, 4), 666);
        IntervalTree<int, int> tree(intervals.begin(), intervals.end());
        auto result1 = tree.query(Interval<int>(1, 2));
        std::cout << "result1: " << std::endl;
        printIntervals(result1);
        auto result2 = tree.query(Interval<int>(2, 3));
        std::cout << "result2: " << std::endl;
        printIntervals(result2);
        auto result3 = tree.query(Interval<int>(3, 5));
        std::cout << "result3: " << std::endl;
        printIntervals(result3);
        auto result4 = tree.query(Interval<int>(4, 5));
        std::cout << "result4: " << std::endl;
        printIntervals(result4);
    }


    return !success;
}
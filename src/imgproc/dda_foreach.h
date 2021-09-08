//
// Created by James Noeckel on 10/13/20.
//

#pragma once
#include <algorithm>
#include <cmath>
//#include <iostream>

/**
 * Run provided routine for each unique pixel location along an 8-connected line (in no particular order)
 * Must use floating point type as input bounds
 * @param lambda taking two integer arguments i,j
 * @param iStart
 * @param jStart
 * @param iEnd
 * @param jEnd
 */
template <typename F, typename T>
void dda_foreach(F &lambda, T iStart, T jStart, T iEnd, T jEnd) {
    T iDiff = iEnd - iStart;
    T jDiff = jEnd - jStart;
    bool swappedIJ = false;
    if (std::abs(iDiff) > std::abs(jDiff)) {
        swappedIJ = true;
        std::swap(iStart, jStart);
        std::swap(iEnd, jEnd);
        std::swap(iDiff, jDiff);
    }
    if (jStart > jEnd) {
        std::swap(iStart, iEnd);
        std::swap(jStart, jEnd);
        iDiff = -iDiff;
        jDiff = -jDiff;
    }
    //step along j
    //initial j offset
    T iIncrement = iDiff / jDiff;
    T i = iStart;
    int jInt = static_cast<int>(std::round(jStart));
    int jMax = static_cast<int>(std::round(jEnd));
    T jOffset = jInt - jStart;
    i += iIncrement * jOffset;
//    std::cout << "increment: " << iIncrement << std::endl;
//    std::cout << "jInt: " << jInt << std::endl;
//    std::cout << "jmax: " << jMax << std::endl;
    for (; jInt <= jMax; ++jInt) {
        int iInt = static_cast<int>(std::round(i));
        if (swappedIJ) {
            lambda(jInt, iInt);
        } else {
            lambda(iInt, jInt);
        }
        i += iIncrement;
    }
}

/**
 * Run provided routine for each unique pixel location along an 8-connected line (in no particular order)
 * Must use floating point type as input bounds
 * @param lambda taking two integer arguments i,j
 * @param iStart
 * @param jStart
 * @param iEnd
 * @param jEnd
 */
template <typename F, typename T>
void dda_foreach(F &lambda, T iStart, T jStart, T kStart, T iEnd, T jEnd, T kEnd) {
    T diffs[3] = {iEnd - iStart, jEnd - jStart, kEnd - kStart};
    T steps[3] = {iStart, jStart, kStart};
    T ends[3] = {iEnd, jEnd, kEnd};
    int maxDir = 0;
    T maxDiff = std::abs(diffs[0]);
    for (int i = 1; i < 3; ++i) {
        T absdiff = std::abs(diffs[i]);
        if (std::abs(absdiff) > maxDiff) {
            maxDir = i;
            maxDiff = absdiff;
        }
    }

    //step along j
    //initial j offset
    T increments[2] = {diffs[(maxDir+1) % 3]/maxDiff, diffs[(maxDir+2) % 3]/maxDiff};
    int stepInts[3];
    stepInts[maxDir] = static_cast<int>(std::round(steps[maxDir]));
    int stepEnd = static_cast<int>(std::round(ends[maxDir]));
    T offset = stepInts[maxDir] - steps[maxDir];
    for (int ind=0; ind<2; ++ind) {
        steps[(maxDir + ind + 1) % 3] += increments[ind] * offset;
    }
    int sgn = diffs[maxDir] > 0 ? 1 : -1;
    for (; stepInts[maxDir] != stepEnd; stepInts[maxDir]+=sgn) {
        for (int ind=0; ind<2; ++ind) {
            stepInts[(maxDir + ind + 1) % 3] = static_cast<int>(std::round(steps[(maxDir + ind + 1) % 3]));
        }
        lambda(stepInts[0], stepInts[1], stepInts[2]);
        for (int ind=0; ind<2; ++ind) {
            steps[(maxDir + ind + 1) % 3] += increments[ind];
        }
    }
    for (int ind=0; ind<2; ++ind) {
        stepInts[(maxDir + ind + 1) % 3] = static_cast<int>(std::round(steps[(maxDir + ind + 1) % 3]));
    }
    lambda(stepInts[0], stepInts[1], stepInts[2]);
}
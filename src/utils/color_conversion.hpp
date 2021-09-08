#pragma once

template <class FT>
FT rgb2grayscale(FT r, FT g, FT b) {
    return 0.11 * b + 0.59 * g + 0.3 * r;
}

template <class FT>
void hsv2rgb(FT h, FT s, FT v, FT &r, FT &g, FT &b) {
    FT c = v * s;
    FT x = c * (1 - std::abs(std::fmod(h/60, 2) - 1));
    FT m = v - c;
    FT rp, gp, bp;
    if (h >= 0 && h < 60) {
        rp = c;
        gp = x;
        bp = 0;
    } else if (h >= 60 && h < 120) {
        rp = x;
        gp = c;
        bp = 0;
    } else if (h >= 120 && h < 180) {
        rp = 0;
        gp = c;
        bp = x;
    } else if (h >= 180 && h < 240) {
        rp = 0;
        gp = x;
        bp = c;
    } else if (h >= 240 && h < 300) {
        rp = x;
        gp = 0;
        bp = c;
    } else {
        rp = c;
        gp = 0;
        bp = x;
    }
    r = rp + m;
    g = gp + m;
    b = bp + m;
}
#pragma once
// pencil 用 Transpose が使う、重なりの数え上げ。
//
// 格子側の配列（のりしろ込み）と FFT 側の pencil は、周期境界のために
// 1対の相手どうしでも複数の塊で重なることがある（例: 端の層が折り返して同じ相手に届く）。
// 送る側と受ける側が同じ関数で同じ順番に塊を数え上げることで、並びの食い違いをなくす。
#include <algorithm>
#include <vector>

namespace pencil_overlap {

// 局所添字 i0 から len 個が、大域座標 g0 から連続して並ぶ区間
struct Run {
    int i0;
    int g0;
    int len;
};

// 局所添字 0..count-1 が大域座標 (gstart + i) mod Ng に対応するとき、
// 折り返しの位置で切って、大域座標が連続する区間に分ける。
inline std::vector<Run> split_runs(int gstart, int count, int Ng) {
    std::vector<Run> runs;
    int g = ((gstart % Ng) + Ng) % Ng;
    int i = 0;
    while (count > 0) {
        const int len = std::min(count, Ng - g);
        runs.push_back({i, g, len});
        i += len;
        count -= len;
        g = 0;
    }
    return runs;
}

// runs のうち、大域座標 [b0, b0 + bl) に入る部分
inline std::vector<Run> clip(const std::vector<Run>& runs, int b0, int bl) {
    std::vector<Run> out;
    for (const Run& r : runs) {
        const int lo = std::max(r.g0, b0);
        const int hi = std::min(r.g0 + r.len, b0 + bl);
        if (lo < hi) out.push_back({r.i0 + (lo - r.g0), lo, hi - lo});
    }
    return out;
}

} // namespace pencil_overlap

#pragma once
#include <cstddef>

inline size_t align_to_64(size_t count) {
    size_t bytes = count * sizeof(double);
    return ((bytes + 63) / 64) * 64 / sizeof(double);
}

// MPI の count は int。実装内部で count * sizeof(double) を 32bit で
// 計算すると 2^31 バイト(2 GiB)を超えた時点であふれるため、
// 集団通信はこの単位に分割して呼ぶ。
// 1<<26 doubles = 67,108,864 doubles = 512 MiB。
inline constexpr size_t MPI_CHUNK_DOUBLES = 1ull << 26;

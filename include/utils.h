#pragma once
#include <cstddef>

inline size_t align_to_64(size_t count) {
    size_t bytes = count * sizeof(double);
    return ((bytes + 63) / 64) * 64 / sizeof(double);
}

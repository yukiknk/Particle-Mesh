#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cstring> 
#include <omp.h>

class ReorderSlabBack {
public:
    ReorderSlabBack(int Ng, int ny_loc, int nz_loc, int stride, const std::array<int,3>& dims, ptrdiff_t local_n0);

    void execute(const double* slab_in, double* cube_out) const;

private:
    struct Segment {
        size_t dst; 
        size_t src;
        size_t len;
    };

    std::vector<Segment> segments_;
};

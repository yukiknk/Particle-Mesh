#include "reorder_slab_back.h"

ReorderSlabBack::ReorderSlabBack(int Ng, int ny_loc, int nz_loc, int stride, const std::array<int,3>& dims, ptrdiff_t local_n0) {
    const int yz_ranks = dims[1] * dims[2];
    const int ny1 = ny_loc + 3;
    const int nz1 = nz_loc + 3;
    const size_t slab_cell = static_cast<std::size_t>(local_n0) * ny1 * nz1;

    std::vector<size_t> pos0(yz_ranks * slab_cell);
    size_t index = 0;
    for(int r = 0; r < yz_ranks; ++r){
        const int ry =  r / dims[2];
        const int rz =  r % dims[2];
        for(size_t i = 0; i < slab_cell; ++i){
            const int xi = i / (ny1 * nz1);
            const int yi = (ry * ny_loc + ((i / nz1) % ny1) -1 + Ng) % Ng;
            const int zi = (rz * nz_loc + i % nz1 -1 + Ng) % Ng;
            pos0[index] = (xi * Ng + yi) * stride + zi;
            ++index;
        }
    }

    segments_.reserve(pos0.size());
    size_t seg_begin = 0;
    for (size_t i = 1; i <= pos0.size(); ++i) {
        bool break_here = (i == pos0.size()) || (pos0[i] != pos0[i-1] + 1);
        if (break_here) {
            size_t dst = seg_begin;
            size_t src = pos0[seg_begin];
            size_t len = i - seg_begin;
            segments_.push_back({dst, src, len});
            seg_begin = i;
        }
    }
}

void ReorderSlabBack::execute(const double* in, double* out) const
{
    const Segment* __restrict seg = segments_.data();
    const size_t S = segments_.size();

    #pragma omp parallel for schedule(static)
    for (size_t s = 0; s < S; ++s) {
        const double* __restrict src = in  + seg[s].src;
        double*       __restrict dst = out + seg[s].dst;
        const size_t n = seg[s].len; 
        std::memcpy(dst, src, n * sizeof(double));
    }
}

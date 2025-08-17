#include "reorder_slab.h"

ReorderSlab::ReorderSlab(int Ng, int ny, int nz, ptrdiff_t local_n0, int stride, const std::vector<int>& recvcounts, const std::vector<int>& rdispls, const std::array<int,3>& dims) {
    const int size = static_cast<int>(recvcounts.size());
    const int ny_loc = ny-1, nz_loc = nz-1;

    size_t recv_total = static_cast<size_t>(rdispls.back()) + recvcounts.back();
    pos0_.resize(recv_total);
    pos1_.resize(recv_total);
    size_t index = 0;
    for (int r = 0; r < size; ++r){
        const int ry = (r / dims[2]) % dims[1];
        const int rz =  r % dims[2];

        for (int i = 0; i < recvcounts[r]; ++i){
            const int xi =  i / (ny * nz);
            const int yi = (ry * ny_loc + (i / nz) % ny) % Ng;
            const int zi = (rz * nz_loc +  i % nz) % Ng;

            pos0_[index] = (xi * Ng + yi) * stride + zi;
            ++index;
        }
    }

    std::vector<size_t> id(recv_total);
    std::iota(id.begin(), id.end(), 0);

    std::sort(id.begin(), id.end(), [&](size_t a, size_t b){ return pos0_[a] < pos0_[b]; });

    std::vector<size_t> pos0_sorted(recv_total);
    for (size_t i = 0; i < recv_total; ++i){
        pos0_sorted[i] = pos0_[id[i]];
        pos1_[i] = id[i];
    }
    pos0_.swap(pos0_sorted);

    seg_.reserve(recv_total + 1);
    seg_.push_back(0);
    for(size_t i = 1; i < recv_total; ++i) if(pos0_[i] != pos0_[i-1]) seg_.push_back(i);
    seg_.push_back(recv_total);
}

void ReorderSlab::execute(const double* __restrict in, double* __restrict out) const {
    const size_t S = seg_.size() - 1;
    const size_t* __restrict seg  = seg_.data();
    const size_t* __restrict idx  = pos1_.data();
    const size_t* __restrict key  = pos0_.data();
    #pragma omp parallel for schedule(guided)
    for (size_t s = 0; s < S; ++s){
        const size_t beg = seg[s];
        const size_t end = seg[s + 1];
        size_t k = beg;
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;

        const size_t n = end - beg;
        const size_t n4 = n & ~size_t(3);
        for (; k < beg + n4; k += 4) {
            a0 += in[idx[k]];
            a1 += in[idx[k + 1]];
            a2 += in[idx[k + 2]];
            a3 += in[idx[k + 3]];
        }
        double acc = (a0 + a1) + (a2 + a3);
        for (; k < end; ++k) acc += in[idx[k]];

        out[key[beg]] = acc;
    }
}

#pragma once
#include <vector>
#include <cstddef>
#include <array>
#include <numeric> 
#include <algorithm>  
#include <cstring>  
#include <omp.h>

class ReorderSlab {
public:
    ReorderSlab(int Ng, int ny, int nz, ptrdiff_t local_n0, int stride, const std::vector<int>& recvcounts, const std::vector<int>& rdispls, const std::array<int,3>& dims);

    void execute(const double* cube_in, double* slab_out) const;

private:
    std::vector<size_t> pos0_;
    std::vector<size_t> pos1_;
    std::vector<size_t> seg_;
};

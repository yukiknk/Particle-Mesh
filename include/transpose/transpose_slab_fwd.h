#pragma once
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "mpi_env.h"
#include "grid.h"
#include "fft.h"
#include "buffer_manager.h"
#include "utils.h"

class TransposeSlabFwd {
public:
    TransposeSlabFwd(const Grid& grid, const FFT& fft, const MPIEnv& mpi, BufferManager& buffer);
    
    TransposeSlabFwd(const TransposeSlabFwd&) = delete;
    TransposeSlabFwd& operator=(const TransposeSlabFwd&) = delete;
    TransposeSlabFwd(TransposeSlabFwd&&) = delete;
    TransposeSlabFwd& operator=(TransposeSlabFwd&&) = delete;

    void execute();

private:
    void alltoallv();
    void reorder();

    int world_size_;
    int color_;
    int Ng_;
    int stride_;
    
    int nx1_, ny1_, nz1_;
    
    std::vector<int> sendcounts_;
    std::vector<int> sdispls_;
    std::vector<int> recvcounts_;
    std::vector<int> rdispls_;
    
    std::vector<size_t> pos0_;
    std::vector<size_t> pos1_;
    std::vector<size_t> seg_;
    
    double* sendbuf_ = nullptr;
    double* recvbuf_ = nullptr;
    double* fftbuf_ = nullptr;
};
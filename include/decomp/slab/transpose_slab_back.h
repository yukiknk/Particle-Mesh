#pragma once
#include <mpi.h>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>

class TransposeSlabBack {
public:
    TransposeSlabBack(int Ng, int color, int nx_loc, int ny1, int nz1, std::array<int,3> dims, ptrdiff_t local_0_start, ptrdiff_t local_n0, MPI_Comm comm = MPI_COMM_WORLD);

    void execute(const double* sendbuf, double* recvbuf) const;

private:
    std::vector<int> sendcounts_, sdispls_, recvcounts_, rdispls_;
    MPI_Comm comm_;
};

#pragma once
#include <mpi.h>
#include "grid.h"
#include "mpi_env.h"

struct GridAuto : public Grid {
    GridAuto(int Ng_, const MPIEnv& mpi) {
        Ng = Ng_;

        dims[0] = dims[1] = dims[2] = 0;
        MPI_Dims_create(mpi.world_size(), 3, dims);

        int rank = mpi.world_rank();
        coords[0] = rank / (dims[1] * dims[2]);
        coords[1] = (rank / dims[2]) % dims[1];
        coords[2] = rank % dims[2];

        split(Ng, coords[0], dims[0], x0, nx);
        split(Ng, coords[1], dims[1], y0, ny);
        split(Ng, coords[2], dims[2], z0, nz);

        n_local = nx * ny * nz;
    }

    private:
    // [0, N) を d 個に分割したときの c 番目の区間 [start, start+len)
    static void split(int N, int c, int d, int& start, int& len) {
        const long long n = N;
        start = static_cast<int>(n * c / d);
        int next = static_cast<int>(n * (c + 1) / d);
        len = next - start;
    }
};

#include "transpose_slab.h"

TransposeSlab::TransposeSlab(int Ng, int nx, int ny, int nz, int x0, int dimsx, const std::vector<ptrdiff_t>& all_i0, const std::vector<ptrdiff_t>& all_n0, MPI_Comm comm) : comm_{comm} {
    int size; MPI_Comm_size(comm_, &size);
    sendcounts_.assign(size, 0);
    sdispls_.assign(size, 0);
    recvcounts_.assign(size, 0);
    rdispls_.assign(size, 0);

    for (int r = 0; r < size; ++r) {
        if (r >= Ng) break;
        const int s0 = static_cast<int>(all_i0[r]);
        const int s1 = s0 + static_cast<int>(all_n0[r]);
        const int l0 = std::max(s0, x0);
        const int l1 = std::min(s1, x0 + nx);
        if (l0 < l1) {
            int planes = l1 - l0;
            sendcounts_[r] = planes * ny * nz;
            sdispls_[r] = (l0 - x0) * ny * nz;
        }
    }

    int coords0 = x0 / (nx-1);
    if (coords0 == dimsx - 1) {
        sendcounts_[0] = ny * nz;
        sdispls_[0]    = (nx-1) * ny * nz;
    }

    MPI_Alltoall(sendcounts_.data(), 1, MPI_INT, recvcounts_.data(), 1, MPI_INT, comm_);

    rdispls_[0] = 0;
    for (int r = 1; r < size; ++r) rdispls_[r] = rdispls_[r-1] + recvcounts_[r-1];

    recv_total_ = static_cast<size_t>(rdispls_.back()) + recvcounts_.back();
}

void TransposeSlab::execute(const double* sendbuf, double* recvbuf) const {
    MPI_Alltoallv(sendbuf, sendcounts_.data(), sdispls_.data(), MPI_DOUBLE, recvbuf, recvcounts_.data(), rdispls_.data(), MPI_DOUBLE, comm_);
}

#include "transpose_slab_back.h"

TransposeSlabBack::TransposeSlabBack(int Ng, int color, int nx_loc, int ny1, int nz1, std::array<int,3> dims, ptrdiff_t local_0_start, ptrdiff_t local_n0, MPI_Comm comm) : comm_{comm} {
    int size; MPI_Comm_size(comm_, &size);
    int rank; MPI_Comm_rank(comm_, &rank);
    sendcounts_.assign(size, 0);
    sdispls_.assign(size, 0);
    recvcounts_.assign(size, 0);
    rdispls_.assign(size, 0);
    const int yz_ranks = dims[1] * dims[2];
    
    if(color == 1){
        for(int r = 0; r < size; ++r){
            const int s0 = nx_loc * (r / (yz_ranks)) - 1, s1 = s0 + nx_loc + 3;
            const int l0 = std::max(s0, static_cast<int>(local_0_start)), l1 = std::min(s1, static_cast<int>(local_0_start + local_n0));
            if(l0 < l1){
                int planes=l1 - l0;
                sendcounts_[r] = planes * ny1 * nz1;
                sdispls_[r] = (l0 - local_0_start) * ny1 * nz1 + local_n0 * ny1 *nz1 * (r % yz_ranks);
            } 
        }
        if(rank == std::min(Ng-1, size-1)) for(int r = 0; r < yz_ranks; ++r){
            sendcounts_[r] = ny1 * nz1;
            sdispls_[r] = ((r + 1) * local_n0 - 1) * ny1 * nz1;
        }else if(rank == 0 || (rank == 1 && local_n0 == 1)) for(int r = size - yz_ranks; r < size; ++r){
            sendcounts_[r] = (local_n0 == 1) ? ny1 * nz1 : 2 * ny1 * nz1;
            sdispls_[r] = (r % yz_ranks) * local_n0 * ny1 * nz1;
        }
    }
    
    MPI_Alltoall(sendcounts_.data(), 1, MPI_INT, recvcounts_.data(), 1, MPI_INT, comm_);
    if(rank / yz_ranks == 0){
        rdispls_[0] = recvcounts_[std::min(Ng-1, size-1)];
        for(int r = 1; r < std::min(Ng-1, size-1); ++r) rdispls_[r] = rdispls_[r-1] + recvcounts_[r-1];
    }else if(rank >= size - yz_ranks){
        int s = (local_n0 == 1) ? 2 : 1;
        rdispls_[s] = 0;
        for(int r = s + 1; r <= std::min(Ng-1, size-1); ++r) rdispls_[r] = rdispls_[r-1] + recvcounts_[r-1];
        rdispls_[0] = rdispls_[std::min(Ng-1, size-1)] + recvcounts_[std::min(Ng-1, size-1)];
        if(local_n0 == 1) rdispls_[1] = rdispls_[0] + recvcounts_[0];
    }else{
        rdispls_[0] = 0;
        for(int r = 1; r < size; ++r) rdispls_[r] = rdispls_[r-1] + recvcounts_[r-1];
    }
}

void TransposeSlabBack::execute(const double* sendbuf, double* recvbuf) const {
    MPI_Alltoallv(sendbuf, sendcounts_.data(), sdispls_.data(), MPI_DOUBLE, recvbuf, recvcounts_.data(), rdispls_.data(), MPI_DOUBLE, comm_);
}

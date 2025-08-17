#pragma once
#include "transpose_plan.h"

class TransposeSlab : public TransposePlan {
public:
    TransposeSlab(int Ng, int nx_loc, int ny, int nz, int x0, int dimsx, const std::vector<ptrdiff_t>& all_i0, const std::vector<ptrdiff_t>& all_n0, MPI_Comm comm);

    void execute(const double* sendbuf, double* recvbuf) const override;

private:
    MPI_Comm comm_;
};

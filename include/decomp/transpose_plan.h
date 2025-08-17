#pragma once
#include <mpi.h>
#include <vector>
#include <cstddef>

class TransposePlan {
public:
    virtual ~TransposePlan() = default;

    virtual void execute(const double* sendbuf, double* recvbuf) const = 0;

    size_t recv_size() const { return recv_total_; }
    const std::vector<int>& recvcounts() const { return recvcounts_; }
    const std::vector<int>& rdispls()    const { return rdispls_;   }

protected:
    std::vector<int> sendcounts_, sdispls_, recvcounts_, rdispls_;
    size_t recv_total_ {0};
};

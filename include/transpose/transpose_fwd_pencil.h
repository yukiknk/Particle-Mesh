#pragma once
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "mpi_env.h"
#include "grid/grid.h"
#include "fft/fft.h"
#include "buffer_manager.h"
#include "timer.h"
#include "utils.h"
#include "transpose_fwd.h"

class TransposeFwdPencil : public TransposeFwd {
public:
    TransposeFwdPencil(const Grid& grid, const FFT& fft, const MPIEnv& mpi,
                       BufferManager& buffer, Timer& timer, int method);
    ~TransposeFwdPencil() override;
    
    TransposeFwdPencil(const TransposeFwdPencil&) = delete;
    TransposeFwdPencil& operator=(const TransposeFwdPencil&) = delete;
    TransposeFwdPencil(TransposeFwdPencil&&) = delete;
    TransposeFwdPencil& operator=(TransposeFwdPencil&&) = delete;

    __attribute__((aligned(256))) void execute() override;

private:
    __attribute__((aligned(256))) void reorder_for_alltoallv();
    __attribute__((aligned(256))) void alltoallv();
    __attribute__((aligned(256))) void reorder_for_fft();
    __attribute__((aligned(256))) void reduce();

    Timer& timer_;
    int t_comm_;
    int t_calc_;
    
    int world_size_;
    int world_rank_;
    int Ng_;
    int stride_;

    int group_size_;
    int num_groups_;
    int group_id_;  
    int local_rank_;
    MPI_Comm group_comm_ = MPI_COMM_NULL;
    MPI_Comm reduce_comm_ = MPI_COMM_NULL;
    
    int nx1_, ny1_, nz1_;
    size_t slice_;   // ny1_ * nz1_ (yz平面サイズ)
    
    // alltoallv 関連
    std::vector<int> sendcounts_;
    std::vector<int> sdispls_;
    std::vector<int> recvcounts_;
    std::vector<int> rdispls_;
    
    // 送信側 reorder: sendbuf へのコピー指示
    std::vector<size_t> send_src_;
    std::vector<size_t> send_dst_;
    std::vector<int>    send_nx_;
    std::vector<int>    send_ny_;
    
    // 受信側 reorder: recvbuf → fftbuf 並べ替え
    std::vector<size_t> pos0_;
    std::vector<size_t> pos1_;
    std::vector<size_t> seg_;
    
    size_t fft_buf_size_;
    
    double* rhobuf_ = nullptr;
    double* sendbuf_ = nullptr;
    double* recvbuf_ = nullptr;
    double* fftbuf_ = nullptr;
};

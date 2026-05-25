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
#include "transpose_bwd.h"          // ← 追加(基底クラス)

// TransposeFwdの逆操作:
//   FFTスラブ分解（x方向のみ分割）→ グリッド3D分解
//   ポテンシャル場 φ(x) を各粒子プロセスが担当するグリッド領域に戻す
//   出力グリッドはゴースト幅 -1/+2 で (nx+3)×(ny+3)×(nz+3)

class TransposeBwdSlab : public TransposeBwd {   // ← : public TransposeBwd を追加
public:
    TransposeBwdSlab(const Grid& grid, const FFT& fft, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method);
    ~TransposeBwdSlab() override;                // ← override を追加

    TransposeBwdSlab(const TransposeBwdSlab&) = delete;
    TransposeBwdSlab& operator=(const TransposeBwdSlab&) = delete;
    TransposeBwdSlab(TransposeBwdSlab&&) = delete;
    TransposeBwdSlab& operator=(TransposeBwdSlab&&) = delete;

    __attribute__((aligned(256))) void execute() override;   // ← override を追加

private:
    __attribute__((aligned(256))) void broadcast();
    __attribute__((aligned(256))) void reorder();
    __attribute__((aligned(256))) void alltoallv();

    Timer& timer_;
    int t_comm_;
    int t_calc_;

    int world_size_;
    int world_rank_;
    int Ng_;
    int stride_;

    int nx3_, ny3_, nz3_;

    int group_size_;
    int num_groups_;
    int group_id_;
    int local_rank_;
    MPI_Comm group_comm_ = MPI_COMM_NULL;
    MPI_Comm bcast_comm_ = MPI_COMM_NULL;

    std::vector<int> sendcounts_;
    std::vector<int> sdispls_;
    std::vector<int> recvcounts_;
    std::vector<int> rdispls_;

    std::vector<size_t> pos0_;

    size_t fft_buf_size_;
    size_t send_total_;

    double* sendbuf_ = nullptr;
    double* recvbuf_ = nullptr;
    double* fftbuf_  = nullptr;
};

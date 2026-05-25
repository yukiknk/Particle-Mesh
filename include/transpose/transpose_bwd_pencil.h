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
#include "transpose_bwd.h"

class TransposeBwdPencil : public TransposeBwd {
public:
    TransposeBwdPencil(const Grid& grid, const FFT& fft, const MPIEnv& mpi,
                       BufferManager& buffer, Timer& timer, int method);
    ~TransposeBwdPencil() override;

    TransposeBwdPencil(const TransposeBwdPencil&) = delete;
    TransposeBwdPencil& operator=(const TransposeBwdPencil&) = delete;
    TransposeBwdPencil(TransposeBwdPencil&&) = delete;
    TransposeBwdPencil& operator=(TransposeBwdPencil&&) = delete;

    void execute() override;

private:
    __attribute__((aligned(256))) void reorder_from_fft();
    __attribute__((aligned(256))) void alltoallv();
    __attribute__((aligned(256))) void reorder_from_alltoallv();

    Timer& timer_;
    int t_comm_;
    int t_calc_;

    int world_size_;
    int world_rank_;
    int Ng_;
    int stride_;

    int nx3_, ny3_, nz3_;   // 拡張グリッド(両側ゴースト、中心差分用)

    std::vector<int> sendcounts_;
    std::vector<int> sdispls_;
    std::vector<int> recvcounts_;
    std::vector<int> rdispls_;

    // 送信側 reorder: fftbuf → sendbuf
    std::vector<size_t> pos0_;          // 構築用(中身は fftbuf 位置)
    std::vector<size_t> fwd_copy_dst_;  // sendbuf 側オフセット(連番側)
    std::vector<size_t> fwd_copy_src_;  // fftbuf 側オフセット
    std::vector<size_t> fwd_copy_len_;  // コピー長(要素数)

    // 受信側 reorder: recvbuf → gridbuf
    std::vector<size_t> pos1_;          // 構築用(中身は gridbuf 位置)
    std::vector<size_t> bwd_copy_dst_;  // gridbuf 側オフセット
    std::vector<size_t> bwd_copy_src_;  // recvbuf 側オフセット(連番側)
    std::vector<size_t> bwd_copy_len_;  // コピー長(要素数)

    size_t fft_buf_size_;
    size_t send_total_;
    size_t recv_total_;

    double* fftbuf_  = nullptr;
    double* sendbuf_ = nullptr;
    double* recvbuf_ = nullptr;
    double* gridbuf_ = nullptr;
};

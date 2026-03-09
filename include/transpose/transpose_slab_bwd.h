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

// TransposeFwdの逆操作:
//   FFTスラブ分解（x方向のみ分割）→ グリッド3D分解
//   ポテンシャル場 φ(x) を各粒子プロセスが担当するグリッド領域に戻す
//   出力グリッドはゴースト幅 -1/+2 で (nx+3)×(ny+3)×(nz+3)

class TransposeSlabBwd {
public:
    TransposeSlabBwd(const Grid& grid, const FFT& fft, const MPIEnv& mpi, BufferManager& buffer, Timer& timer);
    ~TransposeSlabBwd();

    TransposeSlabBwd(const TransposeSlabBwd&) = delete;
    TransposeSlabBwd& operator=(const TransposeSlabBwd&) = delete;
    TransposeSlabBwd(TransposeSlabBwd&&) = delete;
    TransposeSlabBwd& operator=(TransposeSlabBwd&&) = delete;

    void execute();

private:
    void broadcast();   // num_groups > 1 の場合: group_id==0 から他グループへ fftbuf_ を配布
    void reorder();     // fftbuf_ から sendbuf_ へデータを並べ替え（fwd の reorder の逆方向）
    void alltoallv();   // グループ内 Alltoallv: sendbuf_ → recvbuf_ へ転送

    Timer& timer_;
    int t_comm_;
    int t_calc_;

    int world_size_;
    int world_rank_;
    int Ng_;
    int stride_;

    // 出力グリッドサイズ（ゴースト幅 -1/+2）
    int nx3_, ny3_, nz3_;

    // グループ分け（fwd と同じ構造）
    int group_size_;
    int num_groups_;
    int group_id_;
    int local_rank_;
    MPI_Comm group_comm_ = MPI_COMM_NULL;
    MPI_Comm bcast_comm_ = MPI_COMM_NULL;  // fwd の reduce_comm_ に対応

    // alltoallv パラメータ
    //   sendcounts_[r] = rank r へ送信する量
    //   recvcounts_[r] = rank r から受信する量（自分の拡張グリッド用）
    std::vector<int> sendcounts_;
    std::vector<int> sdispls_;
    std::vector<int> recvcounts_;
    std::vector<int> rdispls_;

    // reorder 用インデックス（fwd の pos0_ に対応）
    // pos0_[i]: fftbuf_ 内のソース位置
    std::vector<size_t> pos0_;

    size_t fft_buf_size_;
    size_t send_total_;

    double* sendbuf_ = nullptr;   // alltoallv 送信バッファ（reorder の出力先）
    double* recvbuf_ = nullptr;   // alltoallv 受信バッファ（最終出力先、拡張グリッド）
    double* fftbuf_  = nullptr;   // FFTスラブ配置のバッファ（入力）
};

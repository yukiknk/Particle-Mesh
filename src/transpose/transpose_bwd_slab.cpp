#include "transpose/transpose_bwd_slab.h"
#include <omp.h>
#include <cstring>
#include "debug.h"
#include "grouping.h"

TransposeBwdSlab::TransposeBwdSlab(const Grid& grid, const FFT& fft, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method)
    : timer_(timer),
      world_size_(mpi.world_size()),
      world_rank_(mpi.world_rank()),
      Ng_(grid.Ng),
      stride_(fft.stride()),
      nx3_(grid.nx + 3),
      ny3_(grid.ny + 3),
      nz3_(grid.nz + 3)
{
    // タイマー登録
    t_comm_ = timer_.register_timer("TransposeBwd Comm");
    t_calc_ = timer_.register_timer("TransposeBwd Calc");

    Grouping grouping(fft.grouping_ng(), mpi, method);
    group_size_ = grouping.group_size;
    num_groups_ = grouping.num_groups;
    group_id_ = grouping.group_id;
    local_rank_ = grouping.local_rank;

    // グループ内コミュニケータ
    MPI_Comm_split(MPI_COMM_WORLD, group_id_, local_rank_, &group_comm_);

    // Broadcast用コミュニケータ（num_groups > 1 の場合）
    if (num_groups_ > 1) {
        MPI_Comm_split(MPI_COMM_WORLD, local_rank_, group_id_, &bcast_comm_);
    }

    int local_n0 = static_cast<int>(fft.local_n0());
    int local_0_start = fft.local_0_start();

    // グループ内の各プロセスのGrid情報を収集
    std::vector<int> group_vx0(group_size_);
    std::vector<int> group_vy0(group_size_);
    std::vector<int> group_vz0(group_size_);
    std::vector<int> group_vnx(group_size_);
    std::vector<int> group_vny(group_size_);
    std::vector<int> group_vnz(group_size_);

    MPI_Allgather(&grid.x0, 1, MPI_INT, group_vx0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.y0, 1, MPI_INT, group_vy0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.z0, 1, MPI_INT, group_vz0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.nx, 1, MPI_INT, group_vnx.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.ny, 1, MPI_INT, group_vny.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.nz, 1, MPI_INT, group_vnz.data(), 1, MPI_INT, group_comm_);

    // ---- 送受信量の計算（グループ内）----
    sendcounts_.assign(group_size_, 0);
    sdispls_.assign(group_size_, 0);
    recvcounts_.assign(group_size_, 0);
    rdispls_.assign(group_size_, 0);

    // sendcounts_[r]: rank r へ送信する量
    // 自分のスラブ [local_0_start, local_0_start+local_n0) と
    // rank r の拡張グリッド [x0_r-1, x0_r+nx_r+2) の交差
    for (int r = 0; r < group_size_; ++r) {
        int r_ny3 = group_vny[r] + 3;
        int r_nz3 = group_vnz[r] + 3;

        // 非折り返し部分の交差
        int l0 = std::max(group_vx0[r] - 1, local_0_start);
        int l1 = std::min(group_vx0[r] + group_vnx[r] + 2, local_0_start + local_n0);
        int count = std::max(0, l1 - l0);

        // 負方向の折り返し: x0_r - 1 < 0 → x = Ng-1 が必要
        if (group_vx0[r] == 0) {
            if (local_0_start + local_n0 == Ng_) {
                count += 1;
            }
        }

        // 正方向の折り返し: x0_r + nx_r + 2 > Ng → x = 0, 1, ... が必要
        int wrap_end = group_vx0[r] + group_vnx[r] + 2 - Ng_;
        if (wrap_end > 0) {
            int wl0 = std::max(0, local_0_start);
            int wl1 = std::min(wrap_end, local_0_start + local_n0);
            count += std::max(0, wl1 - wl0);
        }

        if (count > 0) {
            sendcounts_[r] = count * r_ny3 * r_nz3;
        }
    }

    for (int r = 1; r < group_size_; ++r) {
        sdispls_[r] = sdispls_[r - 1] + sendcounts_[r - 1];
    }

    // recvcounts_[r]: rank r から受信する量
    MPI_Alltoall(sendcounts_.data(), 1, MPI_INT, recvcounts_.data(), 1, MPI_INT, group_comm_);

    // rdispls_[r]: 拡張グリッド内のオフセット（fwd の sdispls_ と同様）
    int nyz3 = ny3_ * nz3_;

    for (int r = 0; r < group_size_; ++r) {
        if (recvcounts_[r] == 0) continue;

        // rank r のスラブ [r*local_n0, (r+1)*local_n0) と自分の拡張グリッドの交差
        int slab_s = r * local_n0;
        int slab_e = slab_s + local_n0;

        // 拡張グリッドの各 x_local に対応する x_abs がスラブに入る最初の位置
        // 非折り返し部分
        int l0 = std::max(grid.x0 - 1, slab_s);
        int l1 = std::min(grid.x0 + grid.nx + 2, slab_e);
        if (l0 < l1) {
            rdispls_[r] = (l0 - (grid.x0 - 1)) * nyz3;
            continue;
        }

        // 負方向折り返し: x_abs = Ng-1 がスラブに含まれる（自分が x0=0 の場合）
        if (grid.x0 == 0 && slab_e == Ng_) {
            rdispls_[r] = 0;  // x_local = 0 (拡張グリッドの先頭)
            continue;
        }

        // 正方向折り返し: x_abs = 0, 1, ... がスラブに含まれる
        int wrap_end = grid.x0 + grid.nx + 2 - Ng_;
        if (wrap_end > 0) {
            int wl0 = std::max(0, slab_s);
            // 折り返し開始位置は拡張グリッド内で (Ng - (x0-1)) 番目
            rdispls_[r] = (Ng_ - (grid.x0 - 1) + wl0) * nyz3;
        }
    }

    send_total_ = static_cast<size_t>(sdispls_.back()) + sendcounts_.back();

    // FFTバッファサイズ
    fft_buf_size_ = align_to_64(static_cast<size_t>(fft.local_alloc()) * 2);

    // バッファ登録
    // fftbuf_ と recvbuf_ は時間的に排他（reorder で fftbuf_ を読み終えた後に alltoallv で recvbuf_ に書く）
    // sendbuf_ は reorder で fftbuf_ を読みながら書くので、fftbuf_ と別領域が必要
    size_t recvbuf_size = align_to_64(static_cast<size_t>(nx3_) * nyz3);
    size_t sendbuf_offset = std::max(recvbuf_size, fft_buf_size_);
    size_t total_size = sendbuf_offset + align_to_64(send_total_);

    buffer.register_buffer(fftbuf_,  0);
    buffer.register_buffer(recvbuf_, 0);
    buffer.register_buffer(sendbuf_, sendbuf_offset);
    buffer.update_max_size(total_size);

    // ---- reorder 用インデックス構築 ----
    // pos0_[i]: sendbuf_ の i 番目の要素に対応する fftbuf_ 内のソース位置
    pos0_.resize(send_total_);

    size_t index = 0;
    for (int r = 0; r < group_size_; ++r) {
        int r_x_start = (group_vx0[r] - 1 + Ng_) % Ng_;
        int r_ny3 = group_vny[r] + 3;
        int r_nz3 = group_vnz[r] + 3;
        int r_y_start = (group_vy0[r] - 1 + Ng_) % Ng_;
        int r_z_start = (group_vz0[r] - 1 + Ng_) % Ng_;
        int r_nx3 = group_vnx[r] + 3;

        for (int ix = 0; ix < r_nx3; ++ix) {
            int xa = (r_x_start + ix) % Ng_;
            if (xa < local_0_start || xa >= local_0_start + local_n0) continue;
            int xi = xa - local_0_start;

            for (int j = 0; j < r_ny3; ++j) {
                int ya = (r_y_start + j) % Ng_;
                for (int k = 0; k < r_nz3; ++k) {
                    int za = (r_z_start + k) % Ng_;
                    pos0_[index] = (static_cast<size_t>(xi) * Ng_ + ya) * stride_ + za;
                    ++index;
                }
            }
        }
    }

    if (world_rank_ == 0) {
        DEBUG_LOG("Set TransposeBwd");
    }
}

TransposeBwdSlab::~TransposeBwdSlab() {
    if (group_comm_ != MPI_COMM_NULL) MPI_Comm_free(&group_comm_);
    if (bcast_comm_ != MPI_COMM_NULL) MPI_Comm_free(&bcast_comm_);
}

void TransposeBwdSlab::execute() {
    if (num_groups_ > 1) {
        broadcast();
        if (world_rank_ == 0) {
            DEBUG_LOG("Broadcast");
        }
    }
    reorder();
    if (world_rank_ == 0) {
        DEBUG_LOG("Reorder");
    }
    alltoallv();
    if (world_rank_ == 0) {
        DEBUG_LOG("Alltoallv");
    }
}

void TransposeBwdSlab::broadcast() {
    timer_.start(MPI_COMM_WORLD);
    MPI_Bcast(fftbuf_, static_cast<int>(fft_buf_size_), MPI_DOUBLE, 0, bcast_comm_);
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

void TransposeBwdSlab::reorder() {
    // fwd の reorder() の逆方向: fftbuf_ → sendbuf_ へデータを並べ替え
    timer_.start(MPI_COMM_WORLD);

    const size_t N = send_total_;
    const size_t* __restrict src_idx = pos0_.data();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        sendbuf_[i] = fftbuf_[src_idx[i]];
    }

    timer_.stop(t_calc_, MPI_COMM_WORLD);
}

void TransposeBwdSlab::alltoallv() {
    timer_.start(MPI_COMM_WORLD);
    MPI_Alltoallv(
        sendbuf_, sendcounts_.data(), sdispls_.data(), MPI_DOUBLE,
        recvbuf_, recvcounts_.data(), rdispls_.data(), MPI_DOUBLE,
        group_comm_
    );
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

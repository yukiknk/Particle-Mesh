#include "transpose/transpose_slab_bwd.h"
#include <omp.h>
#include <cstring>
#include "debug.h"

TransposeSlabBwd::TransposeSlabBwd(const Grid& grid, const FFT& fft, const MPIEnv& mpi, BufferManager& buffer, Timer& timer)
    : timer_(timer),
      world_size_(mpi.world_size()),
      world_rank_(mpi.world_rank()),
      Ng_(grid.Ng),
      stride_(fft.stride()),
      nx1_(grid.nx + 1),
      ny1_(grid.ny + 1),
      nz1_(grid.nz + 1)
{
    // タイマー登録
    t_comm_ = timer_.register_timer("TransposeBwd Comm");
    t_calc_ = timer_.register_timer("TransposeBwd Calc");

    // グループ分け（fwd と同じ）
    group_size_ = (world_size_ <= Ng_) ? world_size_ : Ng_;
    num_groups_ = world_size_ / group_size_;
    group_id_   = world_rank_ / group_size_;
    local_rank_ = world_rank_ % group_size_;

    // グループ内コミュニケータ
    MPI_Comm_split(MPI_COMM_WORLD, group_id_, local_rank_, &group_comm_);

    // Broadcast用コミュニケータ（num_groups > 1 の場合）
    if (num_groups_ > 1) {
        MPI_Comm_split(MPI_COMM_WORLD, local_rank_, group_id_, &bcast_comm_);
    }

    // bwd では fwd の send/recv が逆転する
    // fwd: grid → fft スラブ （alltoallv で各プロセスが持つ grid片 → fft スラブ担当プロセス）
    // bwd: fft スラブ → grid  （alltoallv で fft スラブ担当プロセス → grid片担当プロセス）
    //
    // fwd の recvcounts → bwd の sendcounts
    // fwd の sendcounts → bwd の recvcounts

    sendcounts_.assign(group_size_, 0);
    sdispls_.assign(group_size_, 0);
    recvcounts_.assign(group_size_, 0);
    rdispls_.assign(group_size_, 0);

    int x0 = grid.x0;
    int local_n0 = static_cast<int>(fft.local_n0());

    // fwd の sendcounts 計算を再現 → それが bwd の recvcounts
    std::vector<int> fwd_sendcounts(group_size_, 0);
    std::vector<int> fwd_sdispls(group_size_, 0);

    for (int r = 0; r < group_size_; ++r) {
        int s0 = r * local_n0;
        int s1 = s0 + local_n0;
        int l0 = std::max(s0, x0);
        int l1 = std::min(s1, x0 + nx1_);
        if (l0 < l1) {
            int planes = l1 - l0;
            fwd_sendcounts[r] = planes * ny1_ * nz1_;
            fwd_sdispls[r]    = (l0 - x0) * ny1_ * nz1_;
        }
    }
    if (grid.coords[0] == grid.dims[0] - 1) {
        fwd_sendcounts[0] = ny1_ * nz1_;
        fwd_sdispls[0]    = grid.nx * ny1_ * nz1_;
    }

    // bwd: recv側 = fwd の send 側
    MPI_Alltoall(fwd_sendcounts.data(), 1, MPI_INT, recvcounts_.data(), 1, MPI_INT, group_comm_);

    // bwd: send側 = fwd の recv 側（Alltoall で取得）
    MPI_Alltoall(recvcounts_.data(), 1, MPI_INT, sendcounts_.data(), 1, MPI_INT, group_comm_);

    for (int r = 1; r < group_size_; ++r) {
        sdispls_[r] = sdispls_[r - 1] + sendcounts_[r - 1];
        rdispls_[r] = rdispls_[r - 1] + recvcounts_[r - 1];
    }

    send_total_ = static_cast<size_t>(sdispls_.back()) + sendcounts_.back();
    size_t recv_total = static_cast<size_t>(rdispls_.back()) + recvcounts_.back();

    // FFTバッファサイズ
    fft_buf_size_ = align_to_64(static_cast<size_t>(fft.local_alloc()) * 2);

    // バッファ登録
    // sendbuf_: グリッド領域サイズ（fwd の Interpolater バッファと共用予定）
    size_t sendbuf_size  = align_to_64(static_cast<size_t>(nx1_) * ny1_ * nz1_);
    size_t recvbuf_offset = std::max(sendbuf_size, fft_buf_size_);
    size_t total_size     = recvbuf_offset + align_to_64(recv_total);

    buffer.register_buffer(fftbuf_,  0);
    buffer.register_buffer(sendbuf_, 0);
    buffer.register_buffer(recvbuf_, recvbuf_offset);
    buffer.update_max_size(total_size);

    // scatter 用インデックス構築（fwd の reorder インデックスと同一構造）
    // pos0_[i]: fftbuf_ 内のインデックス（ソートキー）
    // pos1_[i]: recvbuf_ 内のインデックス
    pos0_.resize(send_total_);
    pos1_.resize(send_total_);

    int local_0_start = fft.local_0_start();

    std::vector<int> group_vx0(group_size_);
    std::vector<int> group_vy0(group_size_);
    std::vector<int> group_vz0(group_size_);
    std::vector<int> group_vny(group_size_);
    std::vector<int> group_vnz(group_size_);

    MPI_Allgather(&grid.x0, 1, MPI_INT, group_vx0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.y0, 1, MPI_INT, group_vy0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.z0, 1, MPI_INT, group_vz0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.ny, 1, MPI_INT, group_vny.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.nz, 1, MPI_INT, group_vnz.data(), 1, MPI_INT, group_comm_);

    // TODO: scatter インデックスを構築する
    // fwd の reorder と同じ座標計算で pos0_, pos1_, seg_ を埋める

    if (world_rank_ == 0) {
        DEBUG_LOG("Set TransposeBwd");
    }
}

TransposeSlabBwd::~TransposeSlabBwd() {
    if (group_comm_ != MPI_COMM_NULL) MPI_Comm_free(&group_comm_);
    if (bcast_comm_ != MPI_COMM_NULL) MPI_Comm_free(&bcast_comm_);
}

void TransposeSlabBwd::execute() {
    if (num_groups_ > 1) {
        broadcast();
        if (world_rank_ == 0) {
            DEBUG_LOG("Broadcast");
        }
    }
    scatter();
    if (world_rank_ == 0) {
        DEBUG_LOG("Scatter");
    }
    alltoallv();
    if (world_rank_ == 0) {
        DEBUG_LOG("Alltoallv");
    }
}

void TransposeSlabBwd::broadcast() {
    // fwd の reduce() の逆: group_id==0 の fftbuf_ を全グループへ配布
    timer_.start(MPI_COMM_WORLD);
    MPI_Bcast(fftbuf_, static_cast<int>(fft_buf_size_), MPI_DOUBLE, 0, bcast_comm_);
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

void TransposeSlabBwd::scatter() {
    // fwd の reorder() の逆: fftbuf_ → recvbuf_ へデータを散らす
    // TODO: pos0_, pos1_, seg_ を使って実装する
    timer_.start(MPI_COMM_WORLD);

    // placeholder
    std::memset(recvbuf_, 0, (static_cast<size_t>(rdispls_.back()) + recvcounts_.back()) * sizeof(double));

    timer_.stop(t_calc_, MPI_COMM_WORLD);
}

void TransposeSlabBwd::alltoallv() {
    // fwd の alltoallv() の逆: send/recv を入れ替えて通信
    timer_.start(MPI_COMM_WORLD);
    MPI_Alltoallv(
        recvbuf_, recvcounts_.data(), rdispls_.data(), MPI_DOUBLE,
        sendbuf_, sendcounts_.data(), sdispls_.data(), MPI_DOUBLE,
        group_comm_
    );
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

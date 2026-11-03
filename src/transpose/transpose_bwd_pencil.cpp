#include "transpose/transpose_bwd_pencil.h"
#include <omp.h>
#include <cstring>
#include <string>
#include "debug.h"
#include "grouping.h"
#include "transpose/pencil_overlap.h"
#include <iostream>

TransposeBwdPencil::TransposeBwdPencil(const Grid& grid, const FFT& fft, const MPIEnv& mpi,
                                       BufferManager& buffer, Timer& timer, int method, int relay)
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
    group_id_   = grouping.group_id;
    local_rank_ = grouping.local_rank;

    relay_ = (relay != 0);

    // FFT を担当するランクか
    const bool is_fft = (fft.color() == 0);

    if (relay_) {
        // 従来: グループ内で Alltoallv、グループ間で Bcast
        MPI_Comm_split(MPI_COMM_WORLD, group_id_, local_rank_, &group_comm_);
        if (num_groups_ > 1) {
            MPI_Comm_split(MPI_COMM_WORLD, local_rank_, group_id_, &broadcast_comm_);
        }
        a2a_comm_ = group_comm_;
        npart_    = group_size_;
    } else {
        // リレーメッシュなし: WORLD 全体で 1 回の Alltoallv。Bcast は行わない。
        // FFT 担当ランクだけが送信側になる。
        a2a_comm_ = MPI_COMM_WORLD;
        npart_    = world_size_;
    }

    // 送受信カウント配列の初期化
    sendcounts_.assign(npart_, 0);
    sdispls_.assign(npart_, 0);
    recvcounts_.assign(npart_, 0);
    rdispls_.assign(npart_, 0);

    // 自分（FFT 側）の pencil。relay なしで FFT を担当しないランクは幅 0（何も送らない）。
    const int my_x0 = static_cast<int>(fft.local_0_start_x());
    const int my_y0 = static_cast<int>(fft.local_0_start_y());
    const int my_lx = (relay_ || is_fft) ? static_cast<int>(fft.local_n0_x()) : 0;
    const int my_ly = (relay_ || is_fft) ? static_cast<int>(fft.local_n0_y()) : 0;

    // ---- 相手（受け取る側）全員の grid 情報を収集 ----
    std::vector<int> vx0(npart_), vy0(npart_), vz0(npart_);
    std::vector<int> vnx(npart_), vny(npart_), vnz(npart_);
    MPI_Allgather(&grid.x0, 1, MPI_INT, vx0.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.y0, 1, MPI_INT, vy0.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.z0, 1, MPI_INT, vz0.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.nx, 1, MPI_INT, vnx.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.ny, 1, MPI_INT, vny.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.nz, 1, MPI_INT, vnz.data(), 1, MPI_INT, a2a_comm_);

    using pencil_overlap::Run;
    using pencil_overlap::split_runs;
    using pencil_overlap::clip;

    // ---- 送信側 ----
    // 相手 r の拡張配列 [nx+3][ny+3][nz+3] は大域座標 x0-1.., y0-1.., z0-1.. に対応し、
    // 両端が周期境界で折り返す。自分の pencil と重なる塊を r の昇順に数え上げ、
    // sendbuf の i 番目に入れる fftbuf 内の位置 pos0_[i] を作る。
    for (int r = 0; r < npart_; ++r) {
        const std::vector<Run> xruns = split_runs(vx0[r] - 1, vnx[r] + 3, Ng_);
        const std::vector<Run> yruns = split_runs(vy0[r] - 1, vny[r] + 3, Ng_);
        const int nz3_r = vnz[r] + 3;
        const int z_start = ((vz0[r] - 1) % Ng_ + Ng_) % Ng_;
        size_t to_r = 0;
        for (const Run& px : clip(xruns, my_x0, my_lx)) {
            for (const Run& py : clip(yruns, my_y0, my_ly)) {
                for (int dx = 0; dx < px.len; ++dx) {
                    const size_t xi = static_cast<size_t>(px.g0 + dx - my_x0);
                    for (int dy = 0; dy < py.len; ++dy) {
                        const size_t yi = static_cast<size_t>(py.g0 + dy - my_y0);
                        const size_t base = (xi * my_ly + yi) * stride_;
                        for (int k = 0; k < nz3_r; ++k) {
                            pos0_.push_back(base + (z_start + k) % Ng_);
                            ++to_r;
                        }
                    }
                }
            }
        }
        sendcounts_[r] = static_cast<int>(to_r);
    }
    for (int r = 1; r < npart_; ++r) {
        sdispls_[r] = sdispls_[r - 1] + sendcounts_[r - 1];
    }
    send_total_ = pos0_.size();

    // ---- 受信カウントを取得 ----
    MPI_Alltoall(sendcounts_.data(), 1, MPI_INT, recvcounts_.data(), 1, MPI_INT, a2a_comm_);

    for (int r = 1; r < npart_; ++r) {
        rdispls_[r] = rdispls_[r - 1] + recvcounts_[r - 1];
    }

    recv_total_ = static_cast<size_t>(rdispls_.back()) + recvcounts_.back();

    // ---- バッファ登録 ----
    fft_buf_size_ = align_to_64(static_cast<size_t>(fft.local_alloc()) * 2);

    size_t gridbuf_size = align_to_64(static_cast<size_t>(nx3_) * ny3_ * nz3_);
    size_t sendbuf_size = align_to_64(send_total_);
    size_t recvbuf_size = align_to_64(recv_total_);

    // fftbuf_ / gridbuf_ はオフセット 0(時間的に排他)
    // sendbuf_ は fftbuf_ を読みながら書くので別領域が必要
    size_t sendbuf_offset = std::max(gridbuf_size, fft_buf_size_);
    size_t recvbuf_offset = sendbuf_offset + sendbuf_size;
    size_t total_size = recvbuf_offset + recvbuf_size;

    buffer.register_buffer(fftbuf_,  0);
    buffer.register_buffer(gridbuf_, 0);
    buffer.register_buffer(sendbuf_, sendbuf_offset);
    buffer.register_buffer(recvbuf_, recvbuf_offset);
    buffer.update_max_size(total_size);

    // ---- 受信側 reorder インデックス構築 ----
    pos1_.resize(recv_total_);

    // FFT 側 Pencil 分割情報。relay なしのときは WORLD 全体で集め直し、
    // FFT を担当しないランクは 0 を報告する（そこからは何も届かない）。
    std::vector<int> ln0x, ln0y, l0sx, l0sy;
    if (relay_) {
        ln0x = fft.ln0x();
        ln0y = fft.ln0y();
        l0sx = fft.l0sx();
        l0sy = fft.l0sy();
    } else {
        ln0x.resize(npart_);
        ln0y.resize(npart_);
        l0sx.resize(npart_);
        l0sy.resize(npart_);
        int my_ln0x = is_fft ? static_cast<int>(fft.local_n0_x()) : 0;
        int my_ln0y = is_fft ? static_cast<int>(fft.local_n0_y()) : 0;
        int my_l0sx = static_cast<int>(fft.local_0_start_x());
        int my_l0sy = static_cast<int>(fft.local_0_start_y());
        MPI_Allgather(&my_ln0x, 1, MPI_INT, ln0x.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&my_ln0y, 1, MPI_INT, ln0y.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&my_l0sx, 1, MPI_INT, l0sx.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&my_l0sy, 1, MPI_INT, l0sy.data(), 1, MPI_INT, MPI_COMM_WORLD);
    }

    // 送信元 r（pencil）が送ってくる塊を、送信側と同じ順番で数え上げて、
    // 自分の拡張配列内の位置を決める
    {
        const std::vector<Run> xruns = split_runs(grid.x0 - 1, nx3_, Ng_);
        const std::vector<Run> yruns = split_runs(grid.y0 - 1, ny3_, Ng_);
        size_t index = 0;
        for (int r = 0; r < npart_; ++r) {
            size_t from_r = 0;
            for (const Run& px : clip(xruns, l0sx[r], ln0x[r])) {
                for (const Run& py : clip(yruns, l0sy[r], ln0y[r])) {
                    for (int dx = 0; dx < px.len; ++dx) {
                        for (int dy = 0; dy < py.len; ++dy) {
                            const size_t base =
                                (static_cast<size_t>(px.i0 + dx) * ny3_ + (py.i0 + dy)) * nz3_;
                            for (int k = 0; k < nz3_; ++k) {
                                if (index < recv_total_) pos1_[index] = base + k;
                                ++index;
                                ++from_r;
                            }
                        }
                    }
                }
            }
            if (from_r != static_cast<size_t>(recvcounts_[r])) {
                std::cerr << "TransposeBwdPencil: rank " << world_rank_ << " が相手 " << r
                          << " から受け取る個数が合わない（数え上げ " << from_r
                          << " / 通知 " << recvcounts_[r] << "）" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 42);
            }
        }
    }

    // ---- 送信側: 連続区間を memcpy リスト化 ----
    // sendbuf[i] = fftbuf[pos0_[i]]。pos0_ が連番(+1)の間は1回の memcpy にまとめる。
    if (send_total_ > 0) {
        size_t start_i = 0;                 // sendbuf 側の区間開始
        size_t start_src = pos0_[0];        // fftbuf 側の区間開始
        size_t len = 1;
        for (size_t i = 1; i < send_total_; ++i) {
            if (pos0_[i] == pos0_[i - 1] + 1) {
                ++len;
            } else {
                fwd_copy_dst_.push_back(start_i);
                fwd_copy_src_.push_back(start_src);
                fwd_copy_len_.push_back(len);
                start_i = i;
                start_src = pos0_[i];
                len = 1;
            }
        }
        fwd_copy_dst_.push_back(start_i);
        fwd_copy_src_.push_back(start_src);
        fwd_copy_len_.push_back(len);
    }

    // ---- 受信側: 連続区間を memcpy リスト化 ----
    // gridbuf[pos1_[i]] = recvbuf[i]。pos1_ が連番(+1)の間は1回の memcpy にまとめる。
    if (recv_total_ > 0) {
        size_t start_dst = pos1_[0];        // gridbuf 側の区間開始
        size_t start_i = 0;                 // recvbuf 側の区間開始
        size_t len = 1;
        for (size_t i = 1; i < recv_total_; ++i) {
            if (pos1_[i] == pos1_[i - 1] + 1) {
                ++len;
            } else {
                bwd_copy_dst_.push_back(start_dst);
                bwd_copy_src_.push_back(start_i);
                bwd_copy_len_.push_back(len);
                start_dst = pos1_[i];
                start_i = i;
                len = 1;
            }
        }
        bwd_copy_dst_.push_back(start_dst);
        bwd_copy_src_.push_back(start_i);
        bwd_copy_len_.push_back(len);
    }

    device_ = buffer.is_device();
#ifdef PM_USE_KOKKOS
    if (device_) {
        d_pos0_ = pm_dev_upload_index(pos0_);
        d_pos1_ = pm_dev_upload_index(pos1_);
    }
#endif

    if (world_rank_ == 0) {
        DEBUG_LOG(std::string("Set TransposeBwdPencil (relay=")
                  + (relay_ ? "on" : "off") + ")");
    }
}

void TransposeBwdPencil::execute() {
    // relay なしでは FFT 担当ランクが直接全格子ランクへ送るので Bcast は不要
    if (relay_ && num_groups_ > 1) {
        broadcast();
        if (world_rank_ == 0) DEBUG_LOG("Broadcast");
    }

    reorder_from_fft();
    if (world_rank_ == 0) DEBUG_LOG("Reorder from FFT");

    alltoallv();
    if (world_rank_ == 0) DEBUG_LOG("Alltoallv");

    reorder_from_alltoallv();
    if (world_rank_ == 0) DEBUG_LOG("Reorder from Alltoallv");
}

void TransposeBwdPencil::broadcast() {
    timer_.start(MPI_COMM_WORLD);

    for (size_t off = 0; off < fft_buf_size_; off += MPI_CHUNK_DOUBLES) {
        const int n = static_cast<int>(std::min(MPI_CHUNK_DOUBLES, fft_buf_size_ - off));
        MPI_Bcast(fftbuf_ + off, n, MPI_DOUBLE, 0, broadcast_comm_);
    }
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

void TransposeBwdPencil::reorder_from_fft() {
    timer_.start(MPI_COMM_WORLD);
#ifdef PM_USE_KOKKOS
    if (device_) {
        pm_dev_gather_index(sendbuf_, fftbuf_, d_pos0_, send_total_);
        timer_.stop(t_calc_, MPI_COMM_WORLD);
        return;
    }
#endif

    const size_t M = fwd_copy_len_.size();
    const size_t* __restrict cdst = fwd_copy_dst_.data();
    const size_t* __restrict csrc = fwd_copy_src_.data();
    const size_t* __restrict clen = fwd_copy_len_.data();

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t k = 0; k < M; ++k) {
        std::memcpy(sendbuf_ + cdst[k], fftbuf_ + csrc[k], clen[k] * sizeof(double));
    }

    timer_.stop(t_calc_, MPI_COMM_WORLD);
}

void TransposeBwdPencil::alltoallv() {
    // 送受信の向きが Fwd と逆(sendcounts/sdispls で送り、recvcounts/rdispls で受ける)
    timer_.start(MPI_COMM_WORLD);
    MPI_Alltoallv(
        sendbuf_, sendcounts_.data(), sdispls_.data(), MPI_DOUBLE,
        recvbuf_, recvcounts_.data(), rdispls_.data(), MPI_DOUBLE,
        a2a_comm_
    );
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

void TransposeBwdPencil::reorder_from_alltoallv() {
    timer_.start(MPI_COMM_WORLD);
#ifdef PM_USE_KOKKOS
    if (device_) {
        pm_dev_zero(gridbuf_, static_cast<size_t>(nx3_) * ny3_ * nz3_);
        pm_dev_scatter_index(gridbuf_, recvbuf_, d_pos1_, recv_total_);
        timer_.stop(t_calc_, MPI_COMM_WORLD);
        return;
    }
#endif

    std::memset(gridbuf_, 0, static_cast<size_t>(nx3_) * ny3_ * nz3_ * sizeof(double));

    const size_t M = bwd_copy_len_.size();
    const size_t* __restrict cdst = bwd_copy_dst_.data();
    const size_t* __restrict csrc = bwd_copy_src_.data();
    const size_t* __restrict clen = bwd_copy_len_.data();

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t k = 0; k < M; ++k) {
        std::memcpy(gridbuf_ + cdst[k], recvbuf_ + csrc[k], clen[k] * sizeof(double));
    }

    timer_.stop(t_calc_, MPI_COMM_WORLD);
}

TransposeBwdPencil::~TransposeBwdPencil() {
#ifdef PM_USE_KOKKOS
    pm_dev_free(d_pos0_);
    pm_dev_free(d_pos1_);
#endif
    // a2a_comm_ は group_comm_ か MPI_COMM_WORLD の借り物なので free しない
    if (group_comm_ != MPI_COMM_NULL) MPI_Comm_free(&group_comm_);
    if (broadcast_comm_ != MPI_COMM_NULL) MPI_Comm_free(&broadcast_comm_);
}

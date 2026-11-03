#include "transpose/transpose_fwd_pencil.h"
#include <omp.h>
#include <cstring>
#include "debug.h"
#include "grouping.h"
#include "transpose/pencil_overlap.h"
#include <iostream>

TransposeFwdPencil::TransposeFwdPencil::TransposeFwdPencil(const Grid& grid, const FFT& fft, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method, int relay)
    : timer_(timer),
      world_size_(mpi.world_size()),
      world_rank_(mpi.world_rank()),
      Ng_(grid.Ng),
      stride_(fft.stride()),
      nx1_(grid.nx + 1),
      ny1_(grid.ny + 1),
      nz1_(grid.nz + 1),
      slice_(static_cast<size_t>(ny1_) * nz1_)
{
    // タイマー登録
    t_comm_ = timer_.register_timer("TransposeFwd Comm");
    t_calc_ = timer_.register_timer("TransposeFwd Calc");

    // ---- グループ分け(FFTE2 と同じ引数・method で揃える)----
    Grouping grouping(fft.grouping_ng(), mpi, method);
    group_size_ = grouping.group_size;
    num_groups_ = grouping.num_groups;
    group_id_   = grouping.group_id;
    local_rank_ = grouping.local_rank;

    relay_ = (relay != 0);

    MPI_Comm_split(MPI_COMM_WORLD, group_id_, local_rank_, &group_comm_);

    if (relay_ && num_groups_ > 1) {
        MPI_Comm_split(MPI_COMM_WORLD, local_rank_, group_id_, &reduce_comm_);
    }

    if (relay_) {
        a2a_comm_ = group_comm_;
        npart_    = group_size_;
    } else {
        a2a_comm_ = MPI_COMM_WORLD;
        npart_    = world_size_;
    }

    // 送受信カウント配列の初期化
    sendcounts_.assign(npart_, 0);
    sdispls_.assign(npart_, 0);
    recvcounts_.assign(npart_, 0);
    rdispls_.assign(npart_, 0);

    // FFT 側の Pencil 分割情報を取得
    std::vector<int> ln0x, ln0y, l0sx, l0sy;
    if (relay_) {
        // group_comm_ 内で Allgather 済み（サイズ group_size_）
        ln0x = fft.ln0x();
        ln0y = fft.ln0y();
        l0sx = fft.l0sx();
        l0sy = fft.l0sy();
    } else {
        // WORLD 全体で集め直す。
        // FFT を担当しない（color != 0）ランクは 0 を報告し、
        // 以降の重なり判定で自動的に「送らない相手」になる。
        ln0x.resize(npart_); ln0y.resize(npart_);
        l0sx.resize(npart_); l0sy.resize(npart_);
        const bool is_fft = (fft.color() == 0);
        int my_ln0x = is_fft ? static_cast<int>(fft.local_n0_x()) : 0;
        int my_ln0y = is_fft ? static_cast<int>(fft.local_n0_y()) : 0;
        int my_l0sx = static_cast<int>(fft.local_0_start_x());
        int my_l0sy = static_cast<int>(fft.local_0_start_y());
        MPI_Allgather(&my_ln0x, 1, MPI_INT, ln0x.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&my_ln0y, 1, MPI_INT, ln0y.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&my_l0sx, 1, MPI_INT, l0sx.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&my_l0sy, 1, MPI_INT, l0sy.data(), 1, MPI_INT, MPI_COMM_WORLD);
    }

    using pencil_overlap::Run;
    using pencil_overlap::split_runs;
    using pencil_overlap::clip;

    // ---- 送信側 ----
    // 自分の密度配列 [nx1][ny1][nz1] は大域座標 x0.., y0.. に対応し、+ 側の1層は
    // 周期境界で折り返すことがある。相手 r の pencil と重なる塊を、r の昇順に数え上げる。
    // 1つの相手に複数の塊が届くことがあるので、個数は足し込み、並びは相手ごとにまとめる。
    {
        const std::vector<Run> xruns = split_runs(grid.x0, nx1_, Ng_);
        const std::vector<Run> yruns = split_runs(grid.y0, ny1_, Ng_);
        size_t offset = 0;
        for (int r = 0; r < npart_; ++r) {
            sdispls_[r] = static_cast<int>(offset);
            for (const Run& px : clip(xruns, l0sx[r], ln0x[r])) {
                for (const Run& py : clip(yruns, l0sy[r], ln0y[r])) {
                    const size_t n = static_cast<size_t>(px.len) * py.len * nz1_;
                    send_src_.push_back((static_cast<size_t>(px.i0) * ny1_ + py.i0) * nz1_);
                    send_dst_.push_back(offset);
                    send_nx_.push_back(px.len);
                    send_ny_.push_back(py.len * nz1_);
                    sendcounts_[r] += static_cast<int>(n);
                    offset += n;
                }
            }
        }
    }

    // ---- 受信カウントを取得 ----
    MPI_Alltoall(sendcounts_.data(), 1, MPI_INT,
                 recvcounts_.data(), 1, MPI_INT, a2a_comm_);

    for (int r = 1; r < npart_; ++r) {
        rdispls_[r] = rdispls_[r - 1] + recvcounts_[r - 1];
    }

    size_t send_total = static_cast<size_t>(sdispls_.back()) + sendcounts_.back();
    size_t recv_total = static_cast<size_t>(rdispls_.back()) + recvcounts_.back();

    // ---- バッファ登録 ----
    fft_buf_size_ = align_to_64(static_cast<size_t>(fft.local_alloc()) * 2);
    size_t rho_size = align_to_64(static_cast<size_t>(nx1_) * ny1_ * nz1_);
    size_t sendbuf_size = align_to_64(send_total);
    size_t recvbuf_size = align_to_64(recv_total);

    size_t sendbuf_offset = std::max(rho_size, fft_buf_size_);
    size_t recvbuf_offset = sendbuf_offset + sendbuf_size;
    size_t total_size = recvbuf_offset + recvbuf_size;

    buffer.register_buffer(rhobuf_, 0);
    buffer.register_buffer(fftbuf_, 0);
    buffer.register_buffer(sendbuf_, sendbuf_offset);
    buffer.register_buffer(recvbuf_, recvbuf_offset);
    buffer.update_max_size(total_size);

    // ---- 受信側 reorder インデックス構築 ----
    pos0_.resize(recv_total);
    pos1_.resize(recv_total);

    // ---- 相手（送信元）全員の grid 情報を収集 ----
    std::vector<int> vnx(npart_), vny(npart_), vnz(npart_);
    std::vector<int> vx0(npart_), vy0(npart_), vz0(npart_);
    MPI_Allgather(&grid.x0, 1, MPI_INT, vx0.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.y0, 1, MPI_INT, vy0.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.z0, 1, MPI_INT, vz0.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.nx, 1, MPI_INT, vnx.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.ny, 1, MPI_INT, vny.data(), 1, MPI_INT, a2a_comm_);
    MPI_Allgather(&grid.nz, 1, MPI_INT, vnz.data(), 1, MPI_INT, a2a_comm_);

    // 自分（FFT 側）の pencil。relay なしで FFT を担当しないランクは幅 0。
    const bool i_have_pencil = relay_ || (fft.color() == 0);
    const int my_x0 = static_cast<int>(fft.local_0_start_x());
    const int my_y0 = static_cast<int>(fft.local_0_start_y());
    const int my_lx = i_have_pencil ? static_cast<int>(fft.local_n0_x()) : 0;
    const int my_ly = i_have_pencil ? static_cast<int>(fft.local_n0_y()) : 0;

    // 送信元 r が送ってくる塊を、送信側と同じ順番で数え上げて、fftbuf 内の位置を決める
    size_t index = 0;
    for (int r = 0; r < npart_; ++r) {
        const std::vector<Run> xruns = split_runs(vx0[r], vnx[r] + 1, Ng_);
        const std::vector<Run> yruns = split_runs(vy0[r], vny[r] + 1, Ng_);
        const int nz1_r = vnz[r] + 1;
        size_t from_r = 0;
        for (const Run& px : clip(xruns, my_x0, my_lx)) {
            for (const Run& py : clip(yruns, my_y0, my_ly)) {
                for (int dx = 0; dx < px.len; ++dx) {
                    const size_t xi = static_cast<size_t>(px.g0 + dx - my_x0);
                    for (int dy = 0; dy < py.len; ++dy) {
                        const size_t yi = static_cast<size_t>(py.g0 + dy - my_y0);
                        const size_t base = (xi * my_ly + yi) * stride_;
                        for (int iz = 0; iz < nz1_r; ++iz) {
                            if (index < recv_total) pos0_[index] = base + (vz0[r] + iz) % Ng_;
                            ++index;
                            ++from_r;
                        }
                    }
                }
            }
        }
        if (from_r != static_cast<size_t>(recvcounts_[r])) {
            std::cerr << "TransposeFwdPencil: rank " << world_rank_ << " が相手 " << r
                      << " から受け取る個数が合わない（数え上げ " << from_r
                      << " / 通知 " << recvcounts_[r] << "）" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 41);
        }
    }

    // ---- pos0_ でソートして reorder 順を確定 ----
    std::vector<size_t> id(recv_total);
    std::iota(id.begin(), id.end(), 0);
    std::sort(id.begin(), id.end(), [&](size_t a, size_t b) {
        return pos0_[a] < pos0_[b];
    });

    std::vector<size_t> pos0_sorted(recv_total);
    for (size_t i = 0; i < recv_total; ++i) {
        pos0_sorted[i] = pos0_[id[i]];
        pos1_[i] = id[i];
    }
    pos0_.swap(pos0_sorted);

    // ---- 同じ fftbuf 位置への書き込みをまとめる seg_ を構築 ----
    seg_.reserve(recv_total + 1);
    if (recv_total > 0) seg_.push_back(0);
    for (size_t i = 1; i < recv_total; ++i) {
        if (pos0_[i] != pos0_[i - 1]) {
            seg_.push_back(i);
        }
    }
    seg_.push_back(recv_total);

    device_ = buffer.is_device();
    send_total_ = send_total;
#ifdef PM_USE_KOKKOS
    if (device_) {
        // 塊コピーの指示を、送信バッファ1要素ごとの読み出し位置に展開する
        std::vector<size_t> send_idx(send_total);
        for (size_t i = 0; i < send_nx_.size(); ++i) {
            const size_t row = static_cast<size_t>(send_ny_[i]);
            for (int j = 0; j < send_nx_[i]; ++j) {
                const size_t d = send_dst_[i] + j * row;
                const size_t s0 = send_src_[i] + j * slice_;
                for (size_t c = 0; c < row; ++c) send_idx[d + c] = s0 + c;
            }
        }
        d_send_ = pm_dev_upload_index(send_idx);
        d_key_ = pm_dev_upload_index(pos0_);
        d_idx_ = pm_dev_upload_index(pos1_);
        d_seg_ = pm_dev_upload_index(seg_);
    }
#endif

    if (world_rank_ == 0) DEBUG_LOG("Set TransposeFwdPencil");
}

TransposeFwdPencil::~TransposeFwdPencil() {
#ifdef PM_USE_KOKKOS
    pm_dev_free(d_send_);
    pm_dev_free(d_key_);
    pm_dev_free(d_idx_);
    pm_dev_free(d_seg_);
#endif
    if (group_comm_ != MPI_COMM_NULL) MPI_Comm_free(&group_comm_);
    if (reduce_comm_ != MPI_COMM_NULL) MPI_Comm_free(&reduce_comm_);
}

void TransposeFwdPencil::execute() {
    reorder_for_alltoallv();
    if (world_rank_ == 0) DEBUG_LOG("Reorder for Alltoallv");

    alltoallv();
    if (world_rank_ == 0) DEBUG_LOG("Alltoallv");

    reorder_for_fft();
    if (world_rank_ == 0) DEBUG_LOG("Reorder for FFT");

    if (relay_ && num_groups_ > 1) {
        reduce();
        if (world_rank_ == 0) DEBUG_LOG("Reduce");
    }
}

void TransposeFwdPencil::reorder_for_alltoallv() {
    timer_.start(MPI_COMM_WORLD);
#ifdef PM_USE_KOKKOS
    if (device_) {
        pm_dev_gather_index(sendbuf_, rhobuf_, d_send_, send_total_);
        timer_.stop(t_calc_, MPI_COMM_WORLD);
        return;
    }
#endif

    const size_t N = send_nx_.size();

    #pragma omp parallel
    {
        for (size_t i = 0; i < N; ++i) {
            double* __restrict dst_base = sendbuf_ + send_dst_[i];
            const double* __restrict src_base = rhobuf_ + send_src_[i];

            const int row_size  = send_ny_[i];           // 1 行(y*z)の要素数
            const int row_bytes = row_size * sizeof(double);

            #pragma omp for nowait
            for (int j = 0; j < send_nx_[i]; ++j) {
                std::memcpy(dst_base + j * row_size,
                            src_base + j * slice_,
                            row_bytes);
            }
        }
    }

    timer_.stop(t_calc_, MPI_COMM_WORLD);
}

void TransposeFwdPencil::alltoallv() {
    timer_.start(MPI_COMM_WORLD);
    MPI_Alltoallv(
        sendbuf_, sendcounts_.data(), sdispls_.data(), MPI_DOUBLE,
        recvbuf_, recvcounts_.data(), rdispls_.data(), MPI_DOUBLE,
        a2a_comm_
    );
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

void TransposeFwdPencil::reorder_for_fft() {
    timer_.start(MPI_COMM_WORLD);
#ifdef PM_USE_KOKKOS
    if (device_) {
        pm_dev_zero(fftbuf_, fft_buf_size_);
        pm_dev_fwd_reorder(fftbuf_, recvbuf_, d_seg_, d_idx_, d_key_, seg_.size() - 1);
        timer_.stop(t_calc_, MPI_COMM_WORLD);
        return;
    }
#endif

    const size_t S = seg_.size() - 1;
    const size_t* __restrict seg = seg_.data();
    const size_t* __restrict idx = pos1_.data();
    const size_t* __restrict key = pos0_.data();

    // fftbuf_ は pos0_ に登録された位置にしか書き込まれないので、
    // padding 領域のゴミを除去するために必要
    std::memset(fftbuf_, 0, fft_buf_size_ * sizeof(double));

    #pragma omp parallel for schedule(guided)
    for (size_t s = 0; s < S; ++s) {
        const size_t beg = seg[s];
        const size_t end = seg[s + 1];

        double acc = 0.0;
        for (size_t k = beg; k < end; ++k) {
            acc += recvbuf_[idx[k]];
        }
        fftbuf_[key[beg]] = acc;
    }

    timer_.stop(t_calc_, MPI_COMM_WORLD);
}

void TransposeFwdPencil::reduce() {
    timer_.start(MPI_COMM_WORLD);

    // count * sizeof(double) が 2^31 を超えないよう分割する
    for (size_t off = 0; off < fft_buf_size_; off += MPI_CHUNK_DOUBLES) {
        const int n = static_cast<int>(std::min(MPI_CHUNK_DOUBLES, fft_buf_size_ - off));
        if (group_id_ == 0) {
            MPI_Reduce(MPI_IN_PLACE, fftbuf_ + off, n, MPI_DOUBLE, MPI_SUM, 0, reduce_comm_);
        } else {
            MPI_Reduce(fftbuf_ + off, nullptr, n, MPI_DOUBLE, MPI_SUM, 0, reduce_comm_);
        }
    }
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

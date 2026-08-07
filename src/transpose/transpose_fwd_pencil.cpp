#include "transpose/transpose_fwd_pencil.h"
#include <omp.h>
#include <cstring>
#include "debug.h"
#include "grouping.h"

TransposeFwdPencil::TransposeFwdPencil(const Grid& grid, const FFT& fft, const MPIEnv& mpi, BufferManager& buffer,Timer& timer, int method)
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

    MPI_Comm_split(MPI_COMM_WORLD, group_id_, local_rank_, &group_comm_);

    if (num_groups_ > 1) {
        MPI_Comm_split(MPI_COMM_WORLD, local_rank_, group_id_, &reduce_comm_);
    }

    // 送受信カウント配列の初期化
    sendcounts_.assign(group_size_, 0);
    sdispls_.assign(group_size_, 0);
    recvcounts_.assign(group_size_, 0);
    rdispls_.assign(group_size_, 0);

    // FFT 側の Pencil 分割情報を取得
    std::vector<int> ln0x = fft.ln0x();   // 各プロセスの x 方向ローカルサイズ
    std::vector<int> ln0y = fft.ln0y();   // 各プロセスの y 方向ローカルサイズ
    std::vector<int> l0sx = fft.l0sx();   // 各プロセスの x 方向開始位置
    std::vector<int> l0sy = fft.l0sy();   // 各プロセスの y 方向開始位置

    int x0 = grid.x0;
    int y0 = grid.y0;

    // ---- 通常領域 ----
    for (int r = 0; r < group_size_; ++r) {
        int s0x = l0sx[r];
        int s1x = s0x + ln0x[r];
        int s0y = l0sy[r];
        int s1y = s0y + ln0y[r];
        int l0x = std::max(s0x, x0);
        int l1x = std::min(s1x, x0 + nx1_);
        int l0y = std::max(s0y, y0);
        int l1y = std::min(s1y, y0 + ny1_);

        if (l0x < l1x && l0y < l1y) {
            int planes_x = l1x - l0x;
            int planes_y = l1y - l0y;
            sendcounts_[r] = planes_x * planes_y * nz1_;
            send_src_.push_back(((l0x - x0) * ny1_ + (l0y - y0)) * nz1_);
            send_nx_.push_back(planes_x);
            send_ny_.push_back(planes_y * nz1_);
        }
    }

    // ---- x+ 境界 ----
    if (grid.coords[0] == grid.dims[0] - 1) {
        for (int r = 0; r < group_size_; ++r) {
            if (l0sx[r] != 0) continue;
            int s0y = l0sy[r];
            int s1y = s0y + ln0y[r];
            int l0y = std::max(s0y, y0);
            int l1y = std::min(s1y, y0 + ny1_);
            if (l0y < l1y) {
                int planes_y = l1y - l0y;
                sendcounts_[r] = planes_y * nz1_;
                send_src_.push_back((l0y - y0) * nz1_);
                send_nx_.push_back(1);
                send_ny_.push_back(planes_y * nz1_);
            }
        }
    }

    // ---- y+ 境界 ----
    if (grid.coords[1] == grid.dims[1] - 1) {
        for (int r = 0; r < group_size_; ++r) {
            if (l0sy[r] != 0) continue;
            int s0x = l0sx[r];
            int s1x = s0x + ln0x[r];
            int l0x = std::max(s0x, x0);
            int l1x = std::min(s1x, x0 + nx1_);
            if (l0x < l1x) {
                int planes_x = l1x - l0x;
                sendcounts_[r] = planes_x * nz1_;
                send_src_.push_back((l0x - x0) * ny1_ * nz1_);
                send_nx_.push_back(planes_x);
                send_ny_.push_back(nz1_);
            }
        }
    }

    // ---- xy+ 角 ----
    if (grid.coords[0] == grid.dims[0] - 1 && grid.coords[1] == grid.dims[1] - 1) {
        sendcounts_[0] = nz1_;
        send_src_.push_back((static_cast<size_t>(nx1_) * ny1_ - 1) * nz1_);
        send_nx_.push_back(1);
        send_ny_.push_back(nz1_);
    }

    // ---- sdispls と send_dst を計算 ----
    for (int r = 0; r < group_size_; ++r) {
        if (r > 0) sdispls_[r] = sdispls_[r - 1] + sendcounts_[r - 1];
        if (sendcounts_[r] > 0) send_dst_.push_back(sdispls_[r]);
    }

    // ---- 受信カウントを取得 ----
    MPI_Alltoall(sendcounts_.data(), 1, MPI_INT,
                 recvcounts_.data(), 1, MPI_INT, group_comm_);

    for (int r = 1; r < group_size_; ++r) {
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

    // ---- グループ内の全プロセスの grid 情報を収集 ----
    std::vector<int> vnx(group_size_), vny(group_size_), vnz(group_size_);
    std::vector<int> vx0(group_size_), vy0(group_size_), vz0(group_size_);
    MPI_Allgather(&grid.x0, 1, MPI_INT, vx0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.y0, 1, MPI_INT, vy0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.z0, 1, MPI_INT, vz0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.nx, 1, MPI_INT, vnx.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.ny, 1, MPI_INT, vny.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.nz, 1, MPI_INT, vnz.data(), 1, MPI_INT, group_comm_);

    int local_n0_x      = fft.local_n0_x();
    int local_n0_y      = fft.local_n0_y();
    int local_0_start_x = fft.local_0_start_x();
    int local_0_start_y = fft.local_0_start_y();

    size_t index = 0;
    for (int r = 0; r < group_size_; ++r) {
        if (recvcounts_[r] == 0) continue;

        int nz1_r = vnz[r] + 1;   // 送信元 r の z 方向サイズ(ゴースト込み)

        // x 方向
        int xi_base;
        //int wx;
        if ((vx0[r] - local_0_start_x) >= local_n0_x) {
            // x+ 折り返しゴースト
            xi_base = 0;
            //wx = 1;
        } else {
            // 通常領域(後半重なり含む): 重なり計算
            int l0x = std::max(vx0[r], local_0_start_x);
            int l1x = std::min(vx0[r] + vnx[r] + 1, local_0_start_x + local_n0_x);
            xi_base = l0x - local_0_start_x;
            //wx = l1x - l0x;
        }

        // y 方向
        int yi_base;
        int wy;
        if ((vy0[r] - local_0_start_y) >= local_n0_y) {
            // y+ 折り返しゴースト
            yi_base = 0;
            wy = 1;
        } else {
            // 通常領域(後半重なり含む): 重なり計算
            int l0y = std::max(vy0[r], local_0_start_y);
            int l1y = std::min(vy0[r] + vny[r] + 1, local_0_start_y + local_n0_y);
            yi_base = l0y - local_0_start_y;
            wy = l1y - l0y;
        }

        // 受信データ i 番目を (xi, yi, zi) に復元して fftbuf の位置を計算
        for (int i = 0; i < recvcounts_[r]; ++i) {
            int xi = xi_base + i / (wy * nz1_r);
            int yi = yi_base + (i / nz1_r) % wy;
            int zi = (vz0[r] + i % nz1_r) % Ng_;

            pos0_[index] = (static_cast<size_t>(xi) * local_n0_y + yi) * stride_ + zi;
            ++index;
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

    if (world_rank_ == 0) DEBUG_LOG("Set TransposeFwdPencil");
}

TransposeFwdPencil::~TransposeFwdPencil() {
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

    if (num_groups_ > 1) {
        reduce();
        if (world_rank_ == 0) DEBUG_LOG("Reduce");
    }
}

void TransposeFwdPencil::reorder_for_alltoallv() {
    timer_.start(MPI_COMM_WORLD);

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
        group_comm_
    );
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

void TransposeFwdPencil::reorder_for_fft() {
    timer_.start(MPI_COMM_WORLD);

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

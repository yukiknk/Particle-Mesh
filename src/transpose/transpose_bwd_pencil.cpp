#include "transpose/transpose_bwd_pencil.h"
#include <omp.h>
#include <cstring>
#include "debug.h"
#include "grouping.h"

TransposeBwdPencil::TransposeBwdPencil(const Grid& grid, const FFT& fft, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method)
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

    MPI_Comm_split(MPI_COMM_WORLD, group_id_, local_rank_, &group_comm_);

    if (num_groups_ > 1) {
        MPI_Comm_split(MPI_COMM_WORLD, local_rank_, group_id_, &broadcast_comm_);
    }

    // 送受信カウント配列の初期化
    sendcounts_.assign(group_size_, 0);
    sdispls_.assign(group_size_, 0);
    recvcounts_.assign(group_size_, 0);
    rdispls_.assign(group_size_, 0);

    // FFT 入力側の 2D 分割情報(自分の担当範囲)
    int local_0_start_x = static_cast<int>(fft.local_0_start_x());
    int local_0_start_y = static_cast<int>(fft.local_0_start_y());
    int local_n0_x = static_cast<int>(fft.local_n0_x());
    int local_n0_y = static_cast<int>(fft.local_n0_y());

    // 全プロセスの Grid 情報を収集
    std::vector<int> vnx(group_size_), vny(group_size_), vnz(group_size_);
    std::vector<int> vx0(group_size_), vy0(group_size_), vz0(group_size_);
    MPI_Allgather(&grid.x0, 1, MPI_INT, vx0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.y0, 1, MPI_INT, vy0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.z0, 1, MPI_INT, vz0.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.nx, 1, MPI_INT, vnx.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.ny, 1, MPI_INT, vny.data(), 1, MPI_INT, group_comm_);
    MPI_Allgather(&grid.nz, 1, MPI_INT, vnz.data(), 1, MPI_INT, group_comm_);

    // ---- 送信量計算 ----
    // 自分の FFT 担当範囲 [local_0_start_*, +local_n0_*) と
    // 各プロセス r の拡張グリッド [vx0[r]-1, vx0[r]+vnx[r]+2)(幅 nx+3, -側1層/+側2層)の重なり。
    // プロセス数が十分多い(3x3x3 以上)前提で、1つの r には1種類のみ送る。
    for (int r = 0; r < group_size_; ++r) {
        int nz3_r = vnz[r] + 3;

        // --- x 方向の重なり/折り返し判定 ---
        int wx = 0;
        // 通常領域
        int s0x = vx0[r] - 1;
        int s1x = s0x + vnx[r] + 3;
        int l0x = std::max(s0x, local_0_start_x);
        int l1x = std::min(s1x, local_0_start_x + local_n0_x);
        if (l0x < l1x) {
            wx = l1x - l0x;
        }
        // x+ 折り返し: r が x 末端、自分が x=0 側担当 → +側ゴースト2層が x=0,1 に来る
        else if ((vx0[r] + vnx[r] == Ng_) && local_0_start_x == 0) {
            wx = 2;
        }
        // x- 折り返し: r が x 始端(vx0[r]==0)、自分が x=Ng 側担当 → -側ゴースト1層が x=Ng-1 に来る
        else if (vx0[r] == 0 && (local_0_start_x + local_n0_x == Ng_)) {
            wx = 1;
        }

        // --- y 方向の重なり/折り返し判定 ---
        int wy = 0;
        int s0y = vy0[r] - 1;
        int s1y = s0y + vny[r] + 3;
        int l0y = std::max(s0y, local_0_start_y);
        int l1y = std::min(s1y, local_0_start_y + local_n0_y);
        if (l0y < l1y) {
            wy = l1y - l0y;
        }
        else if ((vy0[r] + vny[r] == Ng_) && local_0_start_y == 0) {
            wy = 2;
        }
        else if (vy0[r] == 0 && (local_0_start_y + local_n0_y == Ng_)) {
            wy = 1;
        }

        // 両方向に重なりがあれば送信
        if (wx > 0 && wy > 0) {
            sendcounts_[r] = wx * wy * nz3_r;
        }
    }

    for (int r = 1; r < group_size_; ++r) {
        sdispls_[r] = sdispls_[r - 1] + sendcounts_[r - 1];
    }

    send_total_ = static_cast<size_t>(sdispls_.back()) + sendcounts_.back();
    pos0_.resize(send_total_);

    size_t index = 0;
    for (int r = 0; r < group_size_; ++r) {
        if (sendcounts_[r] == 0) continue;

        int nz3_r = vnz[r] + 3;

        // --- x 方向: base(fftbuf ローカル座標)と幅 ---
        int xi_base;
        //int wx;
        int s0x = vx0[r] - 1;
        int s1x = s0x + vnx[r] + 3;
        int l0x = std::max(s0x, local_0_start_x);
        int l1x = std::min(s1x, local_0_start_x + local_n0_x);
        if (l0x < l1x) {
            xi_base = l0x - local_0_start_x;     // 通常領域
            //wx = l1x - l0x;
        } else if ((vx0[r] + vnx[r] == Ng_) && local_0_start_x == 0) {
            xi_base = 0;                          // x+ 折り返し: x=0,1 を送る
            //wx = 2;
        } else { // x- 折り返し: x=Ng-1 を送る(自分が x=Ng 側担当)
            xi_base = local_n0_x - 1;
            //wx = 1;
        }

        // --- y 方向 ---
        int yi_base, wy;
        int s0y = vy0[r] - 1;
        int s1y = s0y + vny[r] + 3;
        int l0y = std::max(s0y, local_0_start_y);
        int l1y = std::min(s1y, local_0_start_y + local_n0_y);
        if (l0y < l1y) {
            yi_base = l0y - local_0_start_y;
            wy = l1y - l0y;
        } else if ((vy0[r] + vny[r] == Ng_) && local_0_start_y == 0) {
            yi_base = 0;
            wy = 2;
        } else {
            yi_base = local_n0_y - 1;
            wy = 1;
        }

        // --- z 方向の開始(拡張グリッドの -1 ゴーストから、周期で巻く)---
        int z_start = (vz0[r] - 1 + Ng_) % Ng_;

        // sendbuf の各要素 i に対応する fftbuf 位置を計算
        for (int i = 0; i < sendcounts_[r]; ++i) {
            int xi = xi_base + i / (wy * nz3_r);
            int yi = yi_base + (i / nz3_r) % wy;
            int zi = (z_start + i % nz3_r) % Ng_;

            pos0_[index] = (static_cast<size_t>(xi) * local_n0_y + yi) * stride_ + zi;
            ++index;
        }
    }

    // ---- 受信カウントを取得 ----
    MPI_Alltoall(sendcounts_.data(), 1, MPI_INT, recvcounts_.data(), 1, MPI_INT, group_comm_);

    for (int r = 1; r < group_size_; ++r) {
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

    std::vector<int> ln0x = fft.ln0x();
    std::vector<int> ln0y = fft.ln0y();
    std::vector<int> l0sx = fft.l0sx();
    std::vector<int> l0sy = fft.l0sy();

    int x0 = grid.x0;
    int y0 = grid.y0;

    index = 0;
    for (int r = 0; r < group_size_; ++r) {
        if (recvcounts_[r] == 0) continue;

        // x 方向: 自分の拡張グリッド座標(x0-1 起点)での base と幅
        int xi_base = l0sx[r] - (x0 - 1);
        //int wx;
        if (xi_base < 0) {
            // 正側折り返し: r(x=0側担当)のデータが自分の +側ゴースト(幅2)へ
            xi_base = nx3_ - 2;
            //wx = 2;
        } else if (xi_base > nx3_ - 1) {
            // 負側折り返し: r(x=Ng側担当)のデータが自分の -側ゴースト(幅1)へ
            xi_base = 0;
            //wx = 1;
        } //else {
            //wx = std::min(nx3_ - xi_base, ln0x[r]);
        //}

        // y 方向(同じ構造)
        int yi_base = l0sy[r] - (y0 - 1);
        int wy;
        if (yi_base < 0) {
            yi_base = ny3_ - 2;
            wy = 2;
        } else if (yi_base > ny3_ - 1) {
            yi_base = 0;
            wy = 1;
        } else {
            wy = std::min(ny3_ - yi_base, ln0y[r]);
        }

        // recvbuf の i 番目を gridbuf のどこに書くか
        for (int i = 0; i < recvcounts_[r]; ++i) {
            int xi = xi_base + (i / (wy * nz3_));
            int yi = yi_base + ((i / nz3_) % wy);
            int zi = i % nz3_;

            pos1_[index] = (static_cast<size_t>(xi) * ny3_ + yi) * nz3_ + zi;
            ++index;
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

    if (world_rank_ == 0) DEBUG_LOG("Set TransposeBwdPencil");
}

void TransposeBwdPencil::execute() {
    if (num_groups_ > 1) {
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
    MPI_Bcast(fftbuf_, static_cast<int>(fft_buf_size_), MPI_DOUBLE, 0, broadcast_comm_);
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

void TransposeBwdPencil::reorder_from_fft() {
    timer_.start(MPI_COMM_WORLD);

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
        group_comm_
    );
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

void TransposeBwdPencil::reorder_from_alltoallv() {
    timer_.start(MPI_COMM_WORLD);

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
    if (group_comm_ != MPI_COMM_NULL) MPI_Comm_free(&group_comm_);
    if (broadcast_comm_ != MPI_COMM_NULL) MPI_Comm_free(&broadcast_comm_);
}

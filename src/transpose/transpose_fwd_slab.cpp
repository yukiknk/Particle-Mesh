#include "transpose/transpose_fwd_slab.h"
#include <omp.h>
#include <cstring>
#include "debug.h"
#include "grouping.h"

TransposeFwdSlab::TransposeFwdSlab(const Grid& grid, const FFT& fft, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method)
    : timer_(timer),
      world_size_(mpi.world_size()),
      world_rank_(mpi.world_rank()),
      Ng_(grid.Ng),
      stride_(fft.stride()),
      nx1_(grid.nx + 1),
      ny1_(grid.ny + 1),
      nz1_(grid.nz + 1)
{
    //タイマー登録
    t_comm_ = timer_.register_timer("TransposeFwd Comm");
    t_calc_ = timer_.register_timer("TransposeFwd Calc");

    Grouping grouping(fft.grouping_ng(), mpi, method);
    group_size_ = grouping.group_size;
    num_groups_ = grouping.num_groups;
    group_id_ = grouping.group_id;
    local_rank_ = grouping.local_rank;
    
    // グループ内コミュニケータ作成
    MPI_Comm_split(MPI_COMM_WORLD, group_id_, local_rank_, &group_comm_);
    
    // Reduce用コミュニケータ（num_groups > 1の場合のみ使用）
    if (num_groups_ > 1) {
        MPI_Comm_split(MPI_COMM_WORLD, local_rank_, group_id_, &reduce_comm_);
    }
    
    // 送受信量の計算（グループ内）
    sendcounts_.assign(group_size_, 0);
    sdispls_.assign(group_size_, 0);
    recvcounts_.assign(group_size_, 0);
    rdispls_.assign(group_size_, 0);

    int x0 = grid.x0;
    int local_n0 = static_cast<int>(fft.local_n0());

    for (int r = 0; r < group_size_; ++r) {
        int s0 = r * local_n0;
        int s1 = s0 + local_n0;
        int l0 = std::max(s0, x0);
        int l1 = std::min(s1, x0 + nx1_);
        
        if (l0 < l1) {
            int planes = l1 - l0;
            sendcounts_[r] = planes * ny1_ * nz1_;
            sdispls_[r] = (l0 - x0) * ny1_ * nz1_;
        }
    }

    // x方向の+側境界のプロセスはゴーストをlocal_rank 0に送る
    if (grid.coords[0] == grid.dims[0] - 1) {
        sendcounts_[0] = ny1_ * nz1_;
        sdispls_[0] = grid.nx * ny1_ * nz1_;
    }

    MPI_Alltoall(sendcounts_.data(), 1, MPI_INT, recvcounts_.data(), 1, MPI_INT, group_comm_);

    for (int r = 1; r < group_size_; ++r) {
        rdispls_[r] = rdispls_[r - 1] + recvcounts_[r - 1];
    }

    size_t recv_total = static_cast<size_t>(rdispls_.back()) + recvcounts_.back();

    // FFTバッファサイズ
    fft_buf_size_ = align_to_64(static_cast<size_t>(fft.local_alloc()) * 2);

    // バッファ登録
    size_t sendbuf_size = align_to_64(static_cast<size_t>(nx1_) * ny1_ * nz1_);
    size_t recvbuf_offset = std::max(sendbuf_size, fft_buf_size_);
    size_t total_size = recvbuf_offset + align_to_64(recv_total);

    buffer.register_buffer(sendbuf_, 0);
    buffer.register_buffer(fftbuf_, 0);
    buffer.register_buffer(recvbuf_, recvbuf_offset);
    buffer.update_max_size(total_size);

    //reorder用のインデックスを構築
    pos0_.resize(recv_total);
    pos1_.resize(recv_total);

    int local_0_start = fft.local_0_start();

    // グループ内の各プロセスのGrid情報を収集
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

    size_t index = 0;
    for (int r = 0; r < group_size_; ++r) {
        int xi_base = group_vx0[r] - local_0_start;
        if (xi_base < 0 || xi_base >= local_n0) xi_base = 0;
        
        int nyz = (group_vny[r] + 1) * (group_vnz[r] + 1);
        
        for (int i = 0; i < recvcounts_[r]; ++i) {
            int xi = xi_base + i / nyz;
            int yi = (group_vy0[r] + (i / (group_vnz[r] + 1)) % (group_vny[r] + 1)) % Ng_;
            int zi = (group_vz0[r] + i % (group_vnz[r] + 1)) % Ng_;

            pos0_[index] = (static_cast<size_t>(xi) * Ng_ + yi) * stride_ + zi;
            ++index;
        }
    }

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

    seg_.reserve(recv_total + 1);
    if(recv_total > 0) seg_.push_back(0);
    for (size_t i = 1; i < recv_total; ++i) {
        if (pos0_[i] != pos0_[i - 1]) {
            seg_.push_back(i);
        }
    }
    seg_.push_back(recv_total);

    if(world_rank_ == 0) {
        DEBUG_LOG("Set TransposeFwd");
    }
}

TransposeFwdSlab::~TransposeFwdSlab() {
    if (group_comm_ != MPI_COMM_NULL) MPI_Comm_free(&group_comm_);
    if (reduce_comm_ != MPI_COMM_NULL) MPI_Comm_free(&reduce_comm_);
}

void TransposeFwdSlab::execute() {
    alltoallv();
    if(world_rank_ == 0) {
        DEBUG_LOG("Alltoallv");
    }
    reorder(); 
    if(world_rank_ == 0) {
        DEBUG_LOG("Reorder");
    }   
    if (num_groups_ > 1) {
        reduce();
        if(world_rank_ == 0) {
            DEBUG_LOG("Reduce");
        }
    }
}

void TransposeFwdSlab::alltoallv() {
    timer_.start(MPI_COMM_WORLD);
    MPI_Alltoallv(
        sendbuf_, sendcounts_.data(), sdispls_.data(), MPI_DOUBLE,
        recvbuf_, recvcounts_.data(), rdispls_.data(), MPI_DOUBLE,
        group_comm_
    );
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

void TransposeFwdSlab::reorder() {
    timer_.start(MPI_COMM_WORLD);
    const size_t S = seg_.size() - 1;
    const size_t* __restrict seg = seg_.data();
    const size_t* __restrict idx = pos1_.data();
    const size_t* __restrict key = pos0_.data();
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

void TransposeFwdSlab::reduce() {
    timer_.start(MPI_COMM_WORLD);
    if (group_id_ == 0) {
        MPI_Reduce(MPI_IN_PLACE, fftbuf_, fft_buf_size_, MPI_DOUBLE, MPI_SUM, 0, reduce_comm_);
    } else {
        MPI_Reduce(fftbuf_, nullptr, fft_buf_size_, MPI_DOUBLE, MPI_SUM, 0, reduce_comm_);
    }
    timer_.stop(t_comm_, MPI_COMM_WORLD);
}

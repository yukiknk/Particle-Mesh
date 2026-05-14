#include "fft/fft_ffte1.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include "debug.h"
#include <omp.h>
#include "grouping.h"

FFT_FFTE1::FFT_FFTE1(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method) 
    : timer_(timer), world_rank_(mpi.world_rank()), Ng_(Ng), stride_(Ng) {
    
    t_fft_ = timer_.register_timer("FFT");
    t_ifft_ = timer_.register_timer("IFFT");
    t_green_ = timer_.register_timer("Green");

    // FFTE は NPU <= Ng/2 の制約があるため、Ng/2 を渡す
    Grouping grouping(Ng_ / 2, mpi, method);
    color_ = grouping.color;
    int local_rank = grouping.local_rank;
    int fft_size = grouping.group_size;

    MPI_Comm_split(MPI_COMM_WORLD, color_, local_rank, &comm_);

    // 全プロセスで local_n0_, local_0_start_ を設定（TransposeSlabFwd/Bwd が参照するため）
    local_n0_ = static_cast<ptrdiff_t>(Ng_) / fft_size;
    local_0_start_ = local_n0_ * static_cast<ptrdiff_t>(local_rank);

    // 全プロセスで local_n1_, local_alloc_ を設定（TransposeSlabFwd/Bwd の fft_buf_size_ 計算に必要）
    // local_rank は MPI_Comm_split で comm_ 内の rank_ と一致するため、ここで計算可能
    local_n1_ = static_cast<ptrdiff_t>(Ng_) / (2 * fft_size);
    local_1_start_ = local_n1_ * static_cast<ptrdiff_t>(local_rank);
    if (local_rank == 0) ++local_n1_;
    else ++local_1_start_;
    local_alloc_ = static_cast<ptrdiff_t>(Ng_) * Ng_ * local_n1_;

    // 全プロセスでバッファ登録（TransposeSlabFwd/Bwd が使用するため）
    size_t real_size = align_to_64(static_cast<size_t>(Ng_ / fft_size + 2) * Ng_ * Ng_);
    size_t calc_size = align_to_64(static_cast<size_t>(Ng_ / fft_size + 2) * Ng_ * Ng_);

    buffer.register_buffer(real_, 0);
    buffer.register_buffer(calc_, real_size);
    buffer.update_max_size(real_size + calc_size);

    if (color_ == 0) {
        MPI_Comm_size(comm_, &size_);
        MPI_Comm_rank(comm_, &rank_);
        fortran_Comm_ = MPI_Comm_c2f(comm_);

        size_t green_bytes = ((static_cast<size_t>(local_alloc_) * sizeof(double) + 63) / 64) * 64;
        green_ = static_cast<double*>(std::aligned_alloc(64, green_bytes));
        if (green_ == nullptr) {
            std::cerr << "FFTE: green allocation failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const double pi_n = 2.0 * M_PI / Ng_;
        #pragma omp parallel for collapse(2)
        for (int ix = 0; ix < Ng_; ++ix) {
            for (int iy = 0; iy < Ng_; ++iy) {
                for (ptrdiff_t iz = 0; iz < local_n1_; ++iz) {
                    size_t index = (static_cast<size_t>(ix) * Ng_ + iy) * local_n1_ + iz;
                    if (!(ix == 0 && iy == 0 && iz + local_1_start_ == 0)) {
                        // Reference implementation does NOT scale by Ng^3 here
                        green_[index] = -3.0 * Omega0 / (4.0 * 
                                        (3.0 - cos(pi_n * ix) - cos(pi_n * iy) - cos(pi_n * (iz + local_1_start_))));
                    } else {
                        green_[index] = 0.0;
                    }
                }
            }
        }
    }
    if (world_rank_ == 0) {
        DEBUG_LOG("Set FFTE");
    }
}

void FFT_FFTE1::create_plan() {
    if (color_ == 0) {
        complex_ = reinterpret_cast<std::complex<double>*>(real_);
        
        int IOPT = 0;
        pdzfft3d_(real_, calc_, &Ng_, &Ng_, &Ng_, &fortran_Comm_, &rank_, &size_, &IOPT);
        pzdfft3d_(real_, calc_, &Ng_, &Ng_, &Ng_, &fortran_Comm_, &rank_, &size_, &IOPT);
    }
}

FFT_FFTE1::~FFT_FFTE1() {
    if (color_ == 0) {
        std::free(green_);
    }
    if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
}

void FFT_FFTE1::forward() {
    if (color_ == 0) {
        timer_.start(comm_);
        int IOPT = -2;
        pdzfft3d_(real_, calc_, &Ng_, &Ng_, &Ng_, &fortran_Comm_, &rank_, &size_, &IOPT);
        timer_.stop(t_fft_, comm_);
        if (world_rank_ == 0) {
            DEBUG_LOG("FFTE FFT");
        }
    }
}

void FFT_FFTE1::backward() {
    if (color_ == 0) {
        timer_.start(comm_);
        int IOPT = 2;
        pzdfft3d_(real_, calc_, &Ng_, &Ng_, &Ng_, &fortran_Comm_, &rank_, &size_, &IOPT);
        timer_.stop(t_ifft_, comm_);
        if (world_rank_ == 0) {
            DEBUG_LOG("FFTE IFFT");
        }
    }
}

void FFT_FFTE1::apply_green(double a) {
    if (color_ != 0) return;

    timer_.start(comm_);
    const double inv_a = 1.0 / a;
    #pragma omp parallel for
    for (ptrdiff_t i = 0; i < local_alloc_; ++i) {
        double val = green_[i] * inv_a;
        complex_[i] *= val;
    }
    timer_.stop(t_green_, comm_);
    if (world_rank_ == 0) {
        DEBUG_LOG("FFTE Green");
    }
}

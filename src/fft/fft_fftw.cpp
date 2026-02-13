#include "fft/fft_fftw.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include "debug.h"
#include <omp.h>

FFT::FFT(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer) : timer_(timer), world_rank_(mpi.world_rank()), Ng_(Ng), stride_(Ng + 2) {
    t_fft_ = timer_.register_timer("FFT");
    t_ifft_ = timer_.register_timer("IFFT");
    t_green_ = timer_.register_timer("Green");
    
    fftw_init_threads();
    fftw_mpi_init();
    fftw_plan_with_nthreads(mpi.nthreads());

    int rank = mpi.world_rank();
    int size = mpi.world_size();

    color_ = rank / Ng_;
    int local_rank = rank % Ng_;
    MPI_Comm_split(MPI_COMM_WORLD, color_, local_rank, &comm_);

    local_alloc_ = fftw_mpi_local_size_3d_transposed(
        Ng_, Ng_, Ng_ / 2 + 1, comm_,
        &local_n0_, &local_0_start_,
        &local_n1_, &local_1_start_
    );

    size_t real_size = align_to_64(2 * local_alloc_);
    size_t complex_size = align_to_64(2 * local_alloc_);
    size_t complex_offset = real_size;
    size_t total_size = real_size + complex_size;

    buffer.register_buffer(real_, 0);
    buffer.register_buffer(reinterpret_cast<double*&>(complex_), complex_offset);
    buffer.update_max_size(total_size);

    if (color_ == 0) {
        size_t green_bytes = ((local_alloc_ * sizeof(double) + 63) / 64) * 64;
        green_ = static_cast<double*>(std::aligned_alloc(64, green_bytes));
        if (green_ == nullptr) {
            std::cerr << "FFT: green allocation failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const double pi_n = 2.0 * M_PI / Ng_;
        #pragma omp parallel for collapse(2)
        for (ptrdiff_t j = 0; j < local_n1_; ++j) {
            for (int i = 0; i < Ng_; ++i) {
                for (int k = 0; k < Ng_ / 2 + 1; ++k) {
                    size_t index = (j * Ng_ + i) * (Ng_ / 2 + 1) + k;
                    int gj = j + local_1_start_;
                    if (i == 0 && gj == 0 && k == 0) {
                        green_[index] = 0.0;
                    } else {
                        green_[index] = -3.0 * Omega0 / (
                            4.0 * static_cast<double>(Ng_) * Ng_ * Ng_ *
                            (3.0 - cos(pi_n * i) - cos(pi_n * gj) - cos(pi_n * k))
                        );
                    }
                }
            }
        }
    }
    if(world_rank_ == 0) {
        DEBUG_LOG("Set FFT");
    }
}

void FFT::create_plan() {
    if (color_ == 0) {
        forward_ = fftw_mpi_plan_dft_r2c_3d(
            Ng_, Ng_, Ng_, real_, complex_, comm_,
            FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT
        );
        backward_ = fftw_mpi_plan_dft_c2r_3d(
            Ng_, Ng_, Ng_, complex_, real_, comm_,
            FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN
        );
    }
}

FFT::~FFT() {
    if (color_ == 0) {
        if (forward_) fftw_destroy_plan(forward_);
        if (backward_) fftw_destroy_plan(backward_);
        std::free(green_);
    }
    if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
}

void FFT::forward() {
    if (color_ == 0) {
        timer_.start(comm_);
        fftw_mpi_execute_dft_r2c(forward_, real_, complex_);
        timer_.stop(t_fft_, comm_);
        if(world_rank_ == 0) {
            DEBUG_LOG("FFT");
        }
    }
}

void FFT::backward() {
    if (color_ == 0) {
        timer_.start(comm_);
        fftw_mpi_execute_dft_c2r(backward_, complex_, real_);
        timer_.stop(t_ifft_, comm_);
        if(world_rank_ == 0) {
            DEBUG_LOG("IFFT");
        }
    }
}

void FFT::apply_green(double a) {
    if (color_ != 0) return;

    timer_.start(comm_);
    const double inv_a = 1.0 / a;
    #pragma omp parallel for
    for (ptrdiff_t i = 0; i < local_alloc_; ++i) {
        double val = green_[i] * inv_a;
        complex_[i][0] *= val;
        complex_[i][1] *= val;
    }
    timer_.stop(t_green_, comm_);
    if(world_rank_ == 0) {
        DEBUG_LOG("Green");
    }
}

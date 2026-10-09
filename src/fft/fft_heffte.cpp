#include "fft/fft_heffte.h"
#include "fft/fft_checksum.h"
#include "utils.h"
#include "debug.h"
#include "grouping.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <omp.h>

#include "heffte.h"

// ---- バックエンドとデバイスメモリの切り替え -------------------------------
#ifdef PM_HEFFTE_CPU
  #ifndef PM_HEFFTE_CPU_BACKEND
  #define PM_HEFFTE_CPU_BACKEND heffte::backend::fftw
  #endif
namespace {
using backend_tag = PM_HEFFTE_CPU_BACKEND;
const char* const kBackendName = "cpu";
void* dev_alloc(size_t bytes) {
    void* p = std::malloc(bytes > 0 ? bytes : 1);
    if (p == nullptr) { std::cerr << "FFT_HEFFTE: malloc failed" << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}
void dev_free(void* p) { std::free(p); }
void host_to_dev(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }
void dev_to_host(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }
void dev_sync() {}
} // namespace
#else
  #include <cuda_runtime.h>
namespace {
using backend_tag = heffte::backend::cufft;
const char* const kBackendName = "cufft";
void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "FFT_HEFFTE: " << what << " failed: " << cudaGetErrorString(e) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, bytes > 0 ? bytes : 1), "cudaMalloc");
    return p;
}
void dev_free(void* p) { if (p != nullptr) cudaFree(p); }
void host_to_dev(void* dst, const void* src, size_t bytes) {
    cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
}
void dev_to_host(void* dst, const void* src, size_t bytes) {
    cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
}
void dev_sync() { cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }
} // namespace
#endif

struct FFT_HEFFTE::Impl {
    std::unique_ptr<heffte::fft3d_r2c<backend_tag>> fft;

    size_t n_real = 0;                              // local_n0 * Ng * Ng
    size_t n_cplx = 0;                              // local_n0 * Ng * (Ng/2+1)

    double* d_real = nullptr;                       // デバイス側
    std::complex<double>* d_cplx = nullptr;         // デバイス側

    std::vector<std::complex<double>> h_cplx;       // ホスト側（Green を掛ける場所）
    std::vector<double> green;                      // ホスト側
};

FFT_HEFFTE::FFT_HEFFTE(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method)
    : impl_(std::make_unique<Impl>()), timer_(timer), Ng_(Ng), world_rank_(mpi.world_rank())
{
    t_fft_   = timer_.register_timer("FFT");
    t_ifft_  = timer_.register_timer("IFFT");
    t_green_ = timer_.register_timer("Green");
    t_copy_  = timer_.register_timer("FFT Host<->Dev Copy");

    Grouping grouping(cap_for(Ng_), mpi, method);
    color_ = grouping.color;
    const int local_rank = grouping.local_rank;
    const int group_size = grouping.group_size;

    MPI_Comm_split(MPI_COMM_WORLD, color_, local_rank, &comm_);

    // Transpose は「グループ内ランク r が x = [r*local_n0, (r+1)*local_n0) を持つ」
    // 等幅スラブを前提にしている。group_size は Ng を割り切る必要がある。
    if (Ng_ % group_size != 0) {
        if (world_rank_ == 0) {
            std::cerr << "FFT_HEFFTE: group_size=" << group_size
                      << " が Ng=" << Ng_ << " を割り切らない" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    local_n0_      = Ng_ / group_size;
    local_0_start_ = static_cast<ptrdiff_t>(local_rank) * local_n0_;

    impl_->n_real = static_cast<size_t>(local_n0_) * Ng_ * Ng_;
    impl_->n_cplx = static_cast<size_t>(local_n0_) * Ng_ * (Ng_ / 2 + 1);

    // Transpose 側は 2 * local_alloc 個の double を FFT バッファとして扱う
    local_alloc_ = static_cast<ptrdiff_t>((impl_->n_real + 1) / 2);

    buffer.register_buffer(real_, 0);
    buffer.update_max_size(align_to_64(2 * static_cast<size_t>(local_alloc_)));

    if (color_ == 0) {
        // Green 関数（波数空間、出力ボックスの並び: x が最も遅く、z(r2c 軸) が最も速い）
        const int nzc = Ng_ / 2 + 1;
        impl_->green.resize(impl_->n_cplx);
        double* green = impl_->green.data();
        const double pi_n = 2.0 * M_PI / Ng_;
        const ptrdiff_t x_start = local_0_start_;
        #pragma omp parallel for collapse(2)
        for (ptrdiff_t ix = 0; ix < local_n0_; ++ix) {
            for (int iy = 0; iy < Ng_; ++iy) {
                const ptrdiff_t gx = ix + x_start;
                for (int kz = 0; kz < nzc; ++kz) {
                    const size_t index = (static_cast<size_t>(ix) * Ng_ + iy) * nzc + kz;
                    if (gx == 0 && iy == 0 && kz == 0) {
                        green[index] = 0.0;
                    } else {
                        green[index] = -3.0 * Omega0 / (
                            4.0 * static_cast<double>(Ng_) * Ng_ * Ng_ *
                            (3.0 - cos(pi_n * gx) - cos(pi_n * iy) - cos(pi_n * kz))
                        );
                    }
                }
            }
        }
    }
    if (world_rank_ == 0) {
        DEBUG_LOG("Set FFT (heFFTe)");
    }
}

void FFT_HEFFTE::create_plan() {
    if (color_ != 0) return;

    // ホスト配列の添字は (x * Ng + y) * Ng + z。heFFTe の box3d は既定で
    // 次元0 が最も速く回るので、次元0 = z, 次元1 = y, 次元2 = x と対応させる。
    const int x0 = static_cast<int>(local_0_start_);
    const int x1 = static_cast<int>(local_0_start_ + local_n0_ - 1);
    heffte::box3d<> inbox ({0, 0, x0}, {Ng_ - 1, Ng_ - 1, x1});
    heffte::box3d<> outbox({0, 0, x0}, {Ng_ / 2, Ng_ - 1, x1});   // r2c は次元0 (z) を半分にする
    const int r2c_direction = 0;

    impl_->fft = std::make_unique<heffte::fft3d_r2c<backend_tag>>(inbox, outbox, r2c_direction, comm_);

    if (static_cast<size_t>(impl_->fft->size_inbox())  != impl_->n_real ||
        static_cast<size_t>(impl_->fft->size_outbox()) != impl_->n_cplx) {
        std::cerr << "FFT_HEFFTE: box size mismatch (inbox " << impl_->fft->size_inbox()
                  << " vs " << impl_->n_real << ", outbox " << impl_->fft->size_outbox()
                  << " vs " << impl_->n_cplx << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    impl_->d_real = static_cast<double*>(dev_alloc(impl_->n_real * sizeof(double)));
    impl_->d_cplx = static_cast<std::complex<double>*>(dev_alloc(impl_->n_cplx * sizeof(std::complex<double>)));
    impl_->h_cplx.resize(impl_->n_cplx);

    int rank = 0, size = 0;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);
    if (rank == 0) {
        std::cout << "heFFTe: backend=" << kBackendName << "  ranks=" << size
                  << "  slab=" << local_n0_ << " x " << Ng_ << " x " << Ng_ << std::endl;
    }
}

FFT_HEFFTE::~FFT_HEFFTE() {
    if (impl_) {
        impl_->fft.reset();
        dev_free(impl_->d_real);
        dev_free(impl_->d_cplx);
    }
    if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
}

void FFT_HEFFTE::forward() {
    if (color_ != 0) return;

    timer_.start(comm_);
    host_to_dev(impl_->d_real, real_, impl_->n_real * sizeof(double));
    timer_.stop(t_copy_, comm_);

    timer_.start(comm_);
    impl_->fft->forward(impl_->d_real, impl_->d_cplx);
    dev_sync();
    timer_.stop(t_fft_, comm_);

    timer_.start(comm_);
    dev_to_host(impl_->h_cplx.data(), impl_->d_cplx, impl_->n_cplx * sizeof(std::complex<double>));
    timer_.stop(t_copy_, comm_);

    if (world_rank_ == 0) {
        DEBUG_LOG("FFT");
    }
}

void FFT_HEFFTE::apply_green(double a) {
    if (color_ != 0) return;

    timer_.start(comm_);
    const double inv_a = 1.0 / a;
    const size_t n = impl_->n_cplx;
    const double* green = impl_->green.data();
    std::complex<double>* c = impl_->h_cplx.data();
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        c[i] *= green[i] * inv_a;
    }
    timer_.stop(t_green_, comm_);
    if (world_rank_ == 0) {
        DEBUG_LOG("Green");
    }
}

void FFT_HEFFTE::backward() {
    if (color_ != 0) return;

    timer_.start(comm_);
    host_to_dev(impl_->d_cplx, impl_->h_cplx.data(), impl_->n_cplx * sizeof(std::complex<double>));
    timer_.stop(t_copy_, comm_);

    timer_.start(comm_);
    impl_->fft->backward(impl_->d_cplx, impl_->d_real);
    dev_sync();
    timer_.stop(t_ifft_, comm_);

    timer_.start(comm_);
    dev_to_host(real_, impl_->d_real, impl_->n_real * sizeof(double));
    timer_.stop(t_copy_, comm_);

    if (pm_checksum_enabled()) pm_print_checksum("HEFFTE", real_, local_n0_, Ng_, Ng_, comm_);

    if (world_rank_ == 0) {
        DEBUG_LOG("IFFT");
    }
}

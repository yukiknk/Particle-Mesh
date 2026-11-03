#include "fft/fft_heffte.h"
#include "fft/fft_checksum.h"
#include "utils.h"
#include "debug.h"
#include "grouping.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include "heffte.h"
#include "pm_device.h"

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

// ボックスは heFFTe の次元順で持つ: 次元0 = z（最も速い）, 次元1 = y, 次元2 = x（最も遅い）。
// ホスト配列の添字 (x * ny + y) * nz + z と一致する。
struct FFT_HEFFTE::Impl {
    std::unique_ptr<heffte::fft3d_r2c<backend_tag>> fft;

    size_t n_real = 0;                              // nx * ny * nz
    size_t n_cplx = 0;                              // nx * ny * nzc

    double* d_real = nullptr;                       // デバイス側
    std::complex<double>* d_cplx = nullptr;         // デバイス側

    std::vector<std::complex<double>> h_cplx;       // ホスト側（Green を掛ける場所）
    std::vector<double> green;                      // ホスト側
    double* d_green = nullptr;                      // デバイスモード: GPU 側の Green 関数
};

namespace {
// 環境変数を整数として読む。未設定や空なら fallback。
int env_int(const char* name, int fallback) {
    const char* e = std::getenv(name);
    if (e == nullptr || e[0] == '\0') return fallback;
    return std::atoi(e);
}

// PM_HEFFTE_ALG: 再分割の通信方法。0=alltoallv(既定) 1=alltoall 2=p2p 3=p2p_plined。名前でも指定できる。
int env_alg(int fallback) {
    const char* e = std::getenv("PM_HEFFTE_ALG");
    if (e == nullptr || e[0] == '\0') return fallback;
    const std::string v(e);
    if (v == "alltoallv" || v == "0") return 0;
    if (v == "alltoall"  || v == "1") return 1;
    if (v == "p2p"       || v == "2") return 2;
    if (v == "p2p_plined"|| v == "3") return 3;
    return fallback;
}

const char* alg_name(int a) {
    return a == 1 ? "alltoall" : (a == 2 ? "p2p" : (a == 3 ? "p2p_plined" : "alltoallv"));
}

const char* box_name(FFT_HEFFTE::Box b) {
    return b == FFT_HEFFTE::Box::Slab ? "slab" : "pencil";
}
} // namespace

FFT_HEFFTE::FFT_HEFFTE(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method,
                       Box box, int cap_override)
    : impl_(std::make_unique<Impl>()), timer_(timer), Ng_(Ng), box_(box), world_rank_(mpi.world_rank())
{
    t_fft_   = timer_.register_timer("FFT");
    t_ifft_  = timer_.register_timer("IFFT");
    t_green_ = timer_.register_timer("Green");
    t_copy_  = timer_.register_timer("FFT Host<->Dev Copy");

    auto die = [&](const std::string& msg) {
        if (world_rank_ == 0) std::cerr << "FFT_HEFFTE(" << box_name(box_) << "): " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    };

    cap_ = cap_for(Ng_, box_);
    if (box_ == Box::Pencil && cap_override > 0) cap_ = cap_override;

    Grouping grouping(cap_, mpi, method);
    color_ = grouping.color;
    const int local_rank = grouping.local_rank;
    const int group_size = grouping.group_size;
    MPI_Comm_split(MPI_COMM_WORLD, color_, local_rank, &comm_);

    group_size_ = group_size;
    local_rank_ = local_rank;
    nz_ = Ng_;

    // 出力（波数空間）を転置された並びにするか。PM_HEFFTE_OUT=same で入力と同じ分割にする（比較用）。
    // ランクごとに食い違わないよう、グループ内ランク 0 の値を全員に配る。
    {
        const char* e = std::getenv("PM_HEFFTE_OUT");
        int t = (e != nullptr && std::string(e) == "same") ? 0 : 1;
        MPI_Bcast(&t, 1, MPI_INT, 0, comm_);
        transposed_out_ = (t != 0);
    }

    if (box_ == Box::Slab) {
        // Transpose は「グループ内ランク r が x = [r*n, (r+1)*n) を持つ」等幅スラブを前提にしている
        if (Ng_ % group_size != 0) die("group_size が Ng を割り切らない");
        nx_ = Ng_ / group_size;
        x0_ = local_rank * nx_;
        y0_ = 0; ny_ = Ng_;
    } else {
        // FFT_FFTE2 と同じ2次元分割（大きい方を x に割り当てる）
        int dims[2] = {0, 0};
        MPI_Dims_create(group_size, 2, dims);
        const int npux = dims[0];
        const int npuy = dims[1];
        if (npux > Ng_ || npuy > Ng_ || Ng_ % npux != 0 || Ng_ % npuy != 0) {
            die("pencil 分割 " + std::to_string(npux) + " x " + std::to_string(npuy) + " が Ng を割り切らない");
        }
        npux_ = npux;
        npuy_ = npuy;
        nx_ = Ng_ / npux;
        ny_ = Ng_ / npuy;
        x0_ = (local_rank / npuy) * nx_;
        y0_ = (local_rank % npuy) * ny_;

        ln0x_.resize(group_size); ln0y_.resize(group_size);
        l0sx_.resize(group_size); l0sy_.resize(group_size);
        MPI_Allgather(&nx_, 1, MPI_INT, ln0x_.data(), 1, MPI_INT, comm_);
        MPI_Allgather(&ny_, 1, MPI_INT, ln0y_.data(), 1, MPI_INT, comm_);
        MPI_Allgather(&x0_, 1, MPI_INT, l0sx_.data(), 1, MPI_INT, comm_);
        MPI_Allgather(&y0_, 1, MPI_INT, l0sy_.data(), 1, MPI_INT, comm_);
    }

    {
        int b[6];
        cplx_box_of(local_rank_, b);
        ox0_ = b[0]; onx_ = b[1];
        oy0_ = b[2]; ony_ = b[3];
        oz0_ = b[4]; onz_ = b[5];
    }

    impl_->n_real = static_cast<size_t>(nx_) * ny_ * nz_;
    impl_->n_cplx = static_cast<size_t>(onx_) * ony_ * onz_;

    // Transpose 側は 2 * local_alloc 個の double を FFT バッファとして扱う
    local_alloc_ = static_cast<ptrdiff_t>((impl_->n_real + 1) / 2);

    buffer.register_buffer(real_, 0);
    buffer.update_max_size(align_to_64(2 * static_cast<size_t>(local_alloc_)));
    device_ = buffer.is_device();
#ifndef PM_USE_KOKKOS
    if (device_) die("デバイスモードは PM_USE_KOKKOS でビルドしたときだけ使える");
#endif

    if (color_ == 0) {
        // Green 関数（波数空間ボックスの並び: x が最も遅く、z(r2c 軸) が最も速い）
        impl_->green.resize(impl_->n_cplx);
        double* green = impl_->green.data();
        const double pi_n = 2.0 * M_PI / Ng_;
        #pragma omp parallel for collapse(2)
        for (int ix = 0; ix < onx_; ++ix) {
            for (int iy = 0; iy < ony_; ++iy) {
                const int gx = ix + ox0_;
                const int gy = iy + oy0_;
                for (int iz = 0; iz < onz_; ++iz) {
                    const int gz = iz + oz0_;
                    const size_t index = (static_cast<size_t>(ix) * ony_ + iy) * onz_ + iz;
                    if (gx == 0 && gy == 0 && gz == 0) {
                        green[index] = 0.0;
                    } else {
                        green[index] = -3.0 * Omega0 / (
                            4.0 * static_cast<double>(Ng_) * Ng_ * Ng_ *
                            (3.0 - cos(pi_n * gx) - cos(pi_n * gy) - cos(pi_n * gz))
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

// グループ内ランク r の実空間ボックス {x0, nx, y0, ny, z0, nz}
void FFT_HEFFTE::real_box_of(int r, int b[6]) const {
    if (box_ == Box::Slab) {
        const int n = Ng_ / group_size_;
        b[0] = r * n; b[1] = n;
        b[2] = 0;     b[3] = Ng_;
    } else {
        const int n_x = Ng_ / npux_, n_y = Ng_ / npuy_;
        b[0] = (r / npuy_) * n_x; b[1] = n_x;
        b[2] = (r % npuy_) * n_y; b[3] = n_y;
    }
    b[4] = 0; b[5] = Ng_;
}

// グループ内ランク r の波数空間ボックス {x0, nx, y0, ny, z0, nz}（z は 0 .. Ng/2）
void FFT_HEFFTE::cplx_box_of(int r, int b[6]) const {
    const int nzc = Ng_ / 2 + 1;
    if (!transposed_out_) {
        // 入力と同じ分割（z だけ半分）
        real_box_of(r, b);
        b[4] = 0; b[5] = nzc;
        return;
    }
    if (box_ == Box::Slab) {
        // y 方向のスラブ。x と z は全体（FFTW の TRANSPOSED_OUT と同じ形）
        const int n = Ng_ / group_size_;
        b[0] = 0;     b[1] = Ng_;
        b[2] = r * n; b[3] = n;
        b[4] = 0;     b[5] = nzc;
    } else {
        // x 方向に通った pencil。y は入力と同じ npuy 分割、z (0 .. Ng/2) を npux 分割
        const int cx = r / npuy_, cy = r % npuy_;
        const int n_y = Ng_ / npuy_;
        const int z_lo = static_cast<int>(static_cast<long long>(nzc) * cx / npux_);
        const int z_hi = static_cast<int>(static_cast<long long>(nzc) * (cx + 1) / npux_);
        b[0] = 0;         b[1] = Ng_;
        b[2] = cy * n_y;  b[3] = n_y;
        b[4] = z_lo;      b[5] = z_hi - z_lo;
    }
}

void FFT_HEFFTE::create_plan() {
    if (color_ != 0) return;

    // ホスト配列の添字は (ix * ny + iy) * nz + iz。heFFTe の box3d は既定で
    // 次元0 が最も速く回るので、次元0 = z, 次元1 = y, 次元2 = x と対応させる。
    auto to_box = [](const int b[6]) {
        return heffte::box3d<>({b[4], b[2], b[0]}, {b[4] + b[5] - 1, b[2] + b[3] - 1, b[0] + b[1] - 1});
    };
    int rb[6], cb[6];
    real_box_of(local_rank_, rb);
    cplx_box_of(local_rank_, cb);
    heffte::box3d<> inbox  = to_box(rb);
    heffte::box3d<> outbox = to_box(cb);
    const int r2c_direction = 0;    // 次元0 (z) を半分にする

    // heFFTe の内部の回し方。環境変数で切り替えられる（未設定なら下の既定値）。
    //   PM_HEFFTE_PENCILS   : 1 = 内部を pencil で回す、0 = slab で回す。既定は渡す形に合わせる
    //   PM_HEFFTE_REORDER   : 1 = 再分割のたびに並べ替える、0 = 並べ替えずに飛び飛びの FFT を使う
    //   PM_HEFFTE_GPU_AWARE : 1 = GPU 上のデータをそのまま MPI に渡す、0 = いったんホストに移す
    //   PM_HEFFTE_ALG       : alltoallv / alltoall / p2p / p2p_plined
    // ランクごとに食い違わないよう、グループ内ランク 0 の値を全員に配る。
    heffte::plan_options options = heffte::default_options<backend_tag>();
    int opt[4] = {
        env_int("PM_HEFFTE_PENCILS", box_ == Box::Pencil ? 1 : 0),
        env_int("PM_HEFFTE_REORDER", options.use_reorder ? 1 : 0),
        env_int("PM_HEFFTE_GPU_AWARE", options.use_gpu_aware ? 1 : 0),
        env_alg(0),
    };
    MPI_Bcast(opt, 4, MPI_INT, 0, comm_);
    options.use_pencils   = (opt[0] != 0);
    options.use_reorder   = (opt[1] != 0);
    options.use_gpu_aware = (opt[2] != 0);
    options.algorithm     = opt[3] == 1 ? heffte::reshape_algorithm::alltoall
                          : opt[3] == 2 ? heffte::reshape_algorithm::p2p
                          : opt[3] == 3 ? heffte::reshape_algorithm::p2p_plined
                                        : heffte::reshape_algorithm::alltoallv;

    impl_->fft = std::make_unique<heffte::fft3d_r2c<backend_tag>>(inbox, outbox, r2c_direction, comm_, options);

    // heFFTe が立てる計画を同じ入力で作り直し、形が変わる段（組み替え）の数を数える。表示用。
    int num_reshapes = 0;
    {
        heffte::ioboxes<> all;
        for (int r = 0; r < group_size_; ++r) {
            int b[6];
            real_box_of(r, b); all.in.push_back(to_box(b));
            cplx_box_of(r, b); all.out.push_back(to_box(b));
        }
        const heffte::logic_plan3d<int> plan = heffte::plan_operations(all, r2c_direction, options, local_rank_);
        for (int i = 0; i < 4; ++i) {
            if (!heffte::match(plan.in_shape[i], plan.out_shape[i])) ++num_reshapes;
        }
    }

    if (static_cast<size_t>(impl_->fft->size_inbox())  != impl_->n_real ||
        static_cast<size_t>(impl_->fft->size_outbox()) != impl_->n_cplx) {
        std::cerr << "FFT_HEFFTE: box size mismatch (inbox " << impl_->fft->size_inbox()
                  << " vs " << impl_->n_real << ", outbox " << impl_->fft->size_outbox()
                  << " vs " << impl_->n_cplx << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    impl_->d_cplx = static_cast<std::complex<double>*>(dev_alloc(impl_->n_cplx * sizeof(std::complex<double>)));
    if (device_) {
        // 実空間は共有バッファ（GPU 上）をそのまま使う。Green 関数は GPU へ一度だけ送る
        impl_->d_green = static_cast<double*>(dev_alloc(impl_->n_cplx * sizeof(double)));
        host_to_dev(impl_->d_green, impl_->green.data(), impl_->n_cplx * sizeof(double));
    } else {
        impl_->d_real = static_cast<double*>(dev_alloc(impl_->n_real * sizeof(double)));
        impl_->h_cplx.resize(impl_->n_cplx);
    }

    int rank = 0, size = 0;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);
    if (rank == 0) {
        std::cout << "heFFTe: backend=" << kBackendName << "  box=" << box_name(box_)
                  << "  ranks=" << size
                  << "  local=" << nx_ << " x " << ny_ << " x " << nz_
                  << "  pencils=" << opt[0] << " reorder=" << opt[1]
                  << " gpu_aware=" << opt[2] << " alg=" << alg_name(opt[3])
                  << "  out=" << (transposed_out_ ? "transposed" : "same")
                  << " reshapes=" << num_reshapes
                  << "  data=" << (device_ ? "device" : "host") << std::endl;
    }
}

FFT_HEFFTE::~FFT_HEFFTE() {
    if (impl_) {
        impl_->fft.reset();
        dev_free(impl_->d_real);
        dev_free(impl_->d_cplx);
        dev_free(impl_->d_green);
    }
    if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
}

void FFT_HEFFTE::forward() {
    if (color_ != 0) return;

    if (device_) {
        timer_.start(comm_);
        impl_->fft->forward(real_, impl_->d_cplx);
        dev_sync();
        timer_.stop(t_fft_, comm_);
        if (world_rank_ == 0) DEBUG_LOG("FFT");
        return;
    }

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

#ifdef PM_USE_KOKKOS
    if (device_) {
        timer_.start(comm_);
        pm_dev_scale_complex(reinterpret_cast<double*>(impl_->d_cplx), impl_->d_green, impl_->n_cplx, 1.0 / a);
        timer_.stop(t_green_, comm_);
        if (world_rank_ == 0) DEBUG_LOG("Green");
        return;
    }
#endif

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

    if (device_) {
        timer_.start(comm_);
        impl_->fft->backward(impl_->d_cplx, real_);
        dev_sync();
        timer_.stop(t_ifft_, comm_);
        if (pm_checksum_enabled()) {
            // 照合用にホストへ写してから計算する（計測には含めない）
            std::vector<double> h(impl_->n_real);
            dev_to_host(h.data(), real_, impl_->n_real * sizeof(double));
            pm_print_checksum("HEFFTE", h.data(), static_cast<size_t>(nx_) * ny_, nz_, nz_, comm_);
        }
        if (world_rank_ == 0) DEBUG_LOG("IFFT");
        return;
    }

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

    if (pm_checksum_enabled()) {
        pm_print_checksum("HEFFTE", real_, static_cast<size_t>(nx_) * ny_, nz_, nz_, comm_);
    }

    if (world_rank_ == 0) {
        DEBUG_LOG("IFFT");
    }
}

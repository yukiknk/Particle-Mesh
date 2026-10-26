// Kokkos 版 Interpolater（-DPM_USE_KOKKOS でビルドしたとき）。
//
// 同じソースで GPU（Kokkos::Cuda）と CPU（Kokkos::OpenMP）の両方に対応する。
// 実行時に環境変数 PM_INTERP で選ぶ:
//   PM_INTERP=gpu（既定。CUDA が有効なとき） / PM_INTERP=cpu
//
// 従来版（src/interpolater.cpp）との違い:
//   - deposit は Kokkos::Experimental::ScatterView で書く。ScatterView は実行先に合わせて足し込み方を選ぶ:
//       CPU (OpenMP): スレッドごとに格子を複製して最後に合計（従来版と同じ方式。atomic を使わない）
//       GPU (Cuda)  : 1枚の格子に atomic で足し込む
//     比較用に PM_DEPOSIT=atomic とすると、CPU でも atomic にする（既定は PM_DEPOSIT=scatter）。
//   - 粒子と格子はホスト側（Particle と BufferManager の共有バッファ）にあるので、
//     GPU のときは deposit / gather のたびにホスト <-> GPU をコピーする。
//     コピーの時間は "Interpolater Copy" として別に測る。
//   - deposit でコピーした粒子座標は、同じイテレーションの gather でもそのまま使う
//     （deposit と gather の間で粒子は動かないため）。
#ifdef PM_USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#include "interpolater.h"
#include "debug.h"

namespace {

using DevExec  = Kokkos::DefaultExecutionSpace;       // CUDA があれば Cuda
using HostExec = Kokkos::DefaultHostExecutionSpace;   // OpenMP
using DevMem   = DevExec::memory_space;
using HostMem  = HostExec::memory_space;

template <class Mem>
using Arr = Kokkos::View<double*, Kokkos::LayoutRight, Mem>;   // CUDA の既定 LayoutLeft と揃わないので明示
// 既存のホスト配列（Particle、共有バッファ）をそのまま Kokkos から触るためのビュー
using HostWrap = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

namespace KE = Kokkos::Experimental;
using HostDevice = Kokkos::Device<HostExec, HostMem>;
using DevDevice  = Kokkos::Device<DevExec, DevMem>;
// CPU: スレッドごとに複製（atomic なし）
using SVHostDup    = KE::ScatterView<double*, Kokkos::LayoutRight, HostDevice, KE::ScatterSum,
                                     KE::ScatterDuplicated, KE::ScatterNonAtomic>;
// CPU: 複製せず atomic（比較用）
using SVHostAtomic = KE::ScatterView<double*, Kokkos::LayoutRight, HostDevice, KE::ScatterSum,
                                     KE::ScatterNonDuplicated, KE::ScatterAtomic>;
// GPU: 複製せず atomic（Cuda での ScatterView の既定と同じ）
using SVDev        = KE::ScatterView<double*, Kokkos::LayoutRight, DevDevice, KE::ScatterSum,
                                     KE::ScatterNonDuplicated, KE::ScatterAtomic>;

struct Geometry {
    int x0, y0, z0;     // このプロセスの格子の開始位置
    int nyg, nzg;       // ny+1, nz+1（ゴースト込み）
    int ny3, nz3;       // ny+3, nz+3（拡張グリッド）
    int64_t ncell;      // (nx+1)*(ny+1)*(nz+1)
    double mass;
};

bool use_atomic_deposit_from_env() {
    const char* e = std::getenv("PM_DEPOSIT");
    return e != nullptr && std::string(e) == "atomic";
}

bool use_gpu_from_env() {
    const char* e = std::getenv("PM_INTERP");
    if (e == nullptr || e[0] == '\0') return !std::is_same_v<DevExec, HostExec>;
    return std::string(e) != "cpu";
}

// ---- 質量割当（CIC）。sv は呼ぶ前に reset() し、呼んだ後に contribute() で格子へ集める ----
template <class Exec, class P, class SV>
void deposit_kernel(const P& x, const P& y, const P& z, const SV& sv, int64_t np, const Geometry g) {
    Kokkos::parallel_for(
        "PM::deposit", Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>(0, np),
        KOKKOS_LAMBDA(const int64_t p) {
            auto rho = sv.access();
            const double xp = x(p), yp = y(p), zp = z(p);
            const int ix = static_cast<int>(xp);
            const int iy = static_cast<int>(yp);
            const int iz = static_cast<int>(zp);
            const double wx1 = xp - ix, wy1 = yp - iy, wz1 = zp - iz;
            const double wx0 = 1.0 - wx1, wy0 = 1.0 - wy1, wz0 = 1.0 - wz1;

            const int64_t li = ix - g.x0, lj = iy - g.y0, lk = iz - g.z0;
            const int64_t lij   = (li * g.nyg + lj) * g.nzg;
            const int64_t lij1  = (li * g.nyg + lj + 1) * g.nzg;
            const int64_t li1j  = ((li + 1) * g.nyg + lj) * g.nzg;
            const int64_t li1j1 = ((li + 1) * g.nyg + lj + 1) * g.nzg;

            const double wz0m = wz0 * g.mass, wz1m = wz1 * g.mass;
            rho(lij   + lk)     += wx0 * wy0 * wz0m;
            rho(lij   + lk + 1) += wx0 * wy0 * wz1m;
            rho(lij1  + lk)     += wx0 * wy1 * wz0m;
            rho(lij1  + lk + 1) += wx0 * wy1 * wz1m;
            rho(li1j  + lk)     += wx1 * wy0 * wz0m;
            rho(li1j  + lk + 1) += wx1 * wy0 * wz1m;
            rho(li1j1 + lk)     += wx1 * wy1 * wz0m;
            rho(li1j1 + lk + 1) += wx1 * wy1 * wz1m;
        });
}

// ---- ポテンシャル勾配（中心差分）。in は拡張グリッド、出力は (nx+1)*(ny+1)*(nz+1) ----
template <class Exec, class G, class O>
void gradient_kernel(const G& in, const O& gx, const O& gy, const O& gz, const Geometry g) {
    const int64_t yz3 = static_cast<int64_t>(g.ny3) * g.nz3;
    const int64_t nz3 = g.nz3;
    Kokkos::parallel_for(
        "PM::gradient", Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>(0, g.ncell),
        KOKKOS_LAMBDA(const int64_t idx) {
            const int64_t kk = idx % g.nzg;
            const int64_t t  = idx / g.nzg;
            const int64_t jj = t % g.nyg;
            const int64_t ii = t / g.nyg;
            const int64_t base = (ii + 1) * yz3 + (jj + 1) * nz3 + (kk + 1);
            gx(idx) = (in(base - yz3) - in(base + yz3)) * 0.5;
            gy(idx) = (in(base - nz3) - in(base + nz3)) * 0.5;
            gz(idx) = (in(base - 1)   - in(base + 1))   * 0.5;
        });
}

// ---- 粒子位置での勾配を CIC で補間し、速度を更新 ----
template <class Exec, class P, class O, class V>
void gather_kernel(const P& x, const P& y, const P& z, const O& gx, const O& gy, const O& gz,
                   const V& vx, const V& vy, const V& vz, int64_t np, const Geometry g, double fa_da) {
    Kokkos::parallel_for(
        "PM::gather", Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>(0, np),
        KOKKOS_LAMBDA(const int64_t p) {
            const double xp = x(p), yp = y(p), zp = z(p);
            const int ix = static_cast<int>(xp);
            const int iy = static_cast<int>(yp);
            const int iz = static_cast<int>(zp);
            const double wx1 = xp - ix, wy1 = yp - iy, wz1 = zp - iz;
            const double wx0 = 1.0 - wx1, wy0 = 1.0 - wy1, wz0 = 1.0 - wz1;

            const int64_t li = ix - g.x0, lj = iy - g.y0, lk = iz - g.z0;
            const int64_t lij   = (li * g.nyg + lj) * g.nzg;
            const int64_t lij1  = (li * g.nyg + lj + 1) * g.nzg;
            const int64_t li1j  = ((li + 1) * g.nyg + lj) * g.nzg;
            const int64_t li1j1 = ((li + 1) * g.nyg + lj + 1) * g.nzg;
            const int64_t c[8] = {lij + lk, lij + lk + 1, lij1 + lk, lij1 + lk + 1,
                                  li1j + lk, li1j + lk + 1, li1j1 + lk, li1j1 + lk + 1};

            const double wz0m = wz0 * g.mass, wz1m = wz1 * g.mass;
            const double w[8] = {wx0 * wy0 * wz0m, wx0 * wy0 * wz1m, wx0 * wy1 * wz0m, wx0 * wy1 * wz1m,
                                 wx1 * wy0 * wz0m, wx1 * wy0 * wz1m, wx1 * wy1 * wz0m, wx1 * wy1 * wz1m};
            double ax = 0.0, ay = 0.0, az = 0.0;
            for (int n = 0; n < 8; ++n) {
                ax += gx(c[n]) * w[n];
                ay += gy(c[n]) * w[n];
                az += gz(c[n]) * w[n];
            }
            vx(p) += ax * fa_da;
            vy(p) += ay * fa_da;
            vz(p) += az * fa_da;
        });
}

}  // namespace

// ------------------------------------------------------------------ Kokkos の初期化
PmKokkosScope::PmKokkosScope(int& argc, char** argv) {
    if (!Kokkos::is_initialized()) Kokkos::initialize(argc, argv);
}
PmKokkosScope::~PmKokkosScope() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
}

// ------------------------------------------------------------------ 中身
struct Interpolater::KImpl {
    bool gpu = true;
    bool atomic_cpu = false;          // CPU で atomic を使うか（PM_DEPOSIT=atomic）
    Geometry g{};
    int64_t np = 0;

    // GPU 用（gpu のときだけ確保）
    Arr<DevMem> x, y, z, vx, vy, vz;   // 粒子
    Arr<DevMem> rho;                   // deposit の出力 (ncell)
    Arr<DevMem> ext;                   // gather の入力（拡張グリッド）
    Arr<DevMem> gx, gy, gz;            // 勾配 (ncell)

    // CPU 用（cpu のときだけ確保）
    Arr<HostMem> hrho;                 // deposit の出力 (ncell)。最後に共有バッファへ写す
    Arr<HostMem> hgx, hgy, hgz;        // 勾配 (ncell)

    // ScatterView（作り直すと複製格子を毎回確保し直すので、一度作って使い回す）
    SVDev sv_dev;
    SVHostDup sv_host_dup;
    SVHostAtomic sv_host_atomic;
};

Interpolater::Interpolater(const Grid& grid, Particle& particle, const MPIEnv& mpi, BufferManager& buffer, Timer& timer)
    : particle_(particle),
      timer_(timer),
      x0_(grid.x0), y0_(grid.y0), z0_(grid.z0),
      nthreads_(mpi.nthreads()),
      mass_(1.0),
      Da_(0.001),
      ny_ghost_(static_cast<size_t>(grid.ny + 1)),
      nz_ghost_(static_cast<size_t>(grid.nz + 1)),
      ncell_local_(static_cast<size_t>(grid.nx + 1) * (grid.ny + 1) * (grid.nz + 1)),
      nx3_(grid.nx + 3), ny3_(grid.ny + 3), nz3_(grid.nz + 3),
      world_rank_(mpi.world_rank()),
      kimpl_(std::make_unique<KImpl>())
{
    t_deposit_ = timer_.register_timer("Interpolater Deposit");
    t_gather_  = timer_.register_timer("Interpolater Gather");
    t_copy_    = timer_.register_timer("Interpolater Copy");

    // 作業用格子はスレッドごとに持たないので、共有バッファは deposit の出力 1 枚分で足りる
    // （gather の入力の拡張グリッドは Transpose 側が大きさを登録している）
    thread_buf_size_ = align_to_64(ncell_local_);
    buffer.register_buffer(buf_, 0);
    buffer.update_max_size(thread_buf_size_);
    ext_size_ = static_cast<size_t>(nx3_) * ny3_ * nz3_;

    KImpl& k = *kimpl_;
    k.gpu = use_gpu_from_env();
    k.atomic_cpu = use_atomic_deposit_from_env();
    k.g = Geometry{x0_, y0_, z0_, static_cast<int>(ny_ghost_), static_cast<int>(nz_ghost_),
                   ny3_, nz3_, static_cast<int64_t>(ncell_local_), mass_};
    k.np = static_cast<int64_t>(particle_.np);

    if (k.gpu) {
        k.x  = Arr<DevMem>(Kokkos::view_alloc("x",  Kokkos::WithoutInitializing), k.np);
        k.y  = Arr<DevMem>(Kokkos::view_alloc("y",  Kokkos::WithoutInitializing), k.np);
        k.z  = Arr<DevMem>(Kokkos::view_alloc("z",  Kokkos::WithoutInitializing), k.np);
        k.vx = Arr<DevMem>(Kokkos::view_alloc("vx", Kokkos::WithoutInitializing), k.np);
        k.vy = Arr<DevMem>(Kokkos::view_alloc("vy", Kokkos::WithoutInitializing), k.np);
        k.vz = Arr<DevMem>(Kokkos::view_alloc("vz", Kokkos::WithoutInitializing), k.np);
        k.rho = Arr<DevMem>("rho", ncell_local_);
        k.ext = Arr<DevMem>(Kokkos::view_alloc("ext", Kokkos::WithoutInitializing), ext_size_);
        k.gx = Arr<DevMem>("gx", ncell_local_);
        k.gy = Arr<DevMem>("gy", ncell_local_);
        k.gz = Arr<DevMem>("gz", ncell_local_);
        k.sv_dev = SVDev(k.rho);
    } else {
        k.hrho = Arr<HostMem>("hrho", ncell_local_);
        if (k.atomic_cpu) k.sv_host_atomic = SVHostAtomic(k.hrho);
        else              k.sv_host_dup    = SVHostDup(k.hrho);
        k.hgx = Arr<HostMem>("hgx", ncell_local_);
        k.hgy = Arr<HostMem>("hgy", ncell_local_);
        k.hgz = Arr<HostMem>("hgz", ncell_local_);
    }

    if (world_rank_ == 0) {
        std::printf("Interpolater: Kokkos %s (PM_INTERP=%s, deposit=%s)\n",
                    k.gpu ? DevExec::name() : HostExec::name(), k.gpu ? "gpu" : "cpu",
                    k.gpu ? "atomic" : (k.atomic_cpu ? "atomic" : "duplicated"));
        std::fflush(stdout);
        DEBUG_LOG("Set Interpolater (Kokkos)");
    }
}

Interpolater::~Interpolater() = default;

void Interpolater::deposit() {
    KImpl& k = *kimpl_;
    HostWrap hx(particle_.x, k.np), hy(particle_.y, k.np), hz(particle_.z, k.np);
    HostWrap hrho(buf_, ncell_local_);

    if (k.gpu) {
        timer_.start(MPI_COMM_WORLD);
        Kokkos::deep_copy(k.x, hx);
        Kokkos::deep_copy(k.y, hy);
        Kokkos::deep_copy(k.z, hz);
        timer_.stop(t_copy_, MPI_COMM_WORLD);

        timer_.start(MPI_COMM_WORLD);
        Kokkos::deep_copy(DevExec(), k.rho, 0.0);
        k.sv_dev.reset();
        deposit_kernel<DevExec>(k.x, k.y, k.z, k.sv_dev, k.np, k.g);
        KE::contribute(k.rho, k.sv_dev);
        DevExec().fence();
        timer_.stop(t_deposit_, MPI_COMM_WORLD);

        timer_.start(MPI_COMM_WORLD);
        Kokkos::deep_copy(hrho, k.rho);
        timer_.stop(t_copy_, MPI_COMM_WORLD);
    } else {
        timer_.start(MPI_COMM_WORLD);
        Kokkos::deep_copy(HostExec(), k.hrho, 0.0);
        if (k.atomic_cpu) {
            k.sv_host_atomic.reset();
            deposit_kernel<HostExec>(hx, hy, hz, k.sv_host_atomic, k.np, k.g);
            KE::contribute(k.hrho, k.sv_host_atomic);
        } else {
            k.sv_host_dup.reset();    // スレッドごとの複製を 0 にする（従来版の memset に相当）
            deposit_kernel<HostExec>(hx, hy, hz, k.sv_host_dup, k.np, k.g);
            KE::contribute(k.hrho, k.sv_host_dup);   // 複製を合計（従来版の足し合わせに相当）
        }
        Kokkos::deep_copy(HostExec(), hrho, k.hrho);   // 共有バッファへ
        HostExec().fence();
        timer_.stop(t_deposit_, MPI_COMM_WORLD);
    }

    if (world_rank_ == 0) {
        DEBUG_LOG("Deposit");
    }
}

void Interpolater::gather(double a) {
    KImpl& k = *kimpl_;
    const double fa_da = function_a(a) * Da_;
    HostWrap hext(buf_, ext_size_);
    HostWrap hvx(particle_.vx, k.np), hvy(particle_.vy, k.np), hvz(particle_.vz, k.np);

    if (k.gpu) {
        timer_.start(MPI_COMM_WORLD);
        Kokkos::deep_copy(k.ext, hext);
        Kokkos::deep_copy(k.vx, hvx);
        Kokkos::deep_copy(k.vy, hvy);
        Kokkos::deep_copy(k.vz, hvz);
        timer_.stop(t_copy_, MPI_COMM_WORLD);

        // 粒子座標は deposit でコピーしたものを使う
        timer_.start(MPI_COMM_WORLD);
        gradient_kernel<DevExec>(k.ext, k.gx, k.gy, k.gz, k.g);
        gather_kernel<DevExec>(k.x, k.y, k.z, k.gx, k.gy, k.gz, k.vx, k.vy, k.vz, k.np, k.g, fa_da);
        DevExec().fence();
        timer_.stop(t_gather_, MPI_COMM_WORLD);

        timer_.start(MPI_COMM_WORLD);
        Kokkos::deep_copy(hvx, k.vx);
        Kokkos::deep_copy(hvy, k.vy);
        Kokkos::deep_copy(hvz, k.vz);
        timer_.stop(t_copy_, MPI_COMM_WORLD);
    } else {
        HostWrap hx(particle_.x, k.np), hy(particle_.y, k.np), hz(particle_.z, k.np);
        timer_.start(MPI_COMM_WORLD);
        gradient_kernel<HostExec>(hext, k.hgx, k.hgy, k.hgz, k.g);
        gather_kernel<HostExec>(hx, hy, hz, k.hgx, k.hgy, k.hgz, hvx, hvy, hvz, k.np, k.g, fa_da);
        HostExec().fence();
        timer_.stop(t_gather_, MPI_COMM_WORLD);
    }

    if (pm_checksum_enabled()) pm_print_velocity_checksum("GATHER", particle_, MPI_COMM_WORLD);
    if (world_rank_ == 0) {
        DEBUG_LOG("Gather");
    }
}
#endif  // PM_USE_KOKKOS

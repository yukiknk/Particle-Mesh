#pragma once
#include <cstring>
#include <cmath>
#include <cstdio>
#include <memory>
#include <omp.h>

#include "mpi_env.h"
#include "grid/grid.h"
#include "particle/particle.h"
#include "buffer_manager.h"
#include "timer.h"
#include "utils.h"
#include "fft/fft_checksum.h"

// Kokkos 版 Interpolater（-DPM_USE_KOKKOS でビルドしたとき）。
// Kokkos の初期化と終了。main で MPIEnv の直後に置き、MPIEnv より先に破棄されるようにする。
#ifdef PM_USE_KOKKOS
struct PmKokkosScope {
    PmKokkosScope(int& argc, char** argv);
    ~PmKokkosScope();
    PmKokkosScope(const PmKokkosScope&) = delete;
    PmKokkosScope& operator=(const PmKokkosScope&) = delete;
};
#endif

// gather 後の粒子速度の照合用（PM_CHECKSUM=1 のときだけ呼ぶ）。sum(vx^2+vy^2+vz^2) を全プロセスで合計して出す
inline void pm_print_velocity_checksum(const char* name, const Particle& pt, MPI_Comm comm) {
    double s = 0.0;
    for (size_t p = 0; p < pt.np; ++p) s += pt.vx[p] * pt.vx[p] + pt.vy[p] * pt.vy[p] + pt.vz[p] * pt.vz[p];
    double g = 0.0;
    MPI_Reduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::printf("[checksum] %-6s sum_v2=%.12e\n", name, g);
        std::fflush(stdout);
    }
}

inline __attribute__((always_inline)) double function_a(const double a) {
    double Omega_m = 0.3;
    double Omega_lambda = 0.7;
    double Omega_k = 1.0 - Omega_m - Omega_lambda;
    return 1.0 / sqrt(Omega_m / a + Omega_k + Omega_lambda * a * a);
}

class Interpolater {
public:
    Interpolater(const Grid& grid, Particle& particle, const MPIEnv& mpi, BufferManager& buffer, Timer& timer);
    ~Interpolater();
    
    Interpolater(const Interpolater&) = delete;
    Interpolater& operator=(const Interpolater&) = delete;
    Interpolater(Interpolater&&) = delete;
    Interpolater& operator=(Interpolater&&) = delete;

    void deposit();
    void gather(double a);

private:
    Particle& particle_;

    Timer& timer_;
    int t_deposit_;
    int t_gather_;
    
    int x0_, y0_, z0_;
    int nthreads_;
    double mass_;
    double Da_;
    
    size_t ny_ghost_;       // ny+1
    size_t nz_ghost_;       // nz+1
    size_t ncell_local_;    // (nx+1)*(ny+1)*(nz+1)
    size_t thread_buf_size_;

    int nx3_, ny3_, nz3_;   // 拡張グリッドサイズ (nx+3, ny+3, nz+3)
    
    double* buf_ = nullptr; // BufferManager 共有バッファ（offset 0）

    // ポテンシャル勾配バッファ (nx+1)*(ny+1)*(nz+1)
    double* potx_ = nullptr;
    double* poty_ = nullptr;
    double* potz_ = nullptr;
    size_t pot_size_;

    int world_rank_;

#ifdef PM_USE_KOKKOS
    // Kokkos 版の中身（Kokkos の型をこのヘッダに出さないため隠す）
    struct KImpl;
    std::unique_ptr<KImpl> kimpl_;
    int t_copy_ = -1;
    size_t ext_size_ = 0;   // gather が読む拡張グリッド (nx+3)*(ny+3)*(nz+3)
#endif
};

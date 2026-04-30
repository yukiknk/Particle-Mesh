#pragma once
#include <cstring>
#include <cmath>
#include <omp.h>

#include "mpi_env.h"
#include "grid/grid.h"
#include "particle/particle.h"
#include "buffer_manager.h"
#include "timer.h"
#include "utils.h"

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
};

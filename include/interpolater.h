#pragma once
#include <cstring>
#include <omp.h>

#include "mpi_env.h"
#include "grid/grid.h"
#include "particle/particle.h"
#include "buffer_manager.h"
#include "timer.h"
#include "utils.h"

class Interpolater {
public:
    Interpolater(const Grid& grid, Particle& particle, const MPIEnv& mpi, BufferManager& buffer, Timer& timer);
    
    Interpolater(const Interpolater&) = delete;
    Interpolater& operator=(const Interpolater&) = delete;
    Interpolater(Interpolater&&) = delete;
    Interpolater& operator=(Interpolater&&) = delete;

    void deposit();
    void gather();

private:
    Particle& particle_;

    Timer& timer_;
    int t_deposit_;
    
    int x0_, y0_, z0_;
    int nthreads_;
    double mass_;
    
    size_t ny_ghost_;
    size_t nz_ghost_;
    size_t ncell_local_;
    size_t thread_buf_size_;
    
    double* buf_ = nullptr;
};

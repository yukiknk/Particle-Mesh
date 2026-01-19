#pragma once
#include <random>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

#include "mpi_env.hpp"
#include "grid/grid.hpp"

struct Particle {
    int np;
    int np_total;
    
    double* x = nullptr;
    double* y = nullptr;
    double* z = nullptr;
    double* vx = nullptr;
    double* vy = nullptr;
    double* vz = nullptr;
    
    Particle(int np_total_, const Grid& grid, const MPIEnv& mpi) 
        : np_total(np_total_), np(np_total_ / mpi.world_size())
    {
        size_t bytes = ((np * sizeof(double) + 63) / 64) * 64;
        
        x  = static_cast<double*>(std::aligned_alloc(64, bytes));
        y  = static_cast<double*>(std::aligned_alloc(64, bytes));
        z  = static_cast<double*>(std::aligned_alloc(64, bytes));
        vx = static_cast<double*>(std::aligned_alloc(64, bytes));
        vy = static_cast<double*>(std::aligned_alloc(64, bytes));
        vz = static_cast<double*>(std::aligned_alloc(64, bytes));
        
        if (!x || !y || !z || !vx || !vy || !vz) {
            std::cerr << "Particle: allocation failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        std::mt19937 rng(42 + mpi.world_rank());
        std::uniform_real_distribution<double> dist_x(static_cast<double>(grid.x0), static_cast<double>(grid.x0 + grid.nx));
        std::uniform_real_distribution<double> dist_y(static_cast<double>(grid.y0), static_cast<double>(grid.y0 + grid.ny));
        std::uniform_real_distribution<double> dist_z(static_cast<double>(grid.z0), static_cast<double>(grid.z0 + grid.nz));
        
        for (int i = 0; i < np; ++i) {
            x[i] = dist_x(rng);
            y[i] = dist_y(rng);
            z[i] = dist_z(rng);
            vx[i] = 0.0;
            vy[i] = 0.0;
            vz[i] = 0.0;
        }
    }
    
    ~Particle() {
        std::free(x);
        std::free(y);
        std::free(z);
        std::free(vx);
        std::free(vy);
        std::free(vz);
    }
    
    Particle(const Particle&) = delete;
    Particle& operator=(const Particle&) = delete;
    Particle(Particle&&) = delete;
    Particle& operator=(Particle&&) = delete;
};

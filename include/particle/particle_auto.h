#pragma once
#include <random>
#include <mpi.h>
#include "debug.h"
#include "mpi_env.h"
#include "grid/grid.h"
#include "particle.h"

struct ParticleAuto : public Particle {
    ParticleAuto(size_t np_total_, const Grid& grid, const MPIEnv& mpi)
        : Particle(np_total_, local_np(np_total_, mpi.world_rank(), mpi.world_size()))   // 基底が確保
    {
        std::mt19937 rng(42 + mpi.world_rank());
        std::uniform_real_distribution<double> dist_x(
            static_cast<double>(grid.x0), static_cast<double>(grid.x0 + grid.nx));
        std::uniform_real_distribution<double> dist_y(
            static_cast<double>(grid.y0), static_cast<double>(grid.y0 + grid.ny));
        std::uniform_real_distribution<double> dist_z(
            static_cast<double>(grid.z0), static_cast<double>(grid.z0 + grid.nz));

        for (size_t i = 0; i < np; ++i) {
            x[i]  = dist_x(rng);
            y[i]  = dist_y(rng);
            z[i]  = dist_z(rng);
            vx[i] = 0.0;
            vy[i] = 0.0;
            vz[i] = 0.0;
        }

        if (mpi.world_rank() == 0) {
            DEBUG_LOG("Set Particle");
        }
    }

    private:
    static size_t local_np(size_t np_total, int rank, int size) {
        return np_total * static_cast<size_t>(rank + 1) / static_cast<size_t>(size)
             - np_total * static_cast<size_t>(rank)     / static_cast<size_t>(size);
    }
};

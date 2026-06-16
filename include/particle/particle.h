#pragma once
#include <cstdlib>
#include <iostream>
#include <mpi.h>

struct Particle {
    size_t np;
    size_t np_total;

    double* x  = nullptr;
    double* y  = nullptr;
    double* z  = nullptr;
    double* vx = nullptr;
    double* vy = nullptr;
    double* vz = nullptr;

    ~Particle() {
        std::free(x);
        std::free(y);
        std::free(z);
        std::free(vx);
        std::free(vy);
        std::free(vz);
    }

    Particle(const Particle&)            = delete;
    Particle& operator=(const Particle&) = delete;
    Particle(Particle&&)                 = delete;
    Particle& operator=(Particle&&)      = delete;

protected:
    // 派生クラスがまず呼ぶ：np を確定して 6 配列を確保
    Particle(size_t np_total_, size_t np_)
        : np(np_), np_total(np_total_)
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
    }
};

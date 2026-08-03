#pragma once
#include <random>
#include <mpi.h>
#include <iostream>
#include "debug.h"
#include "mpi_env.h"
#include "grid/grid.h"
#include "grid/grid_input.h"
#include "particle.h"

struct ParticleInput : public Particle {
    // GridInput からファイル分割・サブ分割情報を受け取る
    ParticleInput(const GridInput& grid, size_t Np3, const MPIEnv& mpi)
        : ParticleInput(compute_np(grid, Np3, mpi), grid, mpi)
    {}

private:
    struct Plan { size_t np; };

    // ファイル粒子総数（各ファイルを 1 回ずつ集計）から倍率を出し、
    // 担当領域の粒子数 × 倍率 を sub プロセスで分配
    static Plan compute_np(const GridInput& grid, size_t Np3, const MPIEnv& mpi) {
        // 各ファイルを 1 回だけ数える：領域内サブインデックスが原点の rank のみ寄与
        int si0 = grid.coords[0] % grid.sub[0];
        int si1 = grid.coords[1] % grid.sub[1];
        int si2 = grid.coords[2] % grid.sub[2];
        unsigned long long contrib =
            (si0 == 0 && si1 == 0 && si2 == 0)
            ? static_cast<unsigned long long>(grid.npart_file) : 0ULL;

        unsigned long long file_total = 0;
        MPI_Allreduce(&contrib, &file_total, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);

        if (file_total == 0) {
            if (mpi.world_rank() == 0) {
                std::cerr << "ParticleInput: file_total is 0" << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 22);
        }
        
        // 倍率 = 要求総粒子数 / ファイル粒子総数
        unsigned long long mult =
            static_cast<unsigned long long>(Np3) / file_total;

        if (mult == 0 || static_cast<unsigned long long>(Np3) % file_total != 0) {
            if (mpi.world_rank() == 0) {
                std::cerr << "ParticleInput: Np^3(" << Np3
                          << ") must be a positive multiple of file particle total("
                          << file_total << ")" << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 23);
        }

        // この領域の総粒子数 = npart_file * mult
        unsigned long long region_np =
            static_cast<unsigned long long>(grid.npart_file) * mult;

        // 領域を担当する sub プロセス数で分配する。
        // sub が非2べき（例 3x2x2 = 12）だと割り切れないため、
        // サブインデックスを線形化して端数を先頭から 1 個ずつ配る。
        int nsub = grid.sub[0] * grid.sub[1] * grid.sub[2];
        int si_lin = (si0 * grid.sub[1] + si1) * grid.sub[2] + si2;

        unsigned long long lo = region_np * static_cast<unsigned long long>(si_lin)
                              / static_cast<unsigned long long>(nsub);
        unsigned long long hi = region_np * static_cast<unsigned long long>(si_lin + 1)
                              / static_cast<unsigned long long>(nsub);

        return Plan{ static_cast<size_t>(hi - lo) };
    }

    ParticleInput(Plan plan, const GridInput& grid, const MPIEnv& mpi)
        : Particle(plan.np, plan.np)   // np_total は後で全体値に上書き
    {
        // 担当グリッド範囲内に一様乱数で生成
        std::mt19937 rng(12345 + mpi.world_rank());
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

        // np_total = 全 rank の np の総和
        unsigned long long local = static_cast<unsigned long long>(np);
        unsigned long long global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        np_total = static_cast<size_t>(global);

        if (mpi.world_rank() == 0) {
            DEBUG_LOG("Set Particle (input)");
        }
    }
};

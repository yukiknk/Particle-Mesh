#pragma once
#include <random>
#include <mpi.h>
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
    // 担当領域の粒子数 × 倍率 を sub プロセスで均等割り
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

        // 倍率 = 要求総粒子数 / ファイル粒子総数（割り切れる前提）
        unsigned long long mult =
            static_cast<unsigned long long>(Np3) / file_total;

        // この領域の総粒子数 = npart_file * mult
        unsigned long long region_np =
            static_cast<unsigned long long>(grid.npart_file) * mult;

        // 領域を担当する sub プロセス数で均等割り（割り切れる前提）
        int nsub = grid.sub[0] * grid.sub[1] * grid.sub[2];
        size_t np = static_cast<size_t>(region_np / nsub);

        return Plan{np};
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

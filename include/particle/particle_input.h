#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <mpi.h>
#include "debug.h"
#include "mpi_env.h"
#include "grid/grid.h"
#include "particle.h"

struct ParticleInput : public Particle {
    ParticleInput(const std::string& root, int Ng, const Grid& grid, const MPIEnv& mpi)
        : ParticleInput(open_and_read_header(root, grid), Ng, mpi)   // ヘッダ先読み→委譲
    {}

private:
    // ヘッダ読み出しの中間結果
    struct Header {
        std::FILE* fp;
        int npart;
        float hubble;
    };

    // fid を決めてファイルを開き、ヘッダを読んで npart/hubble を返す（fp は開いたまま）
    static Header open_and_read_header(const std::string& root, const Grid& grid) {
        int idx = (grid.coords[0] * grid.dims[1] + grid.coords[1]) * grid.dims[2] + grid.coords[2];
        int group = idx / 1024;
        char subdir[8];
        std::snprintf(subdir, sizeof(subdir), "%04d", group);
        std::string sep = (root.empty() || root.back() == '/') ? "" : "/";
        std::string path = root + sep + subdir + "/snap_050." + std::to_string(idx);

        std::FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::cerr << "ParticleInput: cannot open " << path << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 21);
        }

        int io_ver = 0, npart = 0, ngas = 0;
        float omega0, omegab, lambda0, hubble, astart, anow, tnow;
        double lunit, munit, tunit;
        std::fread(&io_ver,  sizeof(int),    1, fp);
        std::fread(&npart,   sizeof(int),    1, fp);
        std::fread(&ngas,    sizeof(int),    1, fp);
        std::fread(&omega0,  sizeof(float),  1, fp);
        std::fread(&omegab,  sizeof(float),  1, fp);
        std::fread(&lambda0, sizeof(float),  1, fp);
        std::fread(&hubble,  sizeof(float),  1, fp);
        std::fread(&astart,  sizeof(float),  1, fp);
        std::fread(&anow,    sizeof(float),  1, fp);
        std::fread(&tnow,    sizeof(float),  1, fp);
        std::fread(&lunit,   sizeof(double), 1, fp);
        std::fread(&munit,   sizeof(double), 1, fp);
        std::fread(&tunit,   sizeof(double), 1, fp);

        return Header{fp, npart, hubble};
    }

    // 委譲先：基底が npart 個ぶん確保 → ボディを読んで埋める
    ParticleInput(Header h, int Ng, const MPIEnv& mpi)
        : Particle(static_cast<size_t>(h.npart), static_cast<size_t>(h.npart))   // np_total は後で上書き
    {
        const size_t n = static_cast<size_t>(h.npart);
        std::vector<float> cache_r(3 * n);
        std::vector<float> cache_v(3 * n);
        std::fread(cache_r.data(), sizeof(float), cache_r.size(), h.fp);
        std::fread(cache_v.data(), sizeof(float), cache_v.size(), h.fp);
        // ID ブロックは読まない
        std::fclose(h.fp);

        for (size_t i = 0; i < np; ++i) {
            x[i]  = static_cast<double>(cache_r[3 * i + 0]) * Ng;
            y[i]  = static_cast<double>(cache_r[3 * i + 1]) * Ng;
            z[i]  = static_cast<double>(cache_r[3 * i + 2]) * Ng;
            vx[i] = static_cast<double>(cache_v[3 * i + 0]) / h.hubble;
            vy[i] = static_cast<double>(cache_v[3 * i + 1]) / h.hubble;
            vz[i] = static_cast<double>(cache_v[3 * i + 2]) / h.hubble;
        }

        // np_total = 全ランクの np の総和
        unsigned long long local = static_cast<unsigned long long>(np);
        unsigned long long global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        np_total = static_cast<size_t>(global);

        if (mpi.world_rank() == 0) {
            DEBUG_LOG("Set Particle (input)");
        }
    }
};

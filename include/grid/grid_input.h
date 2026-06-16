#pragma once
#include <mpi.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "grid.h"
#include "mpi_env.h"

struct GridInput : public Grid {
    GridInput(const std::string& root, const MPIEnv& mpi, int Ng_) {
        load(root, mpi, Ng_);
    }

private:
    void load(const std::string& root, const MPIEnv& mpi, int Ng_) {
        Ng = Ng_;

        int idump[4];
        std::vector<double> bmin_all, bmax_all;

        // rank0 が読んで Bcast
        if (mpi.world_rank() == 0) {
            std::string sep = (root.empty() || root.back() == '/') ? "" : "/";
            std::string filename = root + sep + "0000/snap_050.boundary";
            std::FILE* fin = std::fopen(filename.c_str(), "rb");
            if (!fin) {
                std::fprintf(stderr, "GridInput: cannot open %s\n", filename.c_str());
                MPI_Abort(MPI_COMM_WORLD, 11);
            }
            std::fread(idump, sizeof(int), 4, fin);
            bmin_all.resize(3 * idump[0]);
            bmax_all.resize(3 * idump[0]);
            std::fread(bmin_all.data(), sizeof(double), 3 * idump[0], fin);
            std::fread(bmax_all.data(), sizeof(double), 3 * idump[0], fin);
            std::fclose(fin);
        }
        MPI_Bcast(idump, 4, MPI_INT, 0, MPI_COMM_WORLD);

        if (mpi.world_rank() != 0) {
            bmin_all.resize(3 * idump[0]);
            bmax_all.resize(3 * idump[0]);
        }
        MPI_Bcast(bmin_all.data(), 3 * idump[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(bmax_all.data(), 3 * idump[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        dims[0] = idump[1];
        dims[1] = idump[2];
        dims[2] = idump[3];

        // coords を rank から算出（元の Grid と同じ規約）
        int rank = mpi.world_rank();
        coords[0] = rank / (dims[1] * dims[2]);
        coords[1] = (rank / dims[2]) % dims[1];
        coords[2] = rank % dims[2];

        // 自領域のインデックス（floor / floor+1）
        int idx = (coords[0] * dims[1] + coords[1]) * dims[2] + coords[2];
        x0 = (int)(bmin_all[3 * idx + 0] * Ng);
        y0 = (int)(bmin_all[3 * idx + 1] * Ng);
        z0 = (int)(bmin_all[3 * idx + 2] * Ng);
        int x1 = (int)(bmax_all[3 * idx + 0] * Ng) + 1;
        int y1 = (int)(bmax_all[3 * idx + 1] * Ng) + 1;
        int z1 = (int)(bmax_all[3 * idx + 2] * Ng) + 1;
        nx = x1 - x0;
        ny = y1 - y0;
        nz = z1 - z0;
        n_local = nx * ny * nz;
    }
};

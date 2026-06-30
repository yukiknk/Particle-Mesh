#pragma once
#include <mpi.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include "grid.h"
#include "mpi_env.h"

struct GridInput : public Grid {
    // ファイル分割（base）に関する情報も保持（ParticleInput が使う）
    int base_dims[3];   // boundary が要求する分割数
    int sub[3];         // 各軸の細分数 eff/base
    int file_idx;       // 担当するファイル領域の ID
    int npart_file;     // そのファイル領域の粒子数（ヘッダ値）

    GridInput(const std::string& root, const MPIEnv& mpi, int Ng_) {
        load(root, mpi, Ng_);
    }

private:
    // base_dims から最小軸を 2 倍する操作を繰り返して eff_dims を作る
    // （16,16,16 -> 32,16,16 -> 32,32,16 -> 32,32,32 -> ... の順）
    static void make_eff_dims(const int base[3], int nproc, int eff[3]) {
        eff[0] = base[0]; eff[1] = base[1]; eff[2] = base[2];
        long long prod = (long long)eff[0] * eff[1] * eff[2];
        while (prod < nproc) {
            // 最小の軸を選ぶ（同点は x->y->z の手前）
            int axis = 0;
            if (eff[1] < eff[axis]) axis = 1;
            if (eff[2] < eff[axis]) axis = 2;
            eff[axis] *= 2;
            prod *= 2;
        }
    }

    void load(const std::string& root, const MPIEnv& mpi, int Ng_) {
        Ng = Ng_;

        int idump[4];
        std::vector<double> bmin_all, bmax_all;

        // rank0 が boundary を読んで Bcast
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

        base_dims[0] = idump[1];
        base_dims[1] = idump[2];
        base_dims[2] = idump[3];
        int base_prod = base_dims[0] * base_dims[1] * base_dims[2];

        int nproc = mpi.world_size();
        if (nproc < base_prod || (nproc % base_prod) != 0) {
            if (mpi.world_rank() == 0) {
                std::fprintf(stderr,
                    "GridInput: nproc(%d) must be a multiple of file split (%d=%dx%dx%d)\n",
                    nproc, base_prod, base_dims[0], base_dims[1], base_dims[2]);
            }
            MPI_Abort(MPI_COMM_WORLD, 14);
        }

        // eff_dims（実効分割）と各軸の細分数
        int eff[3];
        make_eff_dims(base_dims, nproc, eff);
        dims[0] = eff[0]; dims[1] = eff[1]; dims[2] = eff[2];
        sub[0] = eff[0] / base_dims[0];
        sub[1] = eff[1] / base_dims[1];
        sub[2] = eff[2] / base_dims[2];

        // eff_dims 上での自分の座標（z 最内 row-major、Cart は使わず手計算）
        int rank = mpi.world_rank();
        coords[0] = rank / (eff[1] * eff[2]);
        coords[1] = (rank / eff[2]) % eff[1];
        coords[2] = rank % eff[2];

        // ファイル分割座標 = eff座標 / 細分数
        int fc[3] = { coords[0] / sub[0], coords[1] / sub[1], coords[2] / sub[2] };
        // 領域内サブインデックス
        int si[3] = { coords[0] % sub[0], coords[1] % sub[1], coords[2] % sub[2] };

        file_idx = (fc[0] * base_dims[1] + fc[1]) * base_dims[2] + fc[2];

        // ファイル領域のグリッド範囲（floor / floor+1）
        double dx = 1.0 / Ng;   // L=1.0
        int X0 = (int)(bmin_all[3 * file_idx + 0] * Ng);
        int Y0 = (int)(bmin_all[3 * file_idx + 1] * Ng);
        int Z0 = (int)(bmin_all[3 * file_idx + 2] * Ng);
        int X1 = (int)(bmax_all[3 * file_idx + 0] * Ng) + 1;
        int Y1 = (int)(bmax_all[3 * file_idx + 1] * Ng) + 1;
        int Z1 = (int)(bmax_all[3 * file_idx + 2] * Ng) + 1;
        (void)dx;

        // 領域を sub で均等分割し、si 番目のサブ範囲を担当
        x0 = X0 + (long long)(X1 - X0) * si[0] / sub[0];
        int x0n = X0 + (long long)(X1 - X0) * (si[0] + 1) / sub[0];
        y0 = Y0 + (long long)(Y1 - Y0) * si[1] / sub[1];
        int y0n = Y0 + (long long)(Y1 - Y0) * (si[1] + 1) / sub[1];
        z0 = Z0 + (long long)(Z1 - Z0) * si[2] / sub[2];
        int z0n = Z0 + (long long)(Z1 - Z0) * (si[2] + 1) / sub[2];
        nx = x0n - x0;
        ny = y0n - y0;
        nz = z0n - z0;
        n_local = nx * ny * nz;

        // 担当ファイルのヘッダから粒子数を読む（座標は読まない）
        npart_file = read_npart(root, file_idx, mpi);
    }

    static int read_npart(const std::string& root, int idx, const MPIEnv& mpi) {
        int group = idx / 1024;
        char subdir[8];
        std::snprintf(subdir, sizeof(subdir), "%04d", group);
        std::string sep = (root.empty() || root.back() == '/') ? "" : "/";
        std::string path = root + sep + subdir + "/snap_050." + std::to_string(idx);

        std::FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::cerr << "GridInput: cannot open " << path << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 21);
        }
        int header[3]; // io_ver, npart, ngas
        std::fread(header, sizeof(int), 3, fp);
        std::fclose(fp);
        return header[1];
    }
};

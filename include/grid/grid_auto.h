// grid_auto.hpp
#pragma once
#include <mpi.h>

struct Grid {
    int Ng;
    int dims[3];
    int coords[3];
    int x0, y0, z0;
    int nx, ny, nz;
    int n_local;
    
    Grid(int Ng_, int rank, int nprocs) 
        : Ng(Ng_) 
    {
        dims[0] = dims[1] = dims[2] = 0;
        MPI_Dims_create(nprocs, 3, dims);
        
        coords[0] = rank / (dims[1] * dims[2]);
        coords[1] = (rank / dims[2]) % dims[1];
        coords[2] = rank % dims[2];
        
        nx = Ng / dims[0];
        ny = Ng / dims[1];
        nz = Ng / dims[2];
        
        x0 = coords[0] * nx;
        y0 = coords[1] * ny;
        z0 = coords[2] * nz;
        
        n_local = nx * ny * nz;
    }
};
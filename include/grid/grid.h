#pragma once

struct Grid {
    int Ng;
    int dims[3];
    int coords[3];
    int x0, y0, z0;
    int nx, ny, nz;
    int n_local;

protected:
    Grid() = default;
};

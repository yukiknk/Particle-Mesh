#include "utils_inline.h"

double getTime(double* now_time);
void report_max_time(double local_sec, MPI_Comm comm, const int i, const int worm_up, double& sum_time);
void init_particles(double * __restrict posx, double * __restrict posy, double * __restrict posz, size_t Np_local, int x0, int y0, int z0, int nx_loc, int ny_loc, int nz_loc, int rank);
void init_buf(double* __restrict buf, int nx,  int ny,  int nz, size_t ncell_local, int nthreads);
void deposit_particles(const double * __restrict posx, const double * __restrict posy, const double * __restrict posz, size_t Np_local, double* __restrict buf, size_t ncell_local, int nx, int ny, int nz, int x0, int y0, int z0, double mass);
void reduce_private_to_shared(double* __restrict buf, size_t ncell_local, int nthreads);
void centered_derivatives(const double* __restrict in, double* __restrict d_dx, double* __restrict d_dy, double* __restrict d_dz, int nx1, int ny1, int nz1);
void update_particles(const double * __restrict posx, const double * __restrict posy, const double * __restrict posz, const double* __restrict d_dx, const double* __restrict d_dy, const double* __restrict d_dz, double * __restrict px, double * __restrict py, double * __restrict pz, const double a, const size_t Np_local, const int ny, const int nz, const int x0, const int y0, const int z0);

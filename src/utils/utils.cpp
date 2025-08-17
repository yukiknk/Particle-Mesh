#include "utils.h"

double getTime(double* now_time){
    struct timespec tv;
    clock_gettime(CLOCK_REALTIME, &tv);

    double told = *now_time;
    *now_time = (tv.tv_sec + (double)tv.tv_nsec*1e-9);
    return *now_time - told;
}

void report_max_time(double local_sec, MPI_Comm comm, const int i, const int worm_up, double& sum_time){
    if(i < worm_up) return;
    
    double global_max = 0.0;
    MPI_Reduce(&local_sec, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    int rank; MPI_Comm_rank(comm, &rank);
    if (rank == 0) sum_time += global_max;
    MPI_Barrier(comm);
}

void init_particles(double * __restrict posx, double * __restrict posy, double * __restrict posz, size_t Np_local, int x0, int y0, int z0, int nx_loc, int ny_loc, int nz_loc, int rank){
    std::mt19937_64 gen_x(static_cast<uint64_t>(rank) * 3ULL);
    std::mt19937_64 gen_y(static_cast<uint64_t>(rank) * 3ULL + 1);
    std::mt19937_64 gen_z(static_cast<uint64_t>(rank) * 3ULL + 2);

    std::uniform_real_distribution<double> rx(static_cast<double>(x0), static_cast<double>(x0 + nx_loc));
    std::uniform_real_distribution<double> ry(static_cast<double>(y0), static_cast<double>(y0 + ny_loc));
    std::uniform_real_distribution<double> rz(static_cast<double>(z0), static_cast<double>(z0 + nz_loc));

    for (size_t p = 0; p < Np_local; ++p) {
        posx[p] = rx(gen_x);
        posy[p] = ry(gen_y);
        posz[p] = rz(gen_z);
    }
}

void init_buf(double* __restrict buf, int nx,  int ny,  int nz, size_t ncell_local, int nthreads){
    const size_t tail = ncell_local * (size_t)(nthreads - 1);
    const int nxm1 = nx - 1, nym1 = ny - 1, nzm1 = nz - 1;

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static) nowait
        for (int ix = 0; ix < nxm1; ++ix) {
            for (int iy = 0; iy < nym1; ++iy) {
                size_t base = ((size_t)ix * ny + (size_t)iy) * (size_t)nz;
                std::fill_n(buf + base, (size_t)nzm1, -1.0);
                buf[base + (size_t)nzm1] = 0.0;
            }
        }

        #pragma omp for schedule(static) nowait
        for (int iy = 0; iy < ny; ++iy) {
            size_t base = ((size_t)nxm1 * ny + (size_t)iy) * (size_t)nz;
            std::fill_n(buf + base, (size_t)nz, 0.0);
        }

        #pragma omp for schedule(static) nowait
        for (int ix = 0; ix < nxm1; ++ix) {
            size_t base = ((size_t)ix * ny + (size_t)nym1) * (size_t)nz;
            std::fill_n(buf + base, (size_t)nz, 0.0);
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < tail; ++i) buf[ncell_local + i] = 0.0;
    }
}

void deposit_particles(const double * __restrict posx, const double * __restrict posy, const double * __restrict posz, size_t Np_local, double* __restrict buf, size_t ncell_local, int nx, int ny, int nz, int x0, int y0, int z0, double mass){
    const size_t yz_stride = static_cast<size_t>(nz);
    const size_t x_stride  = static_cast<size_t>(ny) * nz;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* __restrict acc = buf + static_cast<size_t>(tid) * ncell_local;

        #pragma omp for schedule(static)
        for (size_t p = 0; p < Np_local; ++p){
            const double xp = posx[p];
            const double yp = posy[p];
            const double zp = posz[p];

            const int ix = static_cast<int>(xp);
            const int iy = static_cast<int>(yp);
            const int iz = static_cast<int>(zp);

            const int i0 = ix - x0;
            const int j0 = iy - y0;
            const int k0 = iz - z0;

            const size_t base = (static_cast<size_t>(i0) * ny + j0) * nz + k0;

            const double tx = xp - static_cast<double>(ix);
            const double ty = yp - static_cast<double>(iy);
            const double tz = zp - static_cast<double>(iz);

            cic_deposit(acc, base, yz_stride, x_stride, tx, ty, tz, mass);
        }
    }
}

void reduce_private_to_shared(double* __restrict buf, size_t ncell_local, int nthreads){
    for (int t = 1; t < nthreads; ++t){
        double* __restrict dst = buf;
        double* __restrict src = buf + (size_t)t * ncell_local;
        #pragma omp parallel for schedule(static)
        for (size_t c = 0; c < ncell_local; ++c){
            dst[c] += src[c];
        }
    }
}

void centered_derivatives(const double* __restrict in, double* __restrict d_dx, double* __restrict d_dy, double* __restrict d_dz, int nx1, int ny1, int nz1){
    int ny = ny1 - 2;
    int nz = nz1 - 2;
    int yz1 = ny1 * nz1;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 1; i < nx1-1; ++i)for(int j = 1; j < ny1-1; ++j){
        const double* __restrict row_i   = in + i      * yz1 + j      * nz1;
        const double* __restrict row_im1 = in + (i-1)  * yz1 + j      * nz1;
        const double* __restrict row_ip1 = in + (i+1)  * yz1 + j      * nz1;
        const double* __restrict row_jm1 = in + i      * yz1 + (j-1)  * nz1;
        const double* __restrict row_jp1 = in + i      * yz1 + (j+1)  * nz1;

        double* __restrict dx = d_dx + ((i - 1) * ny + (j - 1)) * nz;
        double* __restrict dy = d_dy + ((i - 1) * ny + (j - 1)) * nz;
        double* __restrict dz = d_dz + ((i - 1) * ny + (j - 1)) * nz;
        for (int kk = 0; kk < nz; ++kk) {
            const int k = kk + 1;
            dx[kk] = (row_im1[k]   - row_ip1[k])   * 0.5;
            dy[kk] = (row_jm1[k]   - row_jp1[k])   * 0.5;
            dz[kk] = (row_i  [k-1] - row_i  [k+1]) * 0.5;
        }
    }
}



void update_particles(const double * __restrict posx, const double * __restrict posy, const double * __restrict posz, const double* __restrict d_dx, const double* __restrict d_dy, const double* __restrict d_dz, double * __restrict px, double * __restrict py, double * __restrict pz, const double a, const size_t Np_local, const int ny, const int nz, const int x0, const int y0, const int z0){
    const double fa_da = function_a(a) * Da;
    const int xs = ny * nz;
    const int ys = nz;
    //double a_half_da = a + Da / 2;
    //double pow_a_fa = pow(a_half_da, -2) * function_a(a_half_da) * Da;
    #pragma omp parallel for schedule(static) 
    for(size_t p = 0; p < Np_local; ++p){ 
        const double xp = posx[p];
        const double yp = posy[p];
        const double zp = posz[p]; 

        const int ix = static_cast<int>(xp);
        const int iy = static_cast<int>(yp);
        const int iz = static_cast<int>(zp); 

        const int i0 = ix - x0, j0 = iy - y0, k0 = iz - z0; 
        const double tx = xp - ix, ty = yp - iy, tz = zp - iz;

        const size_t b = (static_cast<size_t>(i0) * ny + j0) * nz + k0;

        const double gx = trilerp1(d_dx, b, xs, ys, tx, ty, tz);
        const double gy = trilerp1(d_dy, b, xs, ys, tx, ty, tz);
        const double gz = trilerp1(d_dz, b, xs, ys, tx, ty, tz);

        px[p] += gx * fa_da;
        py[p] += gy * fa_da;
        pz[p] += gz * fa_da;
    }
}

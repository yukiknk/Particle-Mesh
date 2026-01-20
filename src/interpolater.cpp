#include "interpolater.h"

Interpolater::Interpolater(const Grid& grid, Particle& particle, const MPIEnv& mpi, BufferManager& buffer)
    : particle_(particle),
      x0_(grid.x0), y0_(grid.y0), z0_(grid.z0),
      nthreads_(mpi.nthreads()),
      mass_(1.0),
      ny_ghost_(static_cast<size_t>(grid.ny + 1)),
      nz_ghost_(static_cast<size_t>(grid.nz + 1)),
      ncell_local_(static_cast<size_t>(grid.nx + 1) * (grid.ny + 1) * (grid.nz + 1))
{
    thread_buf_size_ = align_to_64(ncell_local_);
    size_t total_size = thread_buf_size_ * nthreads_;
    
    buffer.register_buffer(buf_, 0);
    buffer.update_max_size(total_size);
}

void Interpolater::deposit() {
    const int np = particle_.np;
    const double* __restrict px = particle_.x;
    const double* __restrict py = particle_.y;
    const double* __restrict pz = particle_.z;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* __restrict acc = buf_ + static_cast<size_t>(tid) * thread_buf_size_;
        std::memset(acc, 0, sizeof(double) * ncell_local_);

        #pragma omp for schedule(static)
        for (int p = 0; p < np; ++p) {
            const double xp = px[p];
            const double yp = py[p];
            const double zp = pz[p];

            const int ix = static_cast<int>(xp);
            const int iy = static_cast<int>(yp);
            const int iz = static_cast<int>(zp);

            const double wx1 = xp - static_cast<double>(ix);
            const double wy1 = yp - static_cast<double>(iy);
            const double wz1 = zp - static_cast<double>(iz);
            const double wx0 = 1.0 - wx1;
            const double wy0 = 1.0 - wy1;
            const double wz0 = 1.0 - wz1;

            const int li = ix - x0_;
            const int li1 = li + 1;
            const int lj = iy - y0_;
            const int lj1 = lj + 1;
            const int lk = iz - z0_;
            const int lk1 = lk + 1;

            const size_t lij = (static_cast<size_t>(li) * ny_ghost_ + lj) * nz_ghost_;
            const size_t lij1 = (static_cast<size_t>(li) * ny_ghost_ + lj1) * nz_ghost_;
            const size_t li1j = (static_cast<size_t>(li1) * ny_ghost_ + lj) * nz_ghost_;
            const size_t li1j1 = (static_cast<size_t>(li1) * ny_ghost_ + lj1) * nz_ghost_;

            const double wx0y0 = wx0 * wy0;
            const double wx0y1 = wx0 * wy1;
            const double wx1y0 = wx1 * wy0;
            const double wx1y1 = wx1 * wy1;
            const double wz0m = wz0 * mass_;
            const double wz1m = wz1 * mass_;

            acc[lij + lk] += wx0y0 * wz0m;
            acc[lij + lk1] += wx0y0 * wz1m;
            acc[lij1 + lk] += wx0y1 * wz0m;
            acc[lij1 + lk1] += wx0y1 * wz1m;
            acc[li1j + lk] += wx1y0 * wz0m;
            acc[li1j + lk1] += wx1y0 * wz1m;
            acc[li1j1 + lk] += wx1y1 * wz0m;
            acc[li1j1 + lk1] += wx1y1 * wz1m;
        }

        for (int t = 1; t < nthreads_; ++t) {
            double* __restrict src = buf_ + static_cast<size_t>(t) * thread_buf_size_;
            #pragma omp for schedule(static)
            for (size_t idx = 0; idx < ncell_local_; ++idx) {
                buf_[idx] += src[idx];
            }
        }
    }
}

void Interpolater::gather() {
    // TODO: 後で実装
}

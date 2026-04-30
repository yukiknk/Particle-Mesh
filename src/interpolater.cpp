#include "interpolater.h"
#include "debug.h"

Interpolater::Interpolater(const Grid& grid, Particle& particle, const MPIEnv& mpi, BufferManager& buffer, Timer& timer)
    : timer_(timer),
      world_rank_(mpi.world_rank()),
      particle_(particle),
      x0_(grid.x0), y0_(grid.y0), z0_(grid.z0),
      nthreads_(mpi.nthreads()),
      mass_(1.0),
      Da_(0.001),
      ny_ghost_(static_cast<size_t>(grid.ny + 1)),
      nz_ghost_(static_cast<size_t>(grid.nz + 1)),
      ncell_local_(static_cast<size_t>(grid.nx + 1) * (grid.ny + 1) * (grid.nz + 1)),
      nx3_(grid.nx + 3), ny3_(grid.ny + 3), nz3_(grid.nz + 3)
{
    //タイマー登録
    t_deposit_ = timer_.register_timer("Interpolater Deposit");
    t_gather_ = timer_.register_timer("Interpolater Gather");
    
    thread_buf_size_ = align_to_64(ncell_local_);
    size_t total_size = thread_buf_size_ * nthreads_;
    
    buffer.register_buffer(buf_, 0);
    buffer.update_max_size(total_size);

    // ポテンシャル勾配バッファ: (nx+1)*(ny+1)*(nz+1)
    pot_size_ = align_to_64(ncell_local_);
    const size_t ALIGN = 64;
    size_t bytes = pot_size_ * sizeof(double);
    potx_ = static_cast<double*>(std::aligned_alloc(ALIGN, bytes));
    poty_ = static_cast<double*>(std::aligned_alloc(ALIGN, bytes));
    potz_ = static_cast<double*>(std::aligned_alloc(ALIGN, bytes));

    if (!potx_ || !poty_ || !potz_) {
        std::cerr << "Interpolater: gradient buffer allocation failed" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if(world_rank_ == 0) {
        DEBUG_LOG("Set Interpolater");
    }
}

Interpolater::~Interpolater() {
    std::free(potx_);
    std::free(poty_);
    std::free(potz_);
}

void Interpolater::deposit() {
    timer_.start(MPI_COMM_WORLD);
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
    timer_.stop(t_deposit_, MPI_COMM_WORLD);
    if(world_rank_ == 0) {
        DEBUG_LOG("Deposit");
    }
}

void Interpolater::gather(double a) {
    timer_.start(MPI_COMM_WORLD);

    // buf_ は BufferManager 共有バッファ（offset 0）
    // TransposeSlabBwd::execute() 後、buf_ には拡張グリッド (nx+3)*(ny+3)*(nz+3) が格納済み
    const double* in = buf_;
    const int yz3 = ny3_ * nz3_;
    const double fa_da = function_a(a) * Da_;

    const int ny = static_cast<int>(ny_ghost_);  // ny+1
    const int nz = static_cast<int>(nz_ghost_);  // nz+1

    #pragma omp parallel
    {
        //ポテンシャル勾配の計算（中心差分）
        #pragma omp for collapse(2) schedule(static)
        for (int i = 1; i < nx3_ - 1; ++i) {
            for (int j = 1; j < ny3_ - 1; ++j) {
                const double* __restrict row_i   = in + i       * yz3 + j       * nz3_;
                const double* __restrict row_im1 = in + (i - 1) * yz3 + j       * nz3_;
                const double* __restrict row_ip1 = in + (i + 1) * yz3 + j       * nz3_;
                const double* __restrict row_jm1 = in + i       * yz3 + (j - 1) * nz3_;
                const double* __restrict row_jp1 = in + i       * yz3 + (j + 1) * nz3_;

                double* __restrict dx = potx_ + ((i - 1) * ny + (j - 1)) * nz;
                double* __restrict dy = poty_ + ((i - 1) * ny + (j - 1)) * nz;
                double* __restrict dz = potz_ + ((i - 1) * ny + (j - 1)) * nz;

                for (int kk = 0; kk < nz; ++kk) {
                    const int k = kk + 1;
                    dx[kk] = (row_im1[k] - row_ip1[k]) * 0.5;
                    dy[kk] = (row_jm1[k] - row_jp1[k]) * 0.5;
                    dz[kk] = (row_i[k - 1] - row_i[k + 1]) * 0.5;
                }
            }
        }

        // CIC補間で粒子運動量を更新
        const int np = particle_.np;
        const double* __restrict px = particle_.x;
        const double* __restrict py = particle_.y;
        const double* __restrict pz = particle_.z;

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

            const int li  = ix - x0_;
            const int li1 = li + 1;
            const int lj  = iy - y0_;
            const int lj1 = lj + 1;
            const int lk  = iz - z0_;
            const int lk1 = lk + 1;

            const size_t lij   = (static_cast<size_t>(li)  * ny + lj)  * nz;
            const size_t lij1  = (static_cast<size_t>(li)  * ny + lj1) * nz;
            const size_t li1j  = (static_cast<size_t>(li1) * ny + lj)  * nz;
            const size_t li1j1 = (static_cast<size_t>(li1) * ny + lj1) * nz;

            const size_t lijk    = lij   + lk;
            const size_t lijk1   = lij   + lk1;
            const size_t lij1k   = lij1  + lk;
            const size_t lij1k1  = lij1  + lk1;
            const size_t li1jk   = li1j  + lk;
            const size_t li1jk1  = li1j  + lk1;
            const size_t li1j1k  = li1j1 + lk;
            const size_t li1j1k1 = li1j1 + lk1;

            const double wx0y0 = wx0 * wy0;
            const double wx0y1 = wx0 * wy1;
            const double wx1y0 = wx1 * wy0;
            const double wx1y1 = wx1 * wy1;
            const double wz0m  = wz0 * mass_;
            const double wz1m  = wz1 * mass_;

            const double wx0y0z0 = wx0y0 * wz0m;
            const double wx0y0z1 = wx0y0 * wz1m;
            const double wx0y1z0 = wx0y1 * wz0m;
            const double wx0y1z1 = wx0y1 * wz1m;
            const double wx1y0z0 = wx1y0 * wz0m;
            const double wx1y0z1 = wx1y0 * wz1m;
            const double wx1y1z0 = wx1y1 * wz0m;
            const double wx1y1z1 = wx1y1 * wz1m;

            const double gx = potx_[lijk]    * wx0y0z0 + potx_[lijk1]   * wx0y0z1
                            + potx_[lij1k]   * wx0y1z0 + potx_[lij1k1]  * wx0y1z1
                            + potx_[li1jk]   * wx1y0z0 + potx_[li1jk1]  * wx1y0z1
                            + potx_[li1j1k]  * wx1y1z0 + potx_[li1j1k1] * wx1y1z1;

            const double gy = poty_[lijk]    * wx0y0z0 + poty_[lijk1]   * wx0y0z1
                            + poty_[lij1k]   * wx0y1z0 + poty_[lij1k1]  * wx0y1z1
                            + poty_[li1jk]   * wx1y0z0 + poty_[li1jk1]  * wx1y0z1
                            + poty_[li1j1k]  * wx1y1z0 + poty_[li1j1k1] * wx1y1z1;

            const double gz = potz_[lijk]    * wx0y0z0 + potz_[lijk1]   * wx0y0z1
                            + potz_[lij1k]   * wx0y1z0 + potz_[lij1k1]  * wx0y1z1
                            + potz_[li1jk]   * wx1y0z0 + potz_[li1jk1]  * wx1y0z1
                            + potz_[li1j1k]  * wx1y1z0 + potz_[li1j1k1] * wx1y1z1;

            particle_.vx[p] += gx * fa_da;
            particle_.vy[p] += gy * fa_da;
            particle_.vz[p] += gz * fa_da;
        }
    }

    timer_.stop(t_gather_, MPI_COMM_WORLD);
    if(world_rank_ == 0) {
        DEBUG_LOG("Gather");
    }
}

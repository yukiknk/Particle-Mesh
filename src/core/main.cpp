#include "utils.h"
#include "fft.h"
#include "mpi_env.h"
#include "transpose_slab.h"
#include "reorder_slab.h"
#include "reorder_slab_back.h"
#include "transpose_slab_back.h"

int main(int argc, char** argv) {
    MPIEnv mpi(argc, argv); //MPI初期化

    int world_rank, world_size, nthreads;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    nthreads = omp_get_max_threads();

    //==========パラメータ初期化==========
    const int Ng = std::atoi(argv[1]);
    const int Nrow = std::atoi(argv[2]);
    const size_t Ng3 = static_cast<size_t>(Ng) * Ng * Ng;
    const size_t Np = static_cast<size_t>(Nrow) * Nrow * Nrow;
    const size_t Np_local = Np / world_size;
    const double mass=static_cast<double>(Ng3)/static_cast<double>(Np);
    #ifdef USE_FFTW
    const int stride = Ng + 2; //FFTWはパディングを考慮
    #else
    const int stride = Ng;
    #endif
    //==========プロセス配置初期化==========
    int dims[3]={0,0,0}; MPI_Dims_create(world_size,3,dims);
    std::array<int,3> dims3 = { dims[0], dims[1], dims[2] };
    int coords[3];
    coords[0] = world_rank / (dims[1] * dims[2]); coords[1] = (world_rank / dims[2])%dims[1]; coords[2] = world_rank % dims[2]; //プロセスの座標

    const int nx_loc=Ng/dims[0], ny_loc=Ng/dims[1], nz_loc=Ng/dims[2];//各プロセスが持つキューブの一辺の長さ
    const int nx=nx_loc+1, ny=ny_loc+1, nz=nz_loc+1; //+方向のゴーストを追加した場合の一辺の長さ
    const int nx1=nx+2, ny1=ny+2, nz1=nz+2; //+方向のゴーストを2,-方向のゴーストを1追加した場合の一辺の長さ
    const size_t ncell_local=static_cast<size_t>(nx)*ny*nz, ncell_local1 = static_cast<size_t>(nx1)*ny1*nz1;//キューブのメッシュ総数
    
    const int x0=coords[0]*nx_loc, y0=coords[1]*ny_loc, z0=coords[2]*nz_loc;//各プロセスのキューブの開始位置

    double* real_buf = static_cast<double*>(aligned_alloc_bytes(std::max(ncell_local*nthreads, static_cast<size_t>(Ng) * (Ng + 2) * (Ng / world_size + 1) + 3 * ncell_local)));

    //==========FFT初期化===========
    FFT fft(Ng, real_buf);
    MPI_Comm fft_comm = fft.comm();
    int color = fft.color();
    ptrdiff_t local_n0 = fft.local_n0();
    ptrdiff_t local_0_start = fft.local_0_start();
    ptrdiff_t local_alloc = fft.local_alloc();

    double* real_buf1 = real_buf + (std::max(static_cast<size_t>(local_alloc * 2), ncell_local1) * sizeof(double) + ALIGN - 1) / ALIGN * ALIGN / sizeof(double);
    double* real_buf2 = real_buf1 + (ncell_local * sizeof(double) + ALIGN - 1) / ALIGN * ALIGN / sizeof(double);
    double* real_buf3 = real_buf2 + (ncell_local * sizeof(double) + ALIGN - 1) / ALIGN * ALIGN / sizeof(double);
    //==========transpose初期化==========
    std::vector<ptrdiff_t> all_i0(world_size), all_n0(world_size);
    MPI_Datatype DT_PTRDIFF;
    MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(ptrdiff_t), &DT_PTRDIFF);
    MPI_Allgather(&local_0_start, 1, DT_PTRDIFF, all_i0.data(), 1, DT_PTRDIFF, MPI_COMM_WORLD);
    MPI_Allgather(&local_n0, 1, DT_PTRDIFF, all_n0.data(), 1, DT_PTRDIFF, MPI_COMM_WORLD);
    TransposeSlab trans(Ng, nx, ny, nz, x0, dims[0], all_i0, all_n0, MPI_COMM_WORLD);
    TransposeSlabBack trans_back(Ng, color, nx_loc, ny1, nz1, dims3, local_0_start, local_n0, MPI_COMM_WORLD);
    //==========reorder初期化==========
    std::unique_ptr<ReorderSlab> reorder;
    std::unique_ptr<ReorderSlabBack> reorder_back;
    if(color == 1){
        reorder = std::make_unique<ReorderSlab>(Ng, ny, nz, local_n0, stride, trans.recvcounts(), trans.rdispls(), dims3);
        reorder_back = std::make_unique<ReorderSlabBack>(Ng, ny_loc, nz_loc, stride, dims3, local_n0);
    }

    //==========粒子初期化==========
    double* posx = static_cast<double*>(aligned_alloc_bytes(Np_local));
    double* posy = static_cast<double*>(aligned_alloc_bytes(Np_local));
    double* posz = static_cast<double*>(aligned_alloc_bytes(Np_local));
    double* px = static_cast<double*>(aligned_alloc_bytes(Np_local));
    double* py = static_cast<double*>(aligned_alloc_bytes(Np_local));
    double* pz = static_cast<double*>(aligned_alloc_bytes(Np_local));
    init_particles(posx, posy, posz, Np_local, x0, y0, z0, nx_loc, ny_loc, nz_loc, world_rank);
    std::memset(px, 0, (sizeof(double) * Np_local + ALIGN-1) / ALIGN * ALIGN);
    std::memset(py, 0, (sizeof(double) * Np_local + ALIGN-1) / ALIGN * ALIGN);
    std::memset(pz, 0, (sizeof(double) * Np_local + ALIGN-1) / ALIGN * ALIGN);
    //==========mainループ==========
    const double a = Start_a;
    const int worm_up = 10;
    const int loop = 20;
    const int all_loop = worm_up + loop; 
    double now_time = 0.0;
    std::array<double, 9> exe_time;
    exe_time.fill(0.0);
    for(int i = 0; i < all_loop; ++i){
        //----------CIC----------
        getTime(&now_time);
        init_buf(real_buf, nx, ny, nz, ncell_local, nthreads);
        deposit_particles(posx, posy, posz, Np_local, real_buf, ncell_local, nx, ny, nz, x0, y0, z0, mass);
        reduce_private_to_shared(real_buf, ncell_local, nthreads);
        report_max_time(getTime(&now_time), MPI_COMM_WORLD, i, worm_up, exe_time[0]);

        //----------Alltoallv----------
        getTime(&now_time);
        trans.execute(real_buf, real_buf1);
        report_max_time(getTime(&now_time), MPI_COMM_WORLD, i, worm_up, exe_time[1]);

        if(color == 1) {
            //----------reorder----------
            getTime(&now_time);
            reorder->execute(real_buf1, real_buf);
            report_max_time(getTime(&now_time), fft_comm, i, worm_up, exe_time[2]);

            //----------FFT----------
            getTime(&now_time);
            fft.forward();
            report_max_time(getTime(&now_time), fft_comm, i, worm_up, exe_time[3]);

            //----------Green----------
            getTime(&now_time);
            fft.apply_green(a);
            report_max_time(getTime(&now_time), fft_comm, i, worm_up, exe_time[4]);

            //----------IFFT----------
            getTime(&now_time);
            fft.backward();
            report_max_time(getTime(&now_time), fft_comm, i, worm_up, exe_time[5]);

            //----------reorder----------
            getTime(&now_time);
            reorder_back->execute(real_buf, real_buf1);
            report_max_time(getTime(&now_time), fft_comm, i, worm_up, exe_time[6]);
        }
        //----------Alltoallv----------
        getTime(&now_time);
        trans_back.execute(real_buf1, real_buf);
        report_max_time(getTime(&now_time), MPI_COMM_WORLD, i, worm_up, exe_time[7]);
        
        //----------update particle----------
        getTime(&now_time);
        centered_derivatives(real_buf, real_buf1, real_buf2, real_buf3, nx1, ny1, nz1);
        update_particles(posx, posy, posz, real_buf1, real_buf2, real_buf3, px, py, pz, a, Np_local, ny, nz, x0, y0, z0);
        report_max_time(getTime(&now_time), MPI_COMM_WORLD, i, worm_up, exe_time[8]);
    }
    
    if(world_rank == 0) for(int i = 0; i < exe_time.size(); ++i){
        std::cout << exe_time[i] / loop << std::endl;
    }

    std::free(real_buf);
    std::free(posx);
    std::free(posy);
    std::free(posz);
    std::free(px);
    std::free(py);
    std::free(pz);
}

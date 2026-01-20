#include <array>
#include <cstdlib>
#include <iostream>

#include "mpi_env.h"
#include "buffer_manager.h"
#include "grid/grid.h"
#include "particle/particle.h"
#include "fft/fft.h"
#include "transpose/transpose_slab_fwd.h"
#include "interpolater.h"
#include "timer.h"

int main(int argc, char** argv) {
    //MPI初期化
    MPIEnv mpi_env(argc, argv);

    if (argc < 3) {
        if (mpi_env.world_rank() == 0) {
            std::cerr << "Usage: " << argv[0] << " <Ng> <Np>" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int Ng = std::atoi(argv[1]);
    int Np = std::atoi(argv[2]);

    const double Omega0 = 1.0;

    const int warm_up = 2;
    const int loop = 10;
    Timer timer(warm_up, loop, mpi_env.world_rank());

    BufferManager buffer_manager; //バッファ初期化
    Grid grid(Ng, mpi_env); //Grid初期化
    FFT fft(Ng, Omega0, mpi_env, buffer_manager, timer); //FFT初期化
    TransposeSlabFwd transpose_fwd(grid, fft, mpi_env, buffer_manager, timer); //Transpose_FWD初期化
    //Transpose_BWD初期化
    Particle particle(Np, grid, mpi_env); //Particle初期化
    Interpolater interpolater(grid, particle, mpi_env, buffer_manager, timer); //Interpolater初期化
    
    //バッファ確保
    buffer_manager.allocate();
    fft.create_plan();

    //ループのセットアップ
    const int all_loop = warm_up + loop;
    double a = 0.9;

    //メインループ
    for (int i = 0; i < all_loop; i++) {
        timer.set_iteration(i);
        
        interpolater.deposit(); //deposit

        transpose_fwd.execute();

        fft.forward(); //FFT
        fft.apply_green(a); //Green
        fft.backward(); //IFFT

        //pack
        //alltoallv
        //unpack

        //update particle
    }

    //時間出力
    timer.print();

    return 0;
}

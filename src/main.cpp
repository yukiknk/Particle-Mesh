#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "mpi_env.h"
#include "buffer_manager.h"
#include "grid/grid.h"
#include "particle/particle.h"
#include "fft/fft.h"
#include "fft/fft_fftw.h"
#include "fft/fft_ffte1.h"
#include "transpose/transpose_slab_fwd.h"
#include "transpose/transpose_slab_bwd.h"
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
    const int all_loop = warm_up + loop;
    double a = 0.9;

    int method_sequence[] = {1, 1, 2, 3};
    
    for (int use_ffte = 0; use_ffte <= 1; ++use_ffte) {
        int seq_idx = 0;
        for (int method : method_sequence) {
            seq_idx++;
            Timer timer(warm_up, loop, mpi_env.world_rank());

            BufferManager buffer_manager; //バッファ初期化
            Grid grid(Ng, mpi_env); //Grid初期化
            
            std::unique_ptr<FFT> fft;
            if (use_ffte) {
                if (mpi_env.world_rank() == 0) std::cout << "[DEBUG] Using FFTE (method=" << method << ")" << std::endl;
                fft = std::make_unique<FFT_FFTE1>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
            } else {
                if (mpi_env.world_rank() == 0) std::cout << "[DEBUG] Using FFTW (method=" << method << ")" << std::endl;
                fft = std::make_unique<FFT_FFTW>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
            }
            
            TransposeSlabFwd transpose_fwd(grid, *fft, mpi_env, buffer_manager, timer, method); //Transpose_FWD初期化
            TransposeSlabBwd transpose_bwd(grid, *fft, mpi_env, buffer_manager, timer, method); //Transpose_BWD初期化
            Particle particle(Np, grid, mpi_env); //Particle初期化
            Interpolater interpolater(grid, particle, mpi_env, buffer_manager, timer); //Interpolater初期化
            
            //バッファ確保
            buffer_manager.allocate(mpi_env.world_rank());
            fft->create_plan();

            //メインループ
            for (int i = 0; i < all_loop; i++) {
                timer.set_iteration(i);
                
                if (mpi_env.world_rank() == 0 && i == 0) std::cout << "[DEBUG] Loop started. " << std::endl;
                
                interpolater.deposit(); //deposit

                transpose_fwd.execute();

                fft->forward(); //FFT
                fft->apply_green(a); //Green
                fft->backward(); //IFFT

                transpose_bwd.execute();

                interpolater.gather(a); //update particle
            }

            //時間出力
            if (mpi_env.world_rank() == 0) {
                std::string fft_name = use_ffte ? "FFTE" : "FFTW";
                if (seq_idx == 1) {
                    std::cout << "\n========== " << fft_name << " Method " << method << " (Warm-up) ==========" << std::endl;
                } else {
                    std::cout << "\n========== " << fft_name << " Method " << method << " ==========" << std::endl;
                }
            }
            timer.print();
        }
    }

    return 0;
}

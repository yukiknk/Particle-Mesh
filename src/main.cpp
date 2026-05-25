#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "mpi_env.h"
#include "buffer_manager.h"
#include "grid/grid.h"
#include "particle/particle.h"
#include "fft/fft.h"
#include "fft/fft_fftw.h"
#include "fft/fft_ffte1.h"
#include "fft/fft_ffte2.h"
#include "transpose/transpose_fwd.h"
#include "transpose/transpose_bwd.h"
#include "transpose/transpose_fwd_slab.h"
#include "transpose/transpose_bwd_slab.h"
#include "transpose/transpose_fwd_pencil.h"
#include "transpose/transpose_bwd_pencil.h"
#include "interpolater.h"
#include "timer.h"
#include "debug.h"

int main(int argc, char** argv) {
    MPIEnv mpi_env(argc, argv);

    if (argc < 3) {
        if (mpi_env.world_rank() == 0) {
            std::cerr << "Usage: " << argv[0] << " <Ng> <Np>" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int Ng = std::atoi(argv[1]);
    int Np = std::atoi(argv[2]);

    const double Omega0 = 1.0;
    const int warm_up = 2;
    const int loop = 10;
    const int all_loop = warm_up + loop;
    double a = 0.9;

    // warm_up: 最初の1回だけプロセス全体を温める(FFTW で代表)
    {
        Timer timer(warm_up, loop, mpi_env.world_rank());
        int method = 1;
        BufferManager buffer_manager;
        Grid grid(Ng, mpi_env);

        std::unique_ptr<FFT> fft =
            std::make_unique<FFT_FFTW>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
        std::unique_ptr<TransposeFwd> transpose_fwd =
            std::make_unique<TransposeFwdSlab>(grid, *fft, mpi_env, buffer_manager, timer, method);
        std::unique_ptr<TransposeBwd> transpose_bwd =
            std::make_unique<TransposeBwdSlab>(grid, *fft, mpi_env, buffer_manager, timer, method);

        Particle particle(Np, grid, mpi_env);
        Interpolater interpolater(grid, particle, mpi_env, buffer_manager, timer);

        buffer_manager.allocate(mpi_env.world_rank());
        fft->create_plan();

        for (int i = 0; i < all_loop; i++) {
            timer.set_iteration(i);
            if (mpi_env.world_rank() == 0 && i == 0) DEBUG_LOG("Warmup started.");

            interpolater.deposit();
            transpose_fwd->execute();
            fft->forward();
            fft->apply_green(a);
            fft->backward();
            transpose_bwd->execute();
            interpolater.gather(a);
        }
    }

    int fft_sequence[] = {0, 1, 2};

    for (int fft_type : fft_sequence) {
        // FFTE2(fft_type==2)は method 概念がないので 1 回のみ
        std::vector<int> methods = (fft_type == 2)
                                 ? std::vector<int>{2}
                                 : std::vector<int>{2, 3};

        for (int method : methods) {
            Timer timer(warm_up, loop, mpi_env.world_rank());
            BufferManager buffer_manager;
            Grid grid(Ng, mpi_env);

            std::unique_ptr<FFT> fft;
            std::string fft_name;
            if (fft_type == 0) {
                if (mpi_env.world_rank() == 0) DEBUG_LOG("Using FFTW");
                fft = std::make_unique<FFT_FFTW>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
                fft_name = "FFTW";
            } else if (fft_type == 1) {
                if (mpi_env.world_rank() == 0) DEBUG_LOG("Using FFTE1D");
                fft = std::make_unique<FFT_FFTE1>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
                fft_name = "FFTE1D";
            } else {
                if (mpi_env.world_rank() == 0) DEBUG_LOG("Using FFTE2D");
                fft = std::make_unique<FFT_FFTE2>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
                fft_name = "FFTE2D";
            }

            std::unique_ptr<TransposeFwd> transpose_fwd;
            std::unique_ptr<TransposeBwd> transpose_bwd;
            if (fft_type != 2) {
                transpose_fwd = std::make_unique<TransposeFwdSlab>(grid, *fft, mpi_env, buffer_manager, timer, method);
                transpose_bwd = std::make_unique<TransposeBwdSlab>(grid, *fft, mpi_env, buffer_manager, timer, method);
            } else {
                transpose_fwd = std::make_unique<TransposeFwdPencil>(grid, *fft, mpi_env, buffer_manager, timer, method);
                transpose_bwd = std::make_unique<TransposeBwdPencil>(grid, *fft, mpi_env, buffer_manager, timer, method);
            }

            Particle particle(Np, grid, mpi_env);
            Interpolater interpolater(grid, particle, mpi_env, buffer_manager, timer);

            buffer_manager.allocate(mpi_env.world_rank());
            fft->create_plan();

            for (int i = 0; i < all_loop; i++) {
                timer.set_iteration(i);
                if (mpi_env.world_rank() == 0 && i == 0) DEBUG_LOG("Loop started.");

                interpolater.deposit();
                transpose_fwd->execute();
                fft->forward();
                fft->apply_green(a);
                fft->backward();
                transpose_bwd->execute();
                interpolater.gather(a);
            }

            if (mpi_env.world_rank() == 0) {
                std::cout << "\n========== " << fft_name << " Method " << method << " ==========" << std::endl;
            }
            timer.print();
        }
    }

    return 0;
}

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <set>

#include "mpi_env.h"
#include "buffer_manager.h"
#include "grid/grid.h"
#include "grid/grid_auto.h"
#include "grid/grid_input.h"
#include "particle/particle.h"
#include "particle/particle_auto.h"
#include "particle/particle_input.h"
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

namespace {

// "a,b,c" を小文字トークン集合へ
std::set<std::string> parse_list(const std::string& s) {
    std::set<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string tok = s.substr(i, j - i);
        for (auto& c : tok) c = std::tolower(c);
        if (!tok.empty()) out.insert(tok);
        i = j + 1;
    }
    return out;
}

// --key=val の val を取り出す。見つからなければ空文字。
std::string get_opt(int argc, char** argv, const std::string& key) {
    std::string prefix = "--" + key + "=";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], prefix.c_str(), prefix.size()) == 0) {
            return std::string(argv[i] + prefix.size());
        }
    }
    return "";
}

} // namespace

int main(int argc, char** argv) {
    MPIEnv mpi_env(argc, argv);
    const int rank = mpi_env.world_rank();

    // ---- 引数パース ----
    std::string ng_s   = get_opt(argc, argv, "ng");
    std::string np_s   = get_opt(argc, argv, "np");
    std::string root_s = get_opt(argc, argv, "root");
    std::string src_s  = get_opt(argc, argv, "source");
    std::string fft_s  = get_opt(argc, argv, "fft");
    std::string meth_s = get_opt(argc, argv, "method");

    if (ng_s.empty()) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " --ng=N [--source=auto,input] [--fft=fftw,ffte1d,ffte2d]"
                      << " [--method=1,2,3] [--np=N] [--root=DIR]" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int Ng = std::atoi(ng_s.c_str());
    int Np_side = np_s.empty() ? 0 : std::atoi(np_s.c_str());
    size_t Np3 = static_cast<size_t>(Np_side) * Np_side * Np_side;

    // 省略時は全選択
    std::set<std::string> sources = src_s.empty()  ? std::set<std::string>{"auto", "input"} : parse_list(src_s);
    std::set<std::string> ffts    = fft_s.empty()  ? std::set<std::string>{"fftw", "ffte1d", "ffte2d"} : parse_list(fft_s);
    std::set<std::string> methods = meth_s.empty() ? std::set<std::string>{"1", "2", "3"} : parse_list(meth_s);

    // 必須引数チェック
    if (sources.count("auto") && Np_side <= 0) {
        if (rank == 0) std::cerr << "source=auto requires --np=N" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (sources.count("input") && root_s.empty()) {
        if (rank == 0) std::cerr << "source=input requires --root=DIR" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const double Omega0 = 1.0;
    const int warm_up = 2;
    const int loop = 10;
    const int all_loop = warm_up + loop;
    double a = 0.9;

    // fft_type: 0=FFTW 1=FFTE1D 2=FFTE2D
    auto fft_name_of = [](int t) -> std::string {
        return t == 0 ? "FFTW" : (t == 1 ? "FFTE1D" : "FFTE2D");
    };
    auto fft_selected = [&](int t) {
        return ffts.count(t == 0 ? "fftw" : (t == 1 ? "ffte1d" : "ffte2d")) > 0;
    };

    const int world_size = mpi_env.world_size();

    // 1 サイクル（grid/particle 生成済み）に対し fft×method を回す
    auto run_combinations =
        [&](Grid& grid, Particle& particle, const std::string& source_name) {
        // grid/particle に依存する FFT/transpose は組み合わせごとに作り直す
        int fft_sequence[] = {0, 1, 2};
        for (int fft_type : fft_sequence) {
            if (!fft_selected(fft_type)) continue;
            for (const std::string& m : {std::string("1"), std::string("2"), std::string("3")}) {
                if (!methods.count(m)) continue;
                int method = std::atoi(m.c_str());

                Timer timer(warm_up, loop, rank);
                BufferManager buffer_manager;

                std::unique_ptr<FFT> fft;
                if (fft_type == 0)
                    fft = std::make_unique<FFT_FFTW>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
                else if (fft_type == 1)
                    fft = std::make_unique<FFT_FFTE1>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
                else
                    fft = std::make_unique<FFT_FFTE2>(Ng, Omega0, mpi_env, buffer_manager, timer, method);

                std::unique_ptr<TransposeFwd> transpose_fwd;
                std::unique_ptr<TransposeBwd> transpose_bwd;
                if (fft_type != 2) {
                    transpose_fwd = std::make_unique<TransposeFwdSlab>(grid, *fft, mpi_env, buffer_manager, timer, method);
                    transpose_bwd = std::make_unique<TransposeBwdSlab>(grid, *fft, mpi_env, buffer_manager, timer, method);
                } else {
                    transpose_fwd = std::make_unique<TransposeFwdPencil>(grid, *fft, mpi_env, buffer_manager, timer, method);
                    transpose_bwd = std::make_unique<TransposeBwdPencil>(grid, *fft, mpi_env, buffer_manager, timer, method);
                }

                Interpolater interpolater(grid, particle, mpi_env, buffer_manager, timer);
                buffer_manager.allocate(rank);
                fft->create_plan();

                for (int i = 0; i < all_loop; i++) {
                    timer.set_iteration(i);
                    if (rank == 0 && i == 0) DEBUG_LOG("Loop started.");
                    interpolater.deposit();
                    transpose_fwd->execute();
                    fft->forward();
                    fft->apply_green(a);
                    fft->backward();
                    transpose_bwd->execute();
                    interpolater.gather(a);
                }

                if (rank == 0) {
                    std::cout << "\n========== " << fft_name_of(fft_type)
                              << " Method " << method
                              << " Source " << source_name
                              << " NPROC " << world_size
                              << " ==========" << std::endl;
                }
                timer.print();
            }
        }
    };

    // ---- warm_up: 引数によらず method=1 / FFTW で 1 回だけ ----
    {
        Timer timer(warm_up, loop, rank);
        int method = 1;
        BufferManager buffer_manager;
        GridAuto grid(Ng, mpi_env);

        std::unique_ptr<FFT> fft =
            std::make_unique<FFT_FFTW>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
        std::unique_ptr<TransposeFwd> transpose_fwd =
            std::make_unique<TransposeFwdSlab>(grid, *fft, mpi_env, buffer_manager, timer, method);
        std::unique_ptr<TransposeBwd> transpose_bwd =
            std::make_unique<TransposeBwdSlab>(grid, *fft, mpi_env, buffer_manager, timer, method);

        ParticleAuto particle(static_cast<size_t>(Ng) * Ng * Ng, grid, mpi_env);
        Interpolater interpolater(grid, particle, mpi_env, buffer_manager, timer);

        buffer_manager.allocate(rank);
        fft->create_plan();

        for (int i = 0; i < all_loop; i++) {
            timer.set_iteration(i);
            if (rank == 0 && i == 0) DEBUG_LOG("Warmup started.");
            interpolater.deposit();
            transpose_fwd->execute();
            fft->forward();
            fft->apply_green(a);
            fft->backward();
            transpose_bwd->execute();
            interpolater.gather(a);
        }
    }

    // ---- auto 系: 生成 → 全組み合わせ → 破棄 ----
    if (sources.count("auto")) {
        if (rank == 0) DEBUG_LOG("=== source: auto ===");
        GridAuto grid(Ng, mpi_env);
        ParticleAuto particle(Np3, grid, mpi_env);
        run_combinations(grid, particle, "auto");
    }

    // ---- input 系: 生成 → 全組み合わせ → 破棄 ----
    if (sources.count("input")) {
        if (rank == 0) DEBUG_LOG("=== source: input ===");
        GridInput grid(root_s, mpi_env, Ng);
        ParticleInput particle(root_s, Ng, grid, mpi_env);
        run_combinations(grid, particle, "input");
    }

    return 0;
}

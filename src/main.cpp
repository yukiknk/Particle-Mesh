#include <array>
#include <cstdlib>
#include <cstring>
#include <cctype>
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
#ifdef USE_HEFFTE
#include "fft/fft_heffte.h"
#endif
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

int gcd_i(int a, int b) {
    while (b) { int t = a % b; a = b; b = t; }
    return a;
}

// 起動時にグループ構成を出力する。
// 上限値を書き換えて挙動を調べる運用では、どの設定で走ったかがログに
// 残っていないと結果を突き合わせられないため。
void print_config(int Ng, int world_size, int pgroup, int relay) {
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(world_size, 3, dims);

    std::cout << "\n===== Configuration =====" << std::endl;
    std::cout << "NPROC " << world_size << "  Ng " << Ng
              << "  dims " << dims[0] << "x" << dims[1] << "x" << dims[2] << std::endl;

    // world_size = 2べき成分 x 奇数成分 に分解
    int pow2 = 1, odd = world_size;
    while (odd % 2 == 0) { odd /= 2; pow2 *= 2; }
    std::cout << "world_size = " << pow2 << " x " << odd
              << "  (2べき成分 x 奇数成分)" << std::endl;
    if (world_size > Ng && pow2 % Ng != 0) {
        std::cout << "  [warn] 2べき成分が Ng の倍数ではありません"
                  << " (slab のスラブ幅が最小になりません)" << std::endl;
    }

    // リレーメッシュの有無（Pencil のみ対応）
    std::cout << "relay mesh = " << (relay ? "on" : "off")
              << (relay ? "  (グループ内 Alltoallv + グループ間 Reduce/Bcast)"
                        : "  (MPI_COMM_WORLD で一括 Alltoallv、Reduce/Bcast なし)")
              << std::endl;
    if (!relay) {
        std::cout << "  [note] relay=off は Pencil(FFTE2D) のみ有効。"
                  << "slab は従来動作のまま。" << std::endl;
    }

    const int cap2 = (pgroup > 0) ? pgroup : FFT_FFTE2::cap_for(Ng);
    struct Entry { const char* name; int cap; bool slab; };
    const Entry tbl[] = {
        {"FFTW ", FFT_FFTW::cap_for(Ng),  true},
        {"FFTE1", FFT_FFTE1::cap_for(Ng), true},
        {"FFTE2", cap2,                   false},
    };

    for (const Entry& e : tbl) {
        int g = gcd_i(world_size, e.cap);
        int G = (g > 0) ? world_size / g : 0;
        std::cout << "  " << e.name
                  << "  cap=" << e.cap
                  << "  group=" << g
                  << " x " << G;
        if (e.slab) {
            std::cout << "  (slab width " << (g > 0 ? Ng / g : 0) << ")";
        } else {
            int d2[2] = {0, 0};
            MPI_Dims_create(g, 2, d2);
            std::cout << "  (NPUX=" << d2[0] << ", NPUY=" << d2[1] << ")";
            if (!relay) std::cout << "  <- この " << g << " プロセスだけが FFT を担当";
        }
        std::cout << std::endl;
    }
    std::cout << "=========================\n" << std::endl;
}

// fft_type: 0=FFTW 1=FFTE1D 2=FFTE2D 3=heFFTe(slab) 4=heFFTe(pencil)
//           5=heFFTe(slab)、6=heFFTe(pencil) をデバイスモードで（共有バッファを GPU 上に置き、
//             Transpose と粒子処理も GPU で回す。PM_USE_KOKKOS でビルドしたときだけ使える）
const int kNumFftTypes = 7;
bool is_device_fft(int t) { return t == 5 || t == 6; }

const char* fft_label(int t) {
    static const char* const names[] = {"FFTW", "FFTE1D", "FFTE2D", "HEFFTE_SLAB", "HEFFTE_PENCIL", "HEFFTE_SLAB_GPU", "HEFFTE_PENCIL_GPU"};
    return names[t];
}

// --fft= に書かれた名前で選ばれているか。"heffte" は "heffte-slab" の別名。
bool fft_requested(const std::set<std::string>& ffts, int t) {
    static const char* const keys[] = {"fftw", "ffte1d", "ffte2d", "heffte-slab", "heffte-pencil", "heffte-slab-gpu", "heffte-pencil-gpu"};
    if (t == 3 && ffts.count("heffte")) return true;
    return ffts.count(keys[t]) > 0;
}

// heFFTe 版の FFT を作る。heFFTe なしのビルドでは実行時に止める。
std::unique_ptr<FFT> make_heffte(int Ng, double Omega0, const MPIEnv& mpi,
                                 BufferManager& buffer, Timer& timer, int method,
                                 int fft_type, int pgroup) {
#ifdef USE_HEFFTE
    const FFT_HEFFTE::Box box = (fft_type == 4 || fft_type == 6) ? FFT_HEFFTE::Box::Pencil : FFT_HEFFTE::Box::Slab;
    return std::make_unique<FFT_HEFFTE>(Ng, Omega0, mpi, buffer, timer, method, box, pgroup);
#else
    (void)Ng; (void)Omega0; (void)buffer; (void)timer; (void)method;
    (void)fft_type; (void)pgroup;
    if (mpi.world_rank() == 0) {
        std::cerr << "--fft=heffte*: このバイナリは heFFTe なしでビルドされています" << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    return nullptr;
#endif
}

// デバイスモード（fft_type 5, 6）のときは共有バッファを GPU 上に置く。部品を作る前に呼ぶ。
void setup_buffer_mode(int fft_type, BufferManager& buffer, int rank) {
    if (!is_device_fft(fft_type)) return;
#ifdef PM_USE_KOKKOS
    buffer.set_device(true);
    (void)rank;
#else
    (void)buffer;
    if (rank == 0) std::cerr << "--fft=heffte-*-gpu: このバイナリは Kokkos なしでビルドされています（KOKKOS=1 でビルドする）" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
}

// FFT の入力の形に合った Transpose を作る。
//   slab  : FFTW, FFTE1D, heFFTe(slab)      relay 非対応（従来動作のまま）
//   pencil: FFTE2D, heFFTe(pencil)
void make_transposes(int fft_type, const Grid& grid, const FFT& fft, const MPIEnv& mpi,
                     BufferManager& buffer, Timer& timer, int method, int relay,
                     std::unique_ptr<TransposeFwd>& fwd, std::unique_ptr<TransposeBwd>& bwd) {
    if (fft_type == 2 || fft_type == 4 || fft_type == 6) {
        fwd = std::make_unique<TransposeFwdPencil>(grid, fft, mpi, buffer, timer, method, relay);
        bwd = std::make_unique<TransposeBwdPencil>(grid, fft, mpi, buffer, timer, method, relay);
    } else {
        fwd = std::make_unique<TransposeFwdSlab>(grid, fft, mpi, buffer, timer, method);
        bwd = std::make_unique<TransposeBwdSlab>(grid, fft, mpi, buffer, timer, method);
    }
}

} // namespace

int main(int argc, char** argv) {
    MPIEnv mpi_env(argc, argv);
#ifdef PM_USE_KOKKOS
    PmKokkosScope kokkos_scope(argc, argv);   // MPIEnv より後に作り、先に破棄する
#endif
    const int rank = mpi_env.world_rank();

    // ---- 引数パース ----
    std::string ng_s   = get_opt(argc, argv, "ng");
    std::string np_s   = get_opt(argc, argv, "np");
    std::string root_s = get_opt(argc, argv, "root");
    std::string src_s  = get_opt(argc, argv, "source");
    std::string fft_s  = get_opt(argc, argv, "fft");
    std::string meth_s = get_opt(argc, argv, "method");
    std::string pg_s   = get_opt(argc, argv, "pgroup");
    const int pgroup   = pg_s.empty() ? 0 : std::atoi(pg_s.c_str());
    std::string rl_s   = get_opt(argc, argv, "relay");
    const int relay    = rl_s.empty() ? 1 : std::atoi(rl_s.c_str());

    if (ng_s.empty()) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " --ng=N [--source=auto,input] [--fft=fftw,ffte1d,ffte2d,heffte-slab,heffte-pencil,heffte-slab-gpu,heffte-pencil-gpu]"
                      << " [--method=1,2,3] [--np=N] [--root=DIR]"
                      << " [--pgroup=N] [--relay=0|1]" << std::endl;
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

    // 必須引数チェック（auto/input とも Np を使う。input は root も必要）
    if ((sources.count("auto") || sources.count("input")) && Np_side <= 0) {
        if (rank == 0) std::cerr << "source=auto/input requires --np=N" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (sources.count("input") && root_s.empty()) {
        if (rank == 0) std::cerr << "source=input requires --root=DIR" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int world_size = mpi_env.world_size();

    if (rank == 0) print_config(Ng, world_size, pgroup, relay);

    const double Omega0 = 1.0;
    const int warm_up = 2;
    const int loop = 10;
    const int all_loop = warm_up + loop;
    double a = 0.9;

    auto fft_name_of = [](int t) -> std::string {
        return fft_label(t);
    };
    auto fft_selected = [&](int t) {
        return fft_requested(ffts, t);
    };

    // 1 サイクル（grid/particle 生成済み）に対し fft×method を回す
    auto run_combinations =
        [&](Grid& grid, Particle& particle, const std::string& source_name) {
        // grid/particle に依存する FFT/transpose は組み合わせごとに作り直す
        for (int fft_type = 0; fft_type < kNumFftTypes; ++fft_type) {
            if (!fft_selected(fft_type)) continue;
            for (const std::string& m : {std::string("1"), std::string("2"), std::string("3")}) {
                if (!methods.count(m)) continue;
                int method = std::atoi(m.c_str());

                Timer timer(warm_up, loop, rank);
                BufferManager buffer_manager;
                setup_buffer_mode(fft_type, buffer_manager, rank);

                std::unique_ptr<FFT> fft;
                if (fft_type == 0)
                    fft = std::make_unique<FFT_FFTW>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
                else if (fft_type == 1)
                    fft = std::make_unique<FFT_FFTE1>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
                else if (fft_type == 2)
                    fft = std::make_unique<FFT_FFTE2>(Ng, Omega0, mpi_env, buffer_manager, timer, method, pgroup);
                else
                    fft = make_heffte(Ng, Omega0, mpi_env, buffer_manager, timer, method, fft_type, pgroup);

                std::unique_ptr<TransposeFwd> transpose_fwd;
                std::unique_ptr<TransposeBwd> transpose_bwd;
                make_transposes(fft_type, grid, *fft, mpi_env, buffer_manager, timer, method, relay,
                                transpose_fwd, transpose_bwd);

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
                    // pgroup / relay は既定値のときだけ省略する。
                    // 同じ nproc を条件違いで複数回まわしたとき、
                    // ログ上で見分けがつかなくなるのを防ぐため。
                    std::cout << "\n========== " << fft_name_of(fft_type)
                              << " Method " << method
                              << " Source " << source_name
                              << " NPROC " << world_size;
                    if (pgroup > 0) std::cout << " PGROUP " << pgroup;
                    if (relay == 0) std::cout << " RELAY 0";
                    std::cout << " ==========" << std::endl;
                }
                timer.print();
            }
        }
    };

    // ---- warm_up: 選択された最初の fft / method で 1 回だけ ----
    {
        Timer timer(warm_up, loop, rank);

        // 計測対象と同じ方式で暖機する
        int wtype = 0;
        for (int t = 0; t < kNumFftTypes; ++t) {
            if (fft_selected(t)) { wtype = t; break; }
        }
        int method = methods.count("1") ? 1 : (methods.count("2") ? 2 : 3);

        BufferManager buffer_manager;
        setup_buffer_mode(wtype, buffer_manager, rank);
        GridAuto grid(Ng, mpi_env);

        std::unique_ptr<FFT> fft;
        if (wtype == 0)
            fft = std::make_unique<FFT_FFTW> (Ng, Omega0, mpi_env, buffer_manager, timer, method);
        else if (wtype == 1)
            fft = std::make_unique<FFT_FFTE1>(Ng, Omega0, mpi_env, buffer_manager, timer, method);
        else if (wtype == 2)
            fft = std::make_unique<FFT_FFTE2>(Ng, Omega0, mpi_env, buffer_manager, timer, method, pgroup);
        else
            fft = make_heffte(Ng, Omega0, mpi_env, buffer_manager, timer, method, wtype, pgroup);

        std::unique_ptr<TransposeFwd> transpose_fwd;
        std::unique_ptr<TransposeBwd> transpose_bwd;
        make_transposes(wtype, grid, *fft, mpi_env, buffer_manager, timer, method, relay,
                        transpose_fwd, transpose_bwd);

        ParticleAuto particle(Np3, grid, mpi_env);
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
        ParticleInput particle(grid, Np3, mpi_env);
        run_combinations(grid, particle, "input");
    }

    return 0;
}

#pragma once
#include <mpi.h>
#include <cstddef>
#include <memory>
#include <vector>

#include "mpi_env.h"
#include "buffer_manager.h"
#include "timer.h"
#include "fft.h"

// heFFTe を使う FFT。実空間配列はホストに置き、forward / backward の中で
// ホスト <-> GPU のコピーをする。Green 関数はホストで掛ける。
// ただしデバイスモード（共有バッファが GPU 上）では、コピーせず GPU 上のまま計算する。
//
// heFFTe に渡す入力の形（Box）を2通り選べる。heFFTe の内部の回し方も同じ形に合わせる。
//   Slab   : x 方向のスラブ。既存の TransposeFwdSlab / TransposeBwdSlab を使う。内部も slab
//   Pencil : x, y の2次元分割。既存の TransposeFwdPencil / TransposeBwdPencil を使う。内部も pencil
// FFTW (TRANSPOSED_OUT) や FFTE (IOPT=-2) と同じく、順変換の出力は転置された並びのままにし、
// 入力の形へ戻す組み替えはしない（逆変換がその並びから始める）。
// どちらでも、ホスト配列の添字は (ix * ny + iy) * nz + iz（z が最も速い）。
//
// 既定は GPU (heffte::backend::cufft)。-DPM_HEFFTE_CPU を付けてビルドすると
// 同じコードが CPU バックエンドで動く（GPU の無い環境での確認用）。
class FFT_HEFFTE : public FFT {
public:
    enum class Box { Slab, Pencil };

    // グループ1つの最大プロセス数（Grouping に渡す cap）
    static int cap_for(int Ng, Box box) {
        return box == Box::Pencil ? (Ng / 2) * (Ng / 2) : Ng;
    }

    // cap_override: Pencil のときだけ有効（--pgroup）。
    FFT_HEFFTE(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method,
               Box box = Box::Slab, int cap_override = 0);
    ~FFT_HEFFTE() override;

    FFT_HEFFTE(const FFT_HEFFTE&) = delete;
    FFT_HEFFTE& operator=(const FFT_HEFFTE&) = delete;
    FFT_HEFFTE(FFT_HEFFTE&&) = delete;
    FFT_HEFFTE& operator=(FFT_HEFFTE&&) = delete;

    void create_plan() override;
    void forward() override;
    void backward() override;
    void apply_green(double a) override;

    int color() const override { return color_; }
    // FFTW と違い r2c 用の詰め物を持たない（1行 = z 方向の個数）
    int stride() const override { return nz_; }
    ptrdiff_t local_alloc() const override { return local_alloc_; }
    int grouping_ng() const override { return cap_; }

    // --- Slab 用 ---
    ptrdiff_t local_n0() const override { return nx_; }
    ptrdiff_t local_0_start() const override { return x0_; }

    // --- Pencil 用 ---
    std::vector<int> ln0x() const override { return ln0x_; }
    std::vector<int> ln0y() const override { return ln0y_; }
    std::vector<int> l0sx() const override { return l0sx_; }
    std::vector<int> l0sy() const override { return l0sy_; }
    ptrdiff_t local_n0_x() const override { return nx_; }
    ptrdiff_t local_n0_y() const override { return ny_; }
    ptrdiff_t local_0_start_x() const override { return x0_; }
    ptrdiff_t local_0_start_y() const override { return y0_; }

private:
    struct Impl;                    // heFFTe / CUDA 依存はここに隠す
    std::unique_ptr<Impl> impl_;

    Timer& timer_;
    int t_fft_;
    int t_ifft_;
    int t_green_;
    int t_copy_;

    int Ng_;
    Box box_;
    int cap_ = 0;
    int color_ = 0;
    int world_rank_;
    MPI_Comm comm_ = MPI_COMM_NULL;

    // このプロセスが持つ実空間ボックス [x0, x0+nx) x [y0, y0+ny) x [0, nz)。z は常に全体 (nz = Ng)
    int x0_ = 0, nx_ = 0;
    int y0_ = 0, ny_ = 0;
    int nz_ = 0;
    // 波数空間ボックス（順変換の出力、逆変換の入力）。z は r2c 軸で全体は 0 .. Ng/2。
    // 既定では実空間と違う分割（転置された並び）にして、入力の形へ戻す組み替えを省く。
    //   Slab   : y 方向のスラブ（x と z は全体）
    //   Pencil : x 方向に通った pencil（y と z で2次元分割）
    // 添字は (ix * ony + iy) * onz + iz。
    int ox0_ = 0, onx_ = 0;
    int oy0_ = 0, ony_ = 0;
    int oz0_ = 0, onz_ = 0;

    // グループ内の分割（全ランクのボックスを作り直すのに使う）
    int group_size_ = 1;
    int local_rank_ = 0;
    int npux_ = 1, npuy_ = 1;       // Pencil の2次元分割
    bool transposed_out_ = true;    // false にすると出力を入力と同じ分割にする（比較用）

    // グループ内ランク r の実空間 / 波数空間ボックス（{x0, nx, y0, ny, z0, nz}）
    void real_box_of(int r, int b[6]) const;
    void cplx_box_of(int r, int b[6]) const;

    ptrdiff_t local_alloc_ = 0;     // Transpose は 2 * local_alloc 個の double を FFT バッファとみなす

    // Pencil: グループ内全プロセスの分割情報
    std::vector<int> ln0x_, ln0y_, l0sx_, l0sy_;

    double* real_ = nullptr;        // 実空間配列（BufferManager の先頭）。通常はホスト側

    // デバイスモード（BufferManager::is_device()、--fft=heffte-slab-gpu）:
    // real_ は GPU 上にあり、heFFTe にそのまま渡す（ホスト <-> GPU のコピーをしない）。
    // Green 関数も GPU で掛ける。
    bool device_ = false;
};

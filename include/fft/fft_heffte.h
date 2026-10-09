#pragma once
#include <mpi.h>
#include <cstddef>
#include <memory>

#include "mpi_env.h"
#include "buffer_manager.h"
#include "timer.h"
#include "fft.h"

// heFFTe を使う FFT。
// 段階1（案A）: FFT_FFTW と同じ slab 型として振る舞い、既存の TransposeFwdSlab /
// TransposeBwdSlab をそのまま使う。実空間配列はホストに置き、forward / backward の
// 中でホスト <-> GPU のコピーをする。Green 関数はホストで掛ける。
//
// 既定は GPU (heffte::backend::cufft)。-DPM_HEFFTE_CPU を付けてビルドすると
// 同じコードが CPU バックエンドで動く（GPU の無い環境での確認用）。
class FFT_HEFFTE : public FFT {
public:
    static int cap_for(int Ng) { return Ng; }

    FFT_HEFFTE(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method);
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
    // FFTW と違い r2c 用の詰め物を持たない（1行 = Ng 個）
    int stride() const override { return Ng_; }
    ptrdiff_t local_n0() const override { return local_n0_; }
    ptrdiff_t local_0_start() const override { return local_0_start_; }
    ptrdiff_t local_alloc() const override { return local_alloc_; }

    int grouping_ng() const override { return cap_for(Ng_); }

private:
    struct Impl;                    // heFFTe / CUDA 依存はここに隠す
    std::unique_ptr<Impl> impl_;

    Timer& timer_;
    int t_fft_;
    int t_ifft_;
    int t_green_;
    int t_copy_;

    int Ng_;
    int color_ = 0;
    int world_rank_;
    MPI_Comm comm_ = MPI_COMM_NULL;

    ptrdiff_t local_alloc_ = 0;     // Transpose は 2 * local_alloc 個の double を FFT バッファとみなす
    ptrdiff_t local_n0_ = 1;        // x 方向のスラブ厚
    ptrdiff_t local_0_start_ = 0;   // スラブの開始 x

    double* real_ = nullptr;        // ホスト側の実空間配列 [local_n0][Ng][Ng]（BufferManager の先頭）
};

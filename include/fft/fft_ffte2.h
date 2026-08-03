#pragma once
#include <mpi.h>
#include <vector>
#include <complex>
#include <cstddef>

#include "mpi_env.h"
#include "buffer_manager.h"
#include "timer.h"
#include "fft.h"

extern "C" {
    void pdzfft3dv_(double* A, double* B, int* NX, int* NY, int* NZ,
                    int* ICOMMY, int* ICOMMZ, int* MEY, int* NPUY, int* NPUZ, int* IOPT);
    void pzdfft3dv_(double* A, double* B, int* NX, int* NY, int* NZ,
                    int* ICOMMY, int* ICOMMZ, int* MEY, int* NPUY, int* NPUZ, int* IOPT);
}

class FFT_FFTE2 : public FFT {
public:
    static int cap_for(int Ng) { return (Ng / 2) * (Ng / 2); }

    FFT_FFTE2(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method);
    ~FFT_FFTE2() override;

    FFT_FFTE2(const FFT_FFTE2&) = delete;
    FFT_FFTE2& operator=(const FFT_FFTE2&) = delete;
    FFT_FFTE2(FFT_FFTE2&&) = delete;
    FFT_FFTE2& operator=(FFT_FFTE2&&) = delete;

    void create_plan() override;
    __attribute__((aligned(256))) void forward() override;
    __attribute__((aligned(256))) void backward() override;
    __attribute__((aligned(256))) void apply_green(double a) override;

    int color() const override { return color_; }
    int stride() const override { return stride_; }
    ptrdiff_t local_alloc() const override { return local_alloc_; }

    int grouping_ng() const override { return cap_for(Ng_); }

    // --- Pencil 用インタフェース(TransposeFwdPencil が参照) ---
    std::vector<int> ln0x() const override { return ln0x_; }
    std::vector<int> ln0y() const override { return ln0y_; }
    std::vector<int> l0sx() const override { return l0sx_; }
    std::vector<int> l0sy() const override { return l0sy_; }

    ptrdiff_t local_n0_x() const override { return local_n0_x_; }
    ptrdiff_t local_n0_y() const override { return local_n0_y_; }
    ptrdiff_t local_0_start_x() const override { return local_0_start_x_; }
    ptrdiff_t local_0_start_y() const override { return local_0_start_y_; }

private:
    Timer& timer_;
    int t_fft_;
    int t_ifft_;
    int t_green_;

    int Ng_;
    int color_;
    int stride_;
    int world_rank_;

    // 2D 分割
    int NPUX_;
    int NPUY_;
    int colorx_;
    int colory_;

    MPI_Comm comm_   = MPI_COMM_NULL;   // FFT 担当グループ全体
    MPI_Comm comm_x_ = MPI_COMM_NULL;
    MPI_Comm comm_y_ = MPI_COMM_NULL;
    MPI_Fint fortran_Comm_x_;
    MPI_Fint fortran_Comm_y_;

    // 入力側の 2D 分割(Transpose が参照)
    ptrdiff_t local_n0_x_ = 1;
    ptrdiff_t local_n0_y_ = 1;
    ptrdiff_t local_0_start_x_ = 0;
    ptrdiff_t local_0_start_y_ = 0;

    // 出力側(green/apply_green 用)
    ptrdiff_t local_n1_y_ = 0;
    ptrdiff_t local_n1_z_ = 0;
    ptrdiff_t local_1_start_y_ = 0;
    ptrdiff_t local_1_start_z_ = 0;

    ptrdiff_t local_alloc_ = 0;

    // 全プロセスの分割情報(Allgather 結果)
    std::vector<int> ln0x_;
    std::vector<int> ln0y_;
    std::vector<int> l0sx_;
    std::vector<int> l0sy_;

    double* real_ = nullptr;
    double* calc_ = nullptr;
    std::complex<double>* complex_ = nullptr;
    double* green_ = nullptr;
};

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
    void pzdfft3d_(double* A, double* B, int* NX, int* NY, int* NZ, int* ICOMM, int* ME, int* NPU, int* IOPT);
    void pdzfft3d_(double* A, double* B, int* NX, int* NY, int* NZ, int* ICOMM, int* ME, int* NPU, int* IOPT);
}

class FFT_FFTE1 : public FFT {
public:
    FFT_FFTE1(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method);
    ~FFT_FFTE1() override;
    
    FFT_FFTE1(const FFT_FFTE1&) = delete;
    FFT_FFTE1& operator=(const FFT_FFTE1&) = delete;
    FFT_FFTE1(FFT_FFTE1&&) = delete;
    FFT_FFTE1& operator=(FFT_FFTE1&&) = delete;

    void create_plan() override;
    __attribute__((aligned(256))) void forward() override;
    __attribute__((aligned(256))) void backward() override;
    __attribute__((aligned(256))) void apply_green(double a) override;

    int color() const override { return color_; }
    int stride() const override { return stride_; }
    ptrdiff_t local_n0() const override { return local_n0_; }
    ptrdiff_t local_0_start() const override { return local_0_start_; }
    ptrdiff_t local_alloc() const override { return local_alloc_; }
    
    int grouping_ng() const override { return Ng_ / 2; }

private:
    Timer& timer_;
    int t_fft_;
    int t_ifft_;
    int t_green_;

    int Ng_;
    int color_;
    int stride_;
    int world_rank_;
    
    int rank_;
    int size_;
    MPI_Fint fortran_Comm_;
    
    MPI_Comm comm_ = MPI_COMM_NULL;
    
    ptrdiff_t local_alloc_ = 0;
    ptrdiff_t local_n0_ = 1;
    ptrdiff_t local_0_start_ = 0;
    ptrdiff_t local_n1_ = 0;
    ptrdiff_t local_1_start_ = 0;
    
    double* real_ = nullptr;
    double* calc_ = nullptr;
    std::complex<double>* complex_ = nullptr;
    double* green_ = nullptr;
};

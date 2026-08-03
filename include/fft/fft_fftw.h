#pragma once
#include <mpi.h>
#include <fftw3-mpi.h>

#include "mpi_env.h"
#include "buffer_manager.h"
#include "timer.h"
#include "fft.h"

class FFT_FFTW : public FFT {
public:
    static int cap_for(int Ng) { return Ng; }
    FFT_FFTW(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method);
    ~FFT_FFTW() override;
    
    FFT_FFTW(const FFT_FFTW&) = delete;
    FFT_FFTW& operator=(const FFT_FFTW&) = delete;
    FFT_FFTW(FFT_FFTW&&) = delete;
    FFT_FFTW& operator=(FFT_FFTW&&) = delete;

    void create_plan() override;
    __attribute__((aligned(256))) void forward() override;
    __attribute__((aligned(256))) void backward() override;
    __attribute__((aligned(256))) void apply_green(double a) override;

    int color() const override { return color_; }
    int stride() const override { return stride_; }
    ptrdiff_t local_n0() const override { return local_n0_; }
    ptrdiff_t local_0_start() const override { return local_0_start_; }
    ptrdiff_t local_alloc() const override { return local_alloc_; }
    
    int grouping_ng() const override { return cap_for(Ng_); }

private:
    Timer& timer_;
    int t_fft_;
    int t_ifft_;
    int t_green_;

    int Ng_;
    int color_;
    int stride_;
    int world_rank_;
    MPI_Comm comm_ = MPI_COMM_NULL;
    
    ptrdiff_t local_alloc_ = 0;
    ptrdiff_t local_n0_ = 1;
    ptrdiff_t local_0_start_ = 0;
    ptrdiff_t local_n1_ = 0;
    ptrdiff_t local_1_start_ = 0;
    
    double* real_ = nullptr;
    fftw_complex* complex_ = nullptr;
    double* green_ = nullptr;
    
    fftw_plan forward_ = nullptr;
    fftw_plan backward_ = nullptr;
};

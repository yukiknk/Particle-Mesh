#pragma once
#include <mpi.h>
#include <fftw3-mpi.h>

#include "mpi_env.hpp"
#include "buffer_manager.hpp"

class FFT {
public:
    FFT(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer);
    ~FFT();
    
    FFT(const FFT&) = delete;
    FFT& operator=(const FFT&) = delete;
    FFT(FFT&&) = delete;
    FFT& operator=(FFT&&) = delete;

    void create_plan();
    void forward();
    void backward();
    void apply_green(double a);

    int color() const { return color_; }
    ptrdiff_t local_n0() const { return local_n0_; }
    ptrdiff_t local_0_start() const { return local_0_start_; }

private:
    int Ng_;
    int color_;
    MPI_Comm comm_ = MPI_COMM_NULL;
    
    ptrdiff_t local_alloc_ = 0;
    ptrdiff_t local_n0_ = 0;
    ptrdiff_t local_0_start_ = 0;
    ptrdiff_t local_n1_ = 0;
    ptrdiff_t local_1_start_ = 0;
    
    double* real_ = nullptr;
    fftw_complex* complex_ = nullptr;
    double* green_ = nullptr;
    
    fftw_plan forward_ = nullptr;
    fftw_plan backward_ = nullptr;
};

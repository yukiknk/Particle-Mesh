#pragma once
#include <mpi.h>
#include <fftw3-mpi.h>
#include <vector>

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

    void forward();
    void backward();
    void apply_green(double a);

    // Transpose用の情報
    int color() const { return color_; }
    ptrdiff_t local_n0() const { return local_n0_; }
    ptrdiff_t local_0_start() const { return local_0_start_; }
    const std::vector<int>& ln0() const { return ln0_; }
    const std::vector<int>& l0s() const { return l0s_; }

private:
    int Ng_;
    int color_;
    MPI_Comm comm_ = MPI_COMM_NULL;
    
    ptrdiff_t local_alloc_ = 0;
    ptrdiff_t local_n0_ = 0;
    ptrdiff_t local_0_start_ = 0;
    ptrdiff_t local_n1_ = 0;
    ptrdiff_t local_1_start_ = 0;
    
    std::vector<int> ln0_;
    std::vector<int> l0s_;
    
    double* real_ = nullptr;
    fftw_complex* complex_ = nullptr;
    double* green_ = nullptr;
    
    fftw_plan forward_ = nullptr;
    fftw_plan backward_ = nullptr;
};

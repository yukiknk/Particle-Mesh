#pragma once
#include <omp.h>
#include <mpi.h>
#include <fftw3-mpi.h>
#include <string>
#include <iostream>
#include <complex>
#include <utils_inline.h>

class FFT {
public:
    FFT(int n, double* buf);

    ~FFT();

    MPI_Comm comm() const;
    int color() const;

    ptrdiff_t local_alloc() const;
    ptrdiff_t local_n0() const ;
    ptrdiff_t local_0_start() const;

    void forward();
    void backward();

    void apply_green(const double a) const;

private:
    int rank_, size_, color_;
    int n_;
    double* real_;
    fftw_complex* complex_;
    MPI_Comm    comm_;

    int nthreads_;

    ptrdiff_t local_alloc_ = 0, local_n0_ = 0, local_0_start_ = -1, local_n1_ = 0, local_1_start_ = -1;

    std::string filename_;
    bool need_export_;

    fftw_plan forward_{nullptr};
    fftw_plan backward_{nullptr};

    void destroy_plans() noexcept;

    double* green_ = nullptr;
};

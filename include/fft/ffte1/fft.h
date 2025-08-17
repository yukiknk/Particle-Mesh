#pragma once
#include <omp.h>
#include <mpi.h>
#include <complex>
#include <utils_inline.h>

extern "C" {
    void pzdfft3d_(double* A, double* B, int* NX, int* NY, int* NZ, int* ICOMM, int* ME, int* NPU, int* IOPT);
    void pdzfft3d_(double* A, double* B, int* NX, int* NY, int* NZ, int* ICOMM, int* ME, int* NPU, int* IOPT);
}

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
    double* data_;
    std::complex<double>* complex_;
    double* calc_;
    MPI_Comm    comm_;
    int fortran_Comm_;

    ptrdiff_t local_alloc_ = 0, local_n0_ = 0, local_0_start_ = -1, local_n1_ = 0, local_1_start_ = -1;

    double* green_ = nullptr;
};

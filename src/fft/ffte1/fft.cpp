#include "fft.h"

FFT::FFT(int n, double* buf) : n_(n), data_(buf) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

    comm_ = MPI_COMM_NULL;
    color_ = (rank_ < n_ / 2) ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color_, rank_, &comm_);

    if(color_ == 1){
        MPI_Comm_size(comm_, &size_);
        fortran_Comm_ = MPI_Comm_c2f(comm_);
        local_n0_ = n_ / size_;
        local_n1_ = n_ / (2 * size_);
        local_0_start_ = local_n0_ * rank_;
        local_1_start_ = local_n1_ * rank_;
        if(rank_ == 0) ++local_n1_;
        else ++local_1_start_;
        local_alloc_ = n_ * n_ * local_n1_;

        int IOPT = 0;
        pdzfft3d_(data_, calc_, &n_, &n_, &n_, &fortran_Comm_, &rank_, &size_, &IOPT);
        pzdfft3d_(data_, calc_, &n_, &n_, &n_, &fortran_Comm_, &rank_, &size_, &IOPT);

        complex_ = reinterpret_cast<std::complex<double>*>(data_);
        calc_ = data_ + local_alloc_ * 2;

        green_ = static_cast<double*>(aligned_alloc_bytes(local_alloc_));
        const double pi_n = 2.0 * M_PI / n_;
        for(int i = 0; i < n_; ++i) for(int j = 0; j < n_; ++j) for(int k = 0; k < local_n1_; ++k){
            if(!(i==0 && j==0 && k+local_1_start_==0)){
                size_t index = (i * n_ + j) * local_n1_ + k;
                green_[index] = -3 * Omega_0 / (4.0 * (3 - cos(pi_n * i) - cos(pi_n * j) - cos(pi_n * (k + local_1_start_))));
            }else green_[0] = 0.0;
        }
    }
}

FFT::~FFT() { 
    if(color_ == 1) {
        std::free(green_);
        MPI_Comm_free(&comm_);
    }
}

void FFT::apply_green(const double a) const {
    const double inv_a = 1.0 / a;
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < local_alloc_; ++i) {
        complex_[i] *= green_[i] * inv_a;
    }
}

MPI_Comm FFT::comm() const { return comm_; }
int FFT::color() const { return color_; }

ptrdiff_t FFT::local_alloc() const { return local_alloc_; }
ptrdiff_t FFT::local_n0() const { return local_n0_; }
ptrdiff_t FFT::local_0_start() const { return local_0_start_; }

void FFT::forward() { 
    int IOPT = -2;
	pdzfft3d_(data_, calc_, &n_, &n_, &n_, &fortran_Comm_, &rank_, &size_, &IOPT);
}
void FFT::backward() { 
    int IOPT = 2;
	pzdfft3d_(data_, calc_, &n_, &n_, &n_, &fortran_Comm_, &rank_, &size_, &IOPT);
}

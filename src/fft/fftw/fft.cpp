#include "fft.h"

FFT::FFT(int n, double* buf) : n_(n), real_(buf) {
    fftw_init_threads();
    fftw_mpi_init();
    nthreads_ = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

    comm_ = MPI_COMM_NULL;
    color_ = (rank_ < n_) ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color_, rank_, &comm_);

    if(color_ == 1){
        MPI_Comm_size(comm_, &size_);
        local_alloc_ = fftw_mpi_local_size_3d_transposed(n_, n_, n_/2+1, comm_, &local_n0_, &local_0_start_, &local_n1_, &local_1_start_);

        const size_t real_doubles = static_cast<size_t>(2) * local_alloc_;
        const size_t real_bytes   = sizeof(double) * real_doubles;

        uintptr_t base     = reinterpret_cast<uintptr_t>(real_);
        uintptr_t cpx_addr = base + real_bytes;
        size_t mis         = cpx_addr % ALIGN;
        if (mis) cpx_addr += (ALIGN - mis);

        complex_ = reinterpret_cast<fftw_complex*>(cpx_addr);

        filename_ = "wisdom_" + std::to_string(n_) + "_mpi_" + std::to_string(size_) + ".dat";
        int wisdom_loaded = 0; 
        if(rank_ == 0){
            FILE* fp = fopen(filename_.c_str(), "r");
            if (fp) {
                fclose(fp);
                wisdom_loaded = fftw_import_wisdom_from_filename(filename_.c_str());
            }
        }
        MPI_Bcast(&wisdom_loaded, 1, MPI_INT, 0, comm_);
        unsigned plan_flag;
        if(wisdom_loaded){
            plan_flag = FFTW_WISDOM_ONLY;
            fftw_mpi_broadcast_wisdom(comm_);
        }else{
            plan_flag = FFTW_EXHAUSTIVE;
        }

        forward_ = fftw_mpi_plan_dft_r2c_3d(n_, n_, n_, real_, complex_, comm_, plan_flag | FFTW_MPI_TRANSPOSED_OUT);
        backward_ = fftw_mpi_plan_dft_c2r_3d(n_, n_, n_, complex_, real_, comm_, plan_flag | FFTW_MPI_TRANSPOSED_IN);

        need_export_ = (wisdom_loaded == 0);
        green_ = static_cast<double*>(fftw_malloc(sizeof(double) * static_cast<size_t>(local_alloc_)));
        const double pi_n = 2.0 * M_PI / n_;
        for(int j = 0; j < local_n1_; ++j) for(int i = 0; i < n_; ++i) for(int k = 0; k < n_ / 2 + 1; ++k){
            if(!(i==0 && (j+local_1_start_)==0 && k==0)){
                size_t index = (j * n_ + i) * (n_ / 2 + 1) + k;
                green_[index] = -3 * Omega_0 / (4.0 * (static_cast<size_t>(n_) * n_ * n_) * (3 - cos(pi_n * i) - cos(pi_n * (j + local_1_start_)) - cos(pi_n * k)));
            }else green_[0] = 0.0;
        }
    }
}

FFT::~FFT() { 
    if(color_ == 1){
        if (rank_ == 0 && need_export_) {
            fftw_export_wisdom_to_filename(filename_.c_str());
            std::cout << "wisdom isn't exist" << std::endl;
        }else{
            if(rank_ == 0) std::cout << "wisdom is exist" << std::endl;
        }
        destroy_plans();
        MPI_Comm_free(&comm_);
        fftw_free(green_);
    }
}

void FFT::apply_green(const double a) const {
    const double inv_a = 1.0 / a;
    #pragma omp parallel for
    for(ptrdiff_t i = 0; i < local_alloc_; ++i) {
        complex_[i][0] *= green_[i] * inv_a;
        complex_[i][1] *= green_[i] * inv_a;
    }
}

void FFT::destroy_plans() noexcept{
    if (forward_)  { fftw_destroy_plan(forward_);  forward_  = nullptr; }
    if (backward_) { fftw_destroy_plan(backward_); backward_ = nullptr; }
}

MPI_Comm FFT::comm() const { return comm_; }
int FFT::color() const { return color_; }

ptrdiff_t FFT::local_alloc() const { return local_alloc_; }
ptrdiff_t FFT::local_n0() const { return local_n0_; }
ptrdiff_t FFT::local_0_start() const { return local_0_start_; }

void FFT::forward() { fftw_mpi_execute_dft_r2c(forward_, real_, complex_);}
void FFT::backward() { fftw_mpi_execute_dft_c2r(backward_, complex_, real_);}

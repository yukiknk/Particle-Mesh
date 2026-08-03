#include "fft/fft_ffte2.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <omp.h>
#include "debug.h"
#include "grouping.h"

FFT_FFTE2::FFT_FFTE2(int Ng, double Omega0, const MPIEnv& mpi, BufferManager& buffer, Timer& timer, int method)
    : timer_(timer),
      Ng_(Ng),
      stride_(Ng),
      world_rank_(mpi.world_rank())
{
    // タイマー登録
    t_fft_   = timer_.register_timer("FFT");
    t_ifft_  = timer_.register_timer("IFFT");
    t_green_ = timer_.register_timer("Green");

    // ---- グループ分け(超過時のみ複数グループ)----
    // 1グループの最大サイズは Ng^2/2(NPUX<=Ng, NPUY<=Ng/2 の積)
    Grouping grouping(fft.grouping_ng(), mpi, method);
    color_ = grouping.color;
    int local_rank = grouping.local_rank;

    // FFT グループ内コミュニケータ
    MPI_Comm_split(MPI_COMM_WORLD, color_, local_rank, &comm_);

    int size, rank;
    MPI_Comm_size(comm_, &size);
    MPI_Comm_rank(comm_, &rank);

    // ---- グループ内 2D 分割 ----
    // MPI_Dims_create は降順(dims[0]>=dims[1])で返す。
    // 大きい方を NPUX(<=Ng 制約)、小さい方を NPUY(<=Ng/2 制約)に割り当て。
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    NPUX_ = dims[0];   // Ng 制約の軸
    NPUY_ = dims[1];   // Ng/2 制約の軸

    // 自分の 2D 座標
    colorx_ = rank / NPUY_;
    colory_ = rank % NPUY_;

    // 2方向コミュニケータ
    MPI_Comm_split(comm_, colory_, rank, &comm_x_);
    MPI_Comm_split(comm_, colorx_, rank, &comm_y_);

    fortran_Comm_x_ = MPI_Comm_c2f(comm_x_);
    fortran_Comm_y_ = MPI_Comm_c2f(comm_y_);

    local_n0_x_ = static_cast<ptrdiff_t>(Ng_) / NPUX_;
    local_n0_y_ = static_cast<ptrdiff_t>(Ng_) / NPUY_;
    local_0_start_x_ = colorx_ * local_n0_x_;
    local_0_start_y_ = colory_ * local_n0_y_;

    local_n1_y_ = static_cast<ptrdiff_t>(Ng_) / NPUX_;
    local_n1_z_ = static_cast<ptrdiff_t>(Ng_) / NPUY_ / 2;
    local_1_start_y_ = colorx_ * local_n1_y_;
    local_1_start_z_ = colory_ * local_n1_z_;
    if (colory_ == 0) ++local_n1_z_;
    else ++local_1_start_z_;
    local_alloc_ = static_cast<ptrdiff_t>(Ng_) * local_n1_y_ * local_n1_z_;  

    // ---- バッファ登録 ----
    // 実数入力 A(NX, NY/NPUY, NZ/NPUZ) 相当のサイズ。
    // 旧コードの real_size_ = (Ng/NPUY + 2) * Ng * Ng / NPUX に対応
    size_t real_size = align_to_64(static_cast<size_t>(Ng_ / NPUY_ + 2) * Ng_ * Ng_ / NPUX_);
    size_t calc_size = real_size;

    buffer.register_buffer(real_, 0);
    buffer.register_buffer(calc_, real_size);
    buffer.update_max_size(real_size + calc_size);

    // ---- Green 関数の確保と構築 ----
    if (color_ == 0) {
        size_t green_bytes = ((static_cast<size_t>(local_alloc_) * sizeof(double) + 63) / 64) * 64;
        green_ = static_cast<double*>(std::aligned_alloc(64, green_bytes));
        if (green_ == nullptr) {
            std::cerr << "FFTE2: green allocation failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const double pi_n = 2.0 * M_PI / Ng_;

        // IOPT=-2 出力レイアウト: 最外 i(複素 x/2 方向, colory分割), 中間 j(colorx分割), 最内 k(全部)
        #pragma omp parallel for collapse(2)
        for (ptrdiff_t i = 0; i < local_n1_z_; ++i) {
            for (ptrdiff_t j = 0; j < local_n1_y_; ++j) {
                for (int k = 0; k < Ng_; ++k) {
                    size_t index = (static_cast<size_t>(i) * local_n1_y_ + j) * Ng_ + k;
                    int gi = i + local_1_start_z_;
                    int gj = j + local_1_start_y_;
                    if (!(gi == 0 && gj == 0 && k == 0)) {
                        green_[index] = -3.0 * Omega0 / (4.0 *
                            (3.0 - cos(pi_n * gi) - cos(pi_n * gj) - cos(pi_n * k)));
                    } else {
                        green_[index] = 0.0;
                    }
                }
            }
        }
    }

    // ---- 入力側分割情報をグループ内で収集 ----
    ln0x_.resize(size);
    ln0y_.resize(size);
    l0sx_.resize(size);
    l0sy_.resize(size);

    int ln0x_local = static_cast<int>(local_n0_x_);
    int ln0y_local = static_cast<int>(local_n0_y_);
    int l0sx_local = static_cast<int>(local_0_start_x_);
    int l0sy_local = static_cast<int>(local_0_start_y_);

    MPI_Allgather(&ln0x_local, 1, MPI_INT, ln0x_.data(), 1, MPI_INT, comm_);
    MPI_Allgather(&ln0y_local, 1, MPI_INT, ln0y_.data(), 1, MPI_INT, comm_);
    MPI_Allgather(&l0sx_local, 1, MPI_INT, l0sx_.data(), 1, MPI_INT, comm_);
    MPI_Allgather(&l0sy_local, 1, MPI_INT, l0sy_.data(), 1, MPI_INT, comm_);

    if (world_rank_ == 0) {
        DEBUG_LOG("Set FFTE2");
    }
}

void FFT_FFTE2::create_plan() {
    if (color_ == 0) {
        complex_ = reinterpret_cast<std::complex<double>*>(real_);

        int IOPT = 0;
        pdzfft3dv_(real_, calc_, &Ng_, &Ng_, &Ng_,
                   &fortran_Comm_y_, &fortran_Comm_x_, &colory_,
                   &NPUY_, &NPUX_, &IOPT);
        pzdfft3dv_(real_, calc_, &Ng_, &Ng_, &Ng_,
                   &fortran_Comm_y_, &fortran_Comm_x_, &colory_,
                   &NPUY_, &NPUX_, &IOPT);
    }
}

void FFT_FFTE2::forward() {
    if (color_ == 0) {
        timer_.start(comm_);
        int IOPT = -2;
        pdzfft3dv_(real_, calc_, &Ng_, &Ng_, &Ng_,
                   &fortran_Comm_y_, &fortran_Comm_x_, &colory_,
                   &NPUY_, &NPUX_, &IOPT);
        timer_.stop(t_fft_, comm_);
        if (world_rank_ == 0) {
            DEBUG_LOG("FFTE2 FFT");
        }
    }
}

void FFT_FFTE2::backward() {
    if (color_ == 0) {
        timer_.start(comm_);
        int IOPT = 2;
        pzdfft3dv_(real_, calc_, &Ng_, &Ng_, &Ng_,
                   &fortran_Comm_y_, &fortran_Comm_x_, &colory_,
                   &NPUY_, &NPUX_, &IOPT);
        timer_.stop(t_ifft_, comm_);
        if (world_rank_ == 0) {
            DEBUG_LOG("FFTE2 IFFT");
        }
    }
}

void FFT_FFTE2::apply_green(double a) {
    if (color_ != 0) return;

    timer_.start(comm_);
    const double inv_a = 1.0 / a;
    #pragma omp parallel for
    for (ptrdiff_t i = 0; i < local_alloc_; ++i) {
        double val = green_[i] * inv_a;
        complex_[i] *= val;
    }
    timer_.stop(t_green_, comm_);
    if (world_rank_ == 0) {
        DEBUG_LOG("FFTE2 Green");
    }
}

FFT_FFTE2::~FFT_FFTE2() {
    if (color_ == 0) {
        std::free(green_);
    }
    if (comm_x_ != MPI_COMM_NULL) MPI_Comm_free(&comm_x_);
    if (comm_y_ != MPI_COMM_NULL) MPI_Comm_free(&comm_y_);
    if (comm_   != MPI_COMM_NULL) MPI_Comm_free(&comm_);
}

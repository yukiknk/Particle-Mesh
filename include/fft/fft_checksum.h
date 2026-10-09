#pragma once
// FFT 実装どうしの結果照合用。環境変数 PM_CHECKSUM=1 のときだけ動く。
// backward() 直後の実空間配列（ポテンシャル）について、二乗和と最大絶対値を出す。
// 計測には含めない（タイマー停止後に呼ぶ）。
#include <mpi.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

inline bool pm_checksum_enabled() {
    const char* e = std::getenv("PM_CHECKSUM");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}

// real: nrows 行 × stride 個の並び（有効なのは各行の先頭 ncols 個）
inline void pm_print_checksum(const char* name, const double* real,
                              size_t nrows, int ncols, int stride, MPI_Comm comm) {
    double sum2 = 0.0, amax = 0.0;
    #pragma omp parallel for reduction(+:sum2) reduction(max:amax)
    for (size_t r = 0; r < nrows; ++r) {
        const double* row = real + r * static_cast<size_t>(stride);
        for (int k = 0; k < ncols; ++k) {
            const double v = row[k];
            sum2 += v * v;
            const double av = std::fabs(v);
            if (av > amax) amax = av;
        }
    }
    double gsum2 = 0.0, gmax = 0.0;
    MPI_Reduce(&sum2, &gsum2, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&amax, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::printf("[checksum] %-6s sum2=%.12e max=%.12e\n", name, gsum2, gmax);
        std::fflush(stdout);
    }
}

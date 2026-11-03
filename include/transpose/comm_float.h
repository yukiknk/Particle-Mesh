#pragma once
#include <mpi.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "buffer_manager.h"
#include "timer.h"
#include "utils.h"
#include "pm_device.h"

// PM_COMM_FLOAT=1 のとき、Transpose（粒子側の格子 <-> FFT の slab / pencil）の全対全を
// 単精度（float）で送る。送る直前に float にし、受けた直後に double に戻すので、
// 詰め・並べ替え・FFT・Green は double のまま。丸めは 1 回の送受信につき相対 6e-8 程度。
// 既定（未設定）は従来どおり double で送る。全ランクで同じ値にすること。
inline bool pm_comm_float_enabled() {
    const char* e = std::getenv("PM_COMM_FLOAT");
    return e != nullptr && std::strcmp(e, "1") == 0;
}

// 全対全を単精度で送るための部品。Transpose の各クラスが 1 つずつ持つ。
//   setup(): コンストラクタで、送受信の counts / displs が決まった後、update_max_size の前に呼ぶ
//   run()  : alltoallv() の代わりに呼ぶ（変換の時間は t_calc、通信は t_comm に入る）
class FloatAlltoallv {
public:
    // 単精度用の送受信バッファを共有バッファの base（double 個）以降に登録し、新しい使用量を返す。
    // on=false なら何もせず base を返す。
    size_t setup(bool on, BufferManager& buffer, size_t base,
                 const std::vector<int>& sendcounts, const std::vector<int>& sdispls,
                 const std::vector<int>& recvcounts, const std::vector<int>& rdispls) {
        on_ = on;
        if (!on_) return base;
        send_runs_ = merge_runs(sendcounts, sdispls);
        recv_runs_ = merge_runs(recvcounts, rdispls);
        const size_t n_send = send_runs_.empty() ? 0 : send_runs_.back().second;
        const size_t n_recv = recv_runs_.empty() ? 0 : recv_runs_.back().second;
        buffer.register_buffer(fsend_, base);
        base += align_to_64((n_send + 1) / 2);      // float 2 個 = double 1 個分
        buffer.register_buffer(frecv_, base);
        base += align_to_64((n_recv + 1) / 2);
        return base;
    }

    bool on() const { return on_; }

    void run(const double* send, const std::vector<int>& sendcounts, const std::vector<int>& sdispls,
             double* recv, const std::vector<int>& recvcounts, const std::vector<int>& rdispls,
             MPI_Comm comm, bool device, Timer& timer, int t_comm, int t_calc) {
        float* fs = reinterpret_cast<float*>(fsend_);
        float* fr = reinterpret_cast<float*>(frecv_);

        timer.start(MPI_COMM_WORLD);
        for (const auto& r : send_runs_) to_float(fs + r.first, send + r.first, r.second - r.first, device);
        timer.stop(t_calc, MPI_COMM_WORLD);

        timer.start(MPI_COMM_WORLD);
        MPI_Alltoallv(fs, sendcounts.data(), sdispls.data(), MPI_FLOAT,
                      fr, recvcounts.data(), rdispls.data(), MPI_FLOAT, comm);
        timer.stop(t_comm, MPI_COMM_WORLD);

        timer.start(MPI_COMM_WORLD);
        for (const auto& r : recv_runs_) to_double(recv + r.first, fr + r.first, r.second - r.first, device);
        timer.stop(t_calc, MPI_COMM_WORLD);
    }

private:
    // 送受信で触る区間 [displ, displ + count) を並べ、重なり・隣接をまとめる。
    // 触らない隙間は変換しない（受信側の隙間にある元の値を壊さないため）。
    static std::vector<std::pair<size_t, size_t>> merge_runs(const std::vector<int>& counts,
                                                             const std::vector<int>& displs) {
        std::vector<std::pair<size_t, size_t>> v;
        for (size_t r = 0; r < counts.size(); ++r) {
            if (counts[r] > 0) {
                v.emplace_back(static_cast<size_t>(displs[r]), static_cast<size_t>(displs[r]) + counts[r]);
            }
        }
        std::sort(v.begin(), v.end());
        std::vector<std::pair<size_t, size_t>> out;
        for (const auto& r : v) {
            if (!out.empty() && r.first <= out.back().second) out.back().second = std::max(out.back().second, r.second);
            else out.push_back(r);
        }
        return out;
    }

    static void to_float(float* dst, const double* src, size_t n, bool device) {
#ifdef PM_USE_KOKKOS
        if (device) { pm_dev_to_float(dst, src, n); return; }
#else
        (void)device;
#endif
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
    }

    static void to_double(double* dst, const float* src, size_t n, bool device) {
#ifdef PM_USE_KOKKOS
        if (device) { pm_dev_to_double(dst, src, n); return; }
#else
        (void)device;
#endif
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(src[i]);
    }

    bool on_ = false;
    double* fsend_ = nullptr;   // 実体は float の並び（共有バッファの一部なので double* で登録する）
    double* frecv_ = nullptr;
    std::vector<std::pair<size_t, size_t>> send_runs_;   // [始まり, 終わり)（要素数）
    std::vector<std::pair<size_t, size_t>> recv_runs_;
};

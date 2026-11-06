#pragma once
#include <cstdlib>
#include <cstring>

// FFTE（src/fft/p*fft3d*.f）は、FFT の中の MPI_ALLTOALL(V) にかかった時間を
// 共通ブロック /PMFTIM/ PMTCOM, PMT0 に足し込む。C++ 側はその前に 0 にして、後で読む。
struct PmFtim {
    double tcomm;   // PMTCOM: 通信時間の合計（秒、OpenMP の master スレッドで測った値）
    double t0;      // PMT0: 作業用
};
extern "C" PmFtim pmftim_;

// PM_FINE_TIMER=1 のとき、FFT / IFFT の中の通信時間を内訳（sub）として出す。
// 既定（未設定）では出力は従来と同じ。
inline bool pm_fine_timer_enabled() {
    const char* e = std::getenv("PM_FINE_TIMER");
    return e != nullptr && std::strcmp(e, "1") == 0;
}

// PM_FFT_COMM_FLOAT=1 のとき、FFTE の FFT の中の全対全（src/fft/pmcomm.f）を単精度（COMPLEX*8）で送る。
// 送る直前に丸め、受けた直後に倍精度へ戻す。FFT の計算は倍精度のまま。既定（未設定）は従来どおり。
// 全ランクで同じ値にすること。FFT の外の転置は PM_COMM_FLOAT（別のスイッチ）。
struct PmCflt {
    int flag;       // IPMCF: 0=倍精度 1=単精度
};
extern "C" PmCflt pmcflt_;

inline bool pm_fft_comm_float_enabled() {
    const char* e = std::getenv("PM_FFT_COMM_FLOAT");
    return e != nullptr && std::strcmp(e, "1") == 0;
}

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

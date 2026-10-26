#pragma once
// GPU に置いたまま計算するモード（--fft=heffte-slab-gpu）で使う、デバイス側の処理。
// 実装は src/pm_device.cpp（Kokkos、nvcc_wrapper でビルド）。ここには Kokkos の型を出さない。
// PM_USE_KOKKOS でビルドしたときだけ使える。
#include <cstddef>
#include <vector>

#ifdef PM_USE_KOKKOS
// メモリ（Kokkos の既定の実行先のメモリ。CUDA があれば GPU、無ければホスト）
void*  pm_dev_alloc(size_t bytes);
void   pm_dev_free(void* p);
void   pm_dev_to_dev(void* dst_dev, const void* src_host, size_t bytes);    // ホスト → デバイス
void   pm_dev_to_host(void* dst_host, const void* src_dev, size_t bytes);   // デバイス → ホスト
size_t* pm_dev_upload_index(const std::vector<size_t>& v);                   // 添字表をデバイスへ（pm_dev_free で解放）
void   pm_dev_fence();

// 計算
void pm_dev_zero(double* p, size_t n);
// TransposeFwdSlab::reorder: fftbuf[key[seg[s]]] = sum_{k in [seg[s], seg[s+1])} recv[idx[k]]
void pm_dev_fwd_reorder(double* fftbuf, const double* recv, const size_t* seg, const size_t* idx,
                        const size_t* key, size_t nseg);
// TransposeBwdSlab::reorder: send[i] = fftbuf[src[i]]
void pm_dev_gather_index(double* send, const double* fftbuf, const size_t* src, size_t n);
// Green 関数: 複素数 c[i] (= 実部虚部の2つ並び) に green[i] * factor を掛ける
void pm_dev_scale_complex(double* c, const double* green, size_t n, double factor);
#endif

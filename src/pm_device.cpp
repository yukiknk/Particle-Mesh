// include/pm_device.h の実装（Kokkos）。kokkos/CMakeLists.txt で nvcc_wrapper によりビルドする。
#ifdef PM_USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <cstring>
#include <iostream>
#include <mpi.h>

#include "pm_device.h"

namespace {
using Exec = Kokkos::DefaultExecutionSpace;
using Mem  = Exec::memory_space;
template <class T>
using DevU  = Kokkos::View<T*, Mem, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <class T>
using HostU = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<int64_t>>;
}  // namespace

void* pm_dev_alloc(size_t bytes) {
    void* p = Kokkos::kokkos_malloc<Mem>("pm_dev", bytes > 0 ? bytes : 1);
    if (p == nullptr) {
        std::cerr << "pm_dev_alloc: allocation failed (" << bytes << " bytes)" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;
}

void pm_dev_free(void* p) {
    if (p != nullptr) Kokkos::kokkos_free<Mem>(p);
}

void pm_dev_to_dev(void* dst_dev, const void* src_host, size_t bytes) {
    DevU<char> d(static_cast<char*>(dst_dev), bytes);
    HostU<const char> h(static_cast<const char*>(src_host), bytes);
    Kokkos::deep_copy(d, h);
}

void pm_dev_to_host(void* dst_host, const void* src_dev, size_t bytes) {
    HostU<char> h(static_cast<char*>(dst_host), bytes);
    DevU<const char> d(static_cast<const char*>(src_dev), bytes);
    Kokkos::deep_copy(h, d);
}

size_t* pm_dev_upload_index(const std::vector<size_t>& v) {
    size_t* p = static_cast<size_t*>(pm_dev_alloc(v.size() * sizeof(size_t)));
    if (!v.empty()) pm_dev_to_dev(p, v.data(), v.size() * sizeof(size_t));
    return p;
}

void pm_dev_fence() { Exec().fence(); }

void pm_dev_zero(double* p, size_t n) {
    DevU<double> v(p, n);
    Kokkos::deep_copy(Exec(), v, 0.0);
    Exec().fence();
}

void pm_dev_fwd_reorder(double* fftbuf, const double* recv, const size_t* seg, const size_t* idx,
                        const size_t* key, size_t nseg) {
    Kokkos::parallel_for("PM::fwd_reorder", Policy(0, static_cast<int64_t>(nseg)),
        KOKKOS_LAMBDA(const int64_t s) {
            const size_t beg = seg[s], end = seg[s + 1];
            double acc = 0.0;
            for (size_t k = beg; k < end; ++k) acc += recv[idx[k]];
            fftbuf[key[beg]] = acc;
        });
    Exec().fence();
}

void pm_dev_gather_index(double* send, const double* fftbuf, const size_t* src, size_t n) {
    Kokkos::parallel_for("PM::bwd_reorder", Policy(0, static_cast<int64_t>(n)),
        KOKKOS_LAMBDA(const int64_t i) { send[i] = fftbuf[src[i]]; });
    Exec().fence();
}

void pm_dev_scale_complex(double* c, const double* green, size_t n, double factor) {
    Kokkos::parallel_for("PM::green", Policy(0, static_cast<int64_t>(n)),
        KOKKOS_LAMBDA(const int64_t i) {
            const double g = green[i] * factor;
            c[2 * i]     *= g;
            c[2 * i + 1] *= g;
        });
    Exec().fence();
}
#endif  // PM_USE_KOKKOS

#pragma once
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include "debug.h"
#include "pm_device.h"

class BufferManager {
public:
    // true にすると共有バッファをデバイス（GPU）上に確保する（--fft=heffte-slab-gpu 用）。
    // 各部品はコンストラクタで is_device() を見て、ホスト用とデバイス用の処理を選ぶので、
    // 部品を作る前に呼ぶこと。PM_USE_KOKKOS でビルドしたときだけ使える。
    void set_device(bool on) { device_ = on; }
    bool is_device() const { return device_; }

    void update_max_size(size_t size) {
        if (size > max_size_) {
            max_size_ = size;
        }
    }
    
    void register_buffer(double*& ptr, size_t offset) {
        registrations_.push_back({&ptr, offset});
    }
    
    void allocate(int world_rank) {
        if (max_size_ == 0) {
            std::cerr << "BufferManager: max_size is 0" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if ((max_size_ * sizeof(double)) % 64 != 0) {
            std::cerr << "BufferManager: size is not 64-byte aligned ("
                      << max_size_ * sizeof(double) << " bytes)" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
#ifdef PM_USE_KOKKOS
        if (device_) {
            data_ = static_cast<double*>(pm_dev_alloc(max_size_ * sizeof(double)));
        } else
#endif
        data_ = static_cast<double*>(std::aligned_alloc(64, max_size_ * sizeof(double)));
        
        if (data_ == nullptr) {
            std::cerr << "BufferManager: allocation failed ("
                      << max_size_ * sizeof(double) << " bytes)" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        for (auto& reg : registrations_) {
            *reg.ptr = data_ + reg.offset;
        }

        if(world_rank == 0) {
            DEBUG_LOG("BufferManager: allocated " + std::to_string(max_size_) + "* sizeof(double)");
        }
    }

    ~BufferManager() {
#ifdef PM_USE_KOKKOS
        if (device_) { pm_dev_free(data_); return; }
#endif
        std::free(data_);
    }
    
    BufferManager() = default;
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;
    BufferManager(BufferManager&&) = delete;
    BufferManager& operator=(BufferManager&&) = delete;

private:
    struct Registration {
        double** ptr;
        size_t offset;
    };
    std::vector<Registration> registrations_;
    size_t max_size_ = 0;
    double* data_ = nullptr;
    bool device_ = false;
};

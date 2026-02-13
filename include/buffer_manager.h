#pragma once
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

class BufferManager {
public:
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
            std::cout << "BufferManager: allocated " << max_size_ << "* sizeof(double)" << std::endl;
        }
    }

    ~BufferManager() {
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
};

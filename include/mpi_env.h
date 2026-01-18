#pragma once
#include <mpi.h>
#include <omp.h>
#include <iostream>

class MPIEnv {
public:
    MPIEnv(int& argc, char**& argv) {
        int provided;
        int rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (rc != MPI_SUCCESS){
            std::cerr << "MPI Failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        nthreads_ = omp_get_max_threads();
        if (provided < MPI_THREAD_FUNNELED) {
            if (world_rank_ == 0) std::cerr << "Error: MPI does not provide MPI_THREAD_FUNNELED (provided=" << provided << ")" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    ~MPIEnv() { 
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) MPI_Finalize(); 
    }

    MPIEnv(const MPIEnv&) = delete;
    MPIEnv& operator=(const MPIEnv&) = delete;
    MPIEnv(MPIEnv&&) = delete;
    MPIEnv& operator=(MPIEnv&&) = delete;

    int world_rank() const {return world_rank_;}
    int world_size() const {return world_size_;}
    int nthreads() const {return nthreads_;}

private:
    int world_rank_;
    int world_size_;
    int nthreads_;
};

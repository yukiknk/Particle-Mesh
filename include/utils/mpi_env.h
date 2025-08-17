#pragma once
#include <mpi.h>

class MPIEnv {
public:
    MPIEnv(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    }
    ~MPIEnv() { MPI_Finalize(); }
};

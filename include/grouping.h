#pragma once
#include "mpi_env.h"

struct Grouping {
    int group_size;
    int num_groups;
    int group_id;
    int local_rank;
    int color;

    Grouping(int Ng, const MPIEnv& mpi, int method);
};

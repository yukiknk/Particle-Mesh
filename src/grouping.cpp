#include "grouping.h"

Grouping::Grouping(int Ng, const MPIEnv& mpi) {
    int world_size = mpi.world_size();
    int world_rank = mpi.world_rank();

    group_size = (world_size <= Ng) ? world_size : Ng;
    num_groups = world_size / group_size;
    
    // main (Sequential) 固有のグループ化処理
    group_id = world_rank / group_size;
    local_rank = world_rank % group_size;
    color = group_id;
}

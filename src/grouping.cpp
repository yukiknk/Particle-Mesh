#include "grouping.h"

Grouping::Grouping(int Ng, const MPIEnv& mpi, int method) {
    int world_size = mpi.world_size();
    int world_rank = mpi.world_rank();

    group_size = (world_size <= Ng) ? world_size : Ng;
    num_groups = world_size / group_size;
    
    switch (method) {
    case 1: // Sequential（連番方式）
        group_id = world_rank / group_size;
        local_rank = world_rank % group_size;
        break;

    case 2: // Interleaved（交互方式）
        group_id = (num_groups > 1) ? world_rank % num_groups : 0;
        local_rank = (num_groups > 1) ? world_rank / num_groups : world_rank;
        break;

    case 3: // 3D Subdivision（3D分割方式）
        if (num_groups <= 1) {
            group_id = 0;
            local_rank = world_rank;
        } else {
            int dims[3]={0,0,0};
            MPI_Dims_create(world_size,3,dims);

            int gdims[3]={0,0,0};
            MPI_Dims_create(group_size,3,gdims);

            int coords[3];
            int rank = mpi.world_rank();
            coords[0] = rank / (dims[1] * dims[2]);
            coords[1] = (rank / dims[2]) % dims[1];
            coords[2] = rank % dims[2];

            int group_coords[3];
            int local_coords[3];

            for(int d=0; d<3; ++d){
                group_coords[d] = coords[d] / gdims[d];
                local_coords[d] = coords[d] % gdims[d];
            }

            int ngroups_dim[3];
            for(int d=0; d<3; ++d){
                ngroups_dim[d] = dims[d] / gdims[d];
            }

            group_id =
                group_coords[0] * (ngroups_dim[1] * ngroups_dim[2]) +
                group_coords[1] * ngroups_dim[2] +
                group_coords[2];

            local_rank =
                local_coords[0] * (gdims[1] * gdims[2]) +
                local_coords[1] * gdims[2] +
                local_coords[2];
        }
        break;

    default:
        group_id = world_rank / group_size;
        local_rank = world_rank % group_size;
        break;
    }

    color = group_id;
}

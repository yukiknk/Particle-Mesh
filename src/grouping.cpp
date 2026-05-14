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
            // Gridと同じ3Dトポロジを構築
            int dims[3] = {0, 0, 0};
            MPI_Dims_create(world_size, 3, dims);
            int coords[3];
            coords[0] = world_rank / (dims[1] * dims[2]);
            coords[1] = (world_rank / dims[2]) % dims[1];
            coords[2] = world_rank % dims[2];

            // num_groupsをz方向に分割
            int gdims[3] = {1, 1, num_groups};

            int lsize[3] = {dims[0] / gdims[0], dims[1] / gdims[1], dims[2] / gdims[2]};
            int gcoords[3], lcoords[3];
            for (int d = 0; d < 3; ++d) {
                gcoords[d] = coords[d] / lsize[d];
                lcoords[d] = coords[d] % lsize[d];
            }
            group_id = gcoords[0] * (gdims[1] * gdims[2]) + gcoords[1] * gdims[2] + gcoords[2];
            local_rank = lcoords[0] * (lsize[1] * lsize[2]) + lcoords[1] * lsize[2] + lcoords[2];
        }
        break;

    default:
        group_id = world_rank / group_size;
        local_rank = world_rank % group_size;
        break;
    }

    color = group_id;
}

#include "grouping.h"
#include <iostream>

namespace {

inline int gcd_i(int a, int b) {
    while (b) { int t = a % b; a = b; b = t; }
    return a;
}

} // namespace

Grouping::Grouping(int cap, const MPIEnv& mpi, int method) {
    const int world_size = mpi.world_size();
    const int world_rank = mpi.world_rank();

    if (cap <= 0) {
        if (world_rank == 0) {
            std::cerr << "Grouping: invalid cap=" << cap << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 30);
    }

    // group_size = min(world_size の2べき成分, cap)
    // cap は2べきなので gcd と一致する
    group_size = gcd_i(world_size, cap);

    if (group_size <= 0 || group_size > cap || world_size % group_size != 0) {
        if (world_rank == 0) {
            std::cerr << "Grouping: invalid group_size=" << group_size
                      << " (world_size=" << world_size
                      << ", cap=" << cap << ")" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 31);
    }

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

            // method 3 は「各軸で gdims[d] が dims[d] を割り切る」ことを要求する。
            // world_size が非2べきの場合これは一般に成立しないため、
            // 黙って壊れる代わりに明示的に落とす。
            {
                bool ok = true;
                long long prod = 1;
                for (int d = 0; d < 3; ++d) {
                    if (gdims[d] <= 0 || dims[d] % gdims[d] != 0) { ok = false; break; }
                    prod *= dims[d] / gdims[d];
                }
                if (!ok || prod != static_cast<long long>(num_groups)) {
                    if (world_rank == 0) {
                        std::cerr << "Grouping: method=3 is not constructible for world_size="
                                  << world_size << " group_size=" << group_size
                                  << " (dims=" << dims[0] << "x" << dims[1] << "x" << dims[2]
                                  << ", gdims=" << gdims[0] << "x" << gdims[1] << "x" << gdims[2]
                                  << "). Use method=1 or method=2." << std::endl;
                    }
                    MPI_Abort(MPI_COMM_WORLD, 34);
                }
            }

            int coords[3];
            int rank = world_rank;
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

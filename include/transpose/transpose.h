#pragma once

#if defined(USE_TRANSPOSE_PENCIL)
#include "transpose_pencil_fwd.hpp"
#include "transpose_pencil_bwd.hpp"
#else
#include "transpose_slab_fwd.hpp"
#include "transpose_slab_bwd.hpp"
#endif

#pragma once

#if defined(USE_TRANSPOSE_PENCIL)
#include "transpose_pencil_fwd.h"
#include "transpose_pencil_bwd.h"
#else
#include "transpose_slab_fwd.h"
#include "transpose_slab_bwd.h"
#endif

#pragma once

#if defined(USE_GRID_INPUT)
#include "grid_input.h"
#elif defined(USE_GRID_AUTO)
#include "grid_auto.h"
#else
#include "grid_auto.h"  // デフォルト
#endif

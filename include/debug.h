#pragma once
#include <iostream>

#ifdef DEBUG_MODE
#define DEBUG_LOG(msg) std::cout << (msg) << std::endl
#else
#define DEBUG_LOG(msg) ((void)0)
#endif

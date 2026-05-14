#pragma once
#include <cstddef>

class FFT {
public:
    virtual ~FFT() = default;
    
    virtual void create_plan() = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void apply_green(double a) = 0;
    
    virtual int color() const = 0;
    virtual int stride() const = 0;
    virtual ptrdiff_t local_n0() const = 0;
    virtual ptrdiff_t local_0_start() const = 0;
    virtual ptrdiff_t local_alloc() const = 0;
    
    // Grouping に渡す有効Ng（FFTEはNg/2, FFTWはNg）
    virtual int grouping_ng() const = 0;
};

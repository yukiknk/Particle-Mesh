#pragma once
#include <cstddef>
#include <vector>

class FFT {
public:
    virtual ~FFT() = default;

    virtual void create_plan() = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void apply_green(double a) = 0;

    virtual int color() const = 0;
    virtual int stride() const = 0;

    // Grouping に渡す有効 Ng(FFTE は Ng/2, FFTW は Ng)
    virtual int grouping_ng() const = 0;

    // 全 FFT 共通(Slab/Pencil 両方が使う)
    virtual ptrdiff_t local_alloc() const = 0;

    // --- Slab 用(Pencil 系 FFT では未使用、デフォルト 0) ---
    virtual ptrdiff_t local_n0() const { return 0; }
    virtual ptrdiff_t local_0_start() const { return 0; }

    // --- Pencil 用(Slab 系 FFT では未使用、デフォルト空 / 0) ---
    virtual std::vector<int> ln0x() const { return {}; }
    virtual std::vector<int> ln0y() const { return {}; }
    virtual std::vector<int> l0sx() const { return {}; }
    virtual std::vector<int> l0sy() const { return {}; }
    virtual ptrdiff_t local_n0_x() const { return 0; }
    virtual ptrdiff_t local_n0_y() const { return 0; }
    virtual ptrdiff_t local_0_start_x() const { return 0; }
    virtual ptrdiff_t local_0_start_y() const { return 0; }
};

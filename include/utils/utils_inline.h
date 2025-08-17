#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <random>
#include <memory> 
#include <mpi.h>
#include <omp.h>

constexpr double Omega_0 = 1.0;
constexpr double Omega_m = 0.3;
constexpr double Omega_lambda = 0.7;
constexpr double Omega_k = 1 - Omega_m - Omega_lambda;
constexpr double Da = 0.001, Start_a = 0.9;
constexpr int ALIGN = 64;

inline __attribute__((always_inline)) void* aligned_alloc_bytes(size_t n_elem){
    size_t bytes=(n_elem * sizeof(double) + ALIGN-1) / ALIGN * ALIGN; 
    return std::aligned_alloc(ALIGN,bytes);
}

inline __attribute__((always_inline)) void cic_deposit(double* __restrict a, size_t base, size_t yz_stride, size_t x_stride, double tx, double ty, double tz, double m){
    const double sx0 = 1.0 - tx, sx1 = tx;
    const double sy0 = 1.0 - ty, sy1 = ty;
    const double sz0 = 1.0 - tz, sz1 = tz;

    const double sxy00 = sx0 * sy0;
    const double sxy10 = sx1 * sy0;
    const double sxy01 = sx0 * sy1;
    const double sxy11 = sx1 * sy1;

    const double m0 = m * sz0;
    const double m1 = m * sz1;

    double* __restrict p = a + base;

    p[0]                  += sxy00 * m0;
    p[x_stride]           += sxy10 * m0;
    p[yz_stride]          += sxy01 * m0;
    p[x_stride + yz_stride] += sxy11 * m0;

    p[1]                  += sxy00 * m1;
    p[x_stride + 1]       += sxy10 * m1;
    p[yz_stride + 1]      += sxy01 * m1;
    p[x_stride + yz_stride + 1] += sxy11 * m1;
}

inline __attribute__((always_inline)) double function_a(const double a){
	return 1.0 / sqrt(Omega_m / a + Omega_k + Omega_lambda * a * a);
}

inline __attribute__((always_inline)) double trilerp1(const double* __restrict f, int b, int xs, int ys, double tx, double ty, double tz)
{
    const double mx = 1.0 - tx;
    const double my = 1.0 - ty;
    const double mz = 1.0 - tz;

    const double f00 = f[b] * mx + f[b + xs] * tx;
    const double f10 = f[b + ys] * mx + f[b + xs + ys] * tx;
    const double fz0 = f00 * my + f10 * ty;

    const double g00 = f[b + 1]        * mx + f[b + xs + 1]        * tx;
    const double g10 = f[b + ys + 1]   * mx + f[b + xs + ys + 1]   * tx;
    const double fz1 = g00 * my + g10 * ty;

    return fz0 * mz + fz1 * tz;
}

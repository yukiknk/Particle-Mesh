# Kokkos 版 Interpolater だけを nvcc_wrapper でビルドして静的ライブラリにする。
# 本体（pm_fft_test）は Makefile.miyabi で mpicxx (nvc++) のままビルドし、これをリンクする。
#   Makefile.miyabi から KOKKOS=1 のときに自動で呼ばれる
cmake_minimum_required(VERSION 3.22)
project(pm_interp_kokkos CXX)
find_package(Kokkos REQUIRED)
set(PM_MPI_INC "" CACHE STRING "MPI の include ディレクトリ（mpicxx --showme:incdirs）")
add_library(pm_interp_kokkos STATIC ../src/interpolater_kokkos.cpp ../src/pm_device.cpp)
target_include_directories(pm_interp_kokkos PRIVATE ../include ${PM_MPI_INC})
target_compile_definitions(pm_interp_kokkos PRIVATE PM_USE_KOKKOS)
target_link_libraries(pm_interp_kokkos PUBLIC Kokkos::kokkos)

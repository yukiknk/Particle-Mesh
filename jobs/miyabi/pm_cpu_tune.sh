#!/bin/sh
#PBS -q debug-g
#PBS -l select=4:mpiprocs=72:ompthreads=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gu99
#PBS -j oe

# CPU 版の調整: 4 ノード、Ng=512 で「プロセス数/ノード × FFT ライブラリ」を総当たりし、CPU で最も速い設定を探す
#   粒子処理は Kokkos CPU（atomic）。従来版も 1/ノード 以外で速くなるか見るため FFTE1D だけ回す
#   追加の調整: qsub -v PPNS=9:18:36,FFTS=ffte1d pm_cpu_tune.sh（コロン区切り）
. /work/gu99/u99001/Particle-Mesh/jobs/miyabi/pm_common.sh
OUT=$PM/runs/cpu_tune_$(date +%Y%m%d_%H%M)_${PBS_JOBID%%.*}
mkdir -p $OUT
PPNS=$(echo ${PPNS:-1:2:4:8} | tr ":" " "); FFTS=$(echo ${FFTS:-fftw:ffte1d:ffte2d} | tr ":" " ")
for P in $PPNS; do
  for F in $FFTS; do
    run cpu_kk 4 $P ./pm_fft_test_kokkos cpu atomic $F 512 256
  done
  run cpu 4 $P ./pm_fft_test_orig cpu scatter ffte1d 512 256
done
echo "summary: $OUT/summary.txt"

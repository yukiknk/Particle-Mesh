#!/bin/sh
#PBS -q debug-g
#PBS -l select=4:mpiprocs=72:ompthreads=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gu99
#PBS -j oe

# CPU 版の調整（進捗報告 ① Kokkos 導入と CPU 最適化）
#   4 ノード、Ng=512、FFT は FFTE-1D（以前の調整で最速）で、
#   粒子処理の方式（従来版 / Kokkos 複製 / Kokkos atomic）× プロセス数/ノード を総当たりする。
#   プロセス数を増やすとスレッド数が減り、複製のコスト（格子 × スレッド数）が下がるので、
#   どこかで複製が atomic より速くなるかを確かめる。
#   プロセス数/ノード 1, 2, 4, 8 → 72, 36, 18, 9 スレッド（72 コアを全部使う）
#                    16, 32       → 4, 2 スレッド（72 を割り切れないので 64 コアだけ使う）
# ぶれ対策: 同じ条件で 3 回投入し、agg_summary.py で中央値をとる
#   for i in 1 2 3; do qsub jobs/miyabi/pm_cpu_tune.sh; done
#   python3 jobs/miyabi/agg_summary.py runs/cpu_tune_*_*_*/summary.txt
# 変更: qsub -v PPNS=1:8,FFT=fftw,NG=512,NP=256 ...（PPNS はコロン区切り）
. /work/gu99/u99001/Particle-Mesh/jobs/miyabi/pm_common.sh
NG=${NG:-512}; NP=${NP:-256}; FFT=${FFT:-ffte1d}
OUT=$PM/runs/cpu_tune_$(date +%Y%m%d_%H%M)_${PBS_JOBID%%.*}
mkdir -p $OUT
echo "# Ng=$NG Np=$NP FFT=$FFT nodes=4 EXTRA=$PM_EXTRA" >> $OUT/summary.txt
PPNS=$(echo ${PPNS:-1:2:4:8:16:32} | tr ":" " ")
for P in $PPNS; do
  run cpu        4 $P ./pm_fft_test_orig   cpu scatter $FFT $NG $NP
  run cpu_dup    4 $P ./pm_fft_test_kokkos cpu scatter $FFT $NG $NP
  run cpu_atomic 4 $P ./pm_fft_test_kokkos cpu atomic  $FFT $NG $NP
done
echo "summary: $OUT/summary.txt"

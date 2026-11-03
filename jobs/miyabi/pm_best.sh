#!/bin/sh
#PBS -q debug-g
#PBS -l walltime=00:30:00
#PBS -W group_list=gu99
#PBS -j oe

# CPU と GPU を「それぞれ最良の設定」で比べるための測定（ノード数ごとに投入）
#   CPU: FFTW / FFTE-1D / FFTE-2D × プロセス数/ノード 1, 2, 4, 8, 16（粒子処理は Kokkos atomic）
#        → 集計時に、各 FFT・各ノード数で最も速いプロセス数を選ぶ
#   GPU: 全部 GPU（heFFTe slab / pencil、1 プロセス/ノード = GPU 1 枚/ノード）
# ノード数は qsub の -l select で決める（N を合わせる）。ぶれ対策で同じ条件を 3 回投入する。
#   for N in 2 4 8 16; do for i in 1 2 3; do
#     qsub -l select=$N:mpiprocs=72:ompthreads=1 -v N=$N,NG=512,NP=256 jobs/miyabi/pm_best.sh; done; done
#   for N in 8 16; do for i in 1 2 3; do
#     qsub -l select=$N:mpiprocs=72:ompthreads=1 -v N=$N,NG=1024,NP=512 jobs/miyabi/pm_best.sh; done; done
#   集計: python3 jobs/miyabi/agg_summary.py runs/best_ng512_*/summary.txt
. /work/gu99/u99001/Particle-Mesh/jobs/miyabi/pm_common.sh
N=${N:?N=ノード数 を指定}; NG=${NG:-512}; NP=${NP:-256}
OUT=$PM/runs/best_ng${NG}_n${N}_$(date +%Y%m%d_%H%M)_${PBS_JOBID%%.*}
mkdir -p $OUT
echo "# Ng=$NG Np=$NP nodes=$N EXTRA=$PM_EXTRA" >> $OUT/summary.txt
PPNS=$(echo ${PPNS:-1:2:4:8:16} | tr ":" " ")
for F in fftw ffte1d ffte2d; do
  for P in $PPNS; do
    run cpu_atomic $N $P ./pm_fft_test_kokkos cpu atomic $F $NG $NP
  done
done
run gpu_slab $N 1 ./pm_fft_test_kokkos gpu atomic heffte-slab-gpu   $NG $NP
run gpu_pen  $N 1 ./pm_fft_test_kokkos gpu atomic heffte-pencil-gpu $NG $NP
echo "summary: $OUT/summary.txt"

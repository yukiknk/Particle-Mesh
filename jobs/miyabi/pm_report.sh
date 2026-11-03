#!/bin/sh
#PBS -q debug-g
#PBS -l select=16:mpiprocs=72:ompthreads=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gu99
#PBS -j oe

# 進捗報告用の測定（Ng ごとに投入）
#   qsub -v NG=512,NP=256,CPU_FFT=<調整で選んだFFT>,CPU_PPN=<調整で選んだプロセス数/ノード> pm_report.sh
#   qsub -v NG=1024,NP=512,CPU_FFT=...,CPU_PPN=... pm_report.sh
# 構成（スライドの流れの順）
#   cpu        : 従来版（粒子処理は従来、FFTE1D、1 プロセス/ノード）… 前回までの CPU 版
#   cpu_dup    : 粒子処理を Kokkos CPU（スレッドごとに複製、従来と同じ方式）… 原因の切り分け用
#   cpu_atomic : 粒子処理を Kokkos CPU（atomic）
#   cpu_best   : CPU で最も速い設定（Kokkos CPU atomic + 調整で選んだ FFT とプロセス数/ノード）
#   gpu_fft    : FFT だけ GPU（heFFTe slab）、粒子処理は従来 … 前回の報告の構成
#   gpu_part   : FFT と粒子処理を GPU（データはホスト経由）
#   gpu_slab   : 全部 GPU（データを GPU に置いたまま、slab）
#   gpu_pen    : 全部 GPU（データを GPU に置いたまま、pencil）
# GPU 版は 1 プロセス/ノード（1 ノードに GPU 1 枚）
# ぶれ対策: 同じ条件で 3 回投入し（別ジョブ = 別のノードの組み合わせになりやすい）、
#   python3 jobs/miyabi/agg_summary.py runs/report_ng512_*/summary.txt で回ごとの中央値をとる
. /work/gu99/u99001/Particle-Mesh/jobs/miyabi/pm_common.sh
NG=${NG:-512}; NP=${NP:-256}; CPU_FFT=${CPU_FFT:-ffte1d}; CPU_PPN=${CPU_PPN:-1}
OUT=$PM/runs/report_ng${NG}_$(date +%Y%m%d_%H%M)_${PBS_JOBID%%.*}
mkdir -p $OUT
echo "# Ng=$NG Np=$NP CPU_FFT=$CPU_FFT CPU_PPN=$CPU_PPN EXTRA=$PM_EXTRA" >> $OUT/summary.txt

# 正しさの確認（Ng=512 のときだけ）: 同じ実行の中で全構成のポテンシャルのチェックサムを比べる
if [ "$NG" -lt 1024 ]; then
  mpiexec -n 4 --map-by ppr:1:node:PE=72 --bind-to core ./pm_wrap.sh 72 gpu atomic 1 ./pm_fft_test_kokkos \
    --ng=$NG --np=$NP --source=auto --fft=ffte1d,heffte-slab,heffte-pencil,heffte-slab-gpu,heffte-pencil-gpu --method=2 \
    > $OUT/check.log 2>&1
  echo "## check exit=$?" >> $OUT/summary.txt
  grep -E "sum2" $OUT/check.log | awk '{print $2, $3}' | sort | uniq -c >> $OUT/summary.txt
fi

if [ "$NG" -ge 1024 ]; then NODES="8 16"; else NODES="2 4 8 16"; fi
for N in $NODES; do
  run cpu        $N 1         ./pm_fft_test_orig   cpu scatter ffte1d            $NG $NP
  run cpu_dup    $N 1         ./pm_fft_test_kokkos cpu scatter ffte1d            $NG $NP
  run cpu_atomic $N 1         ./pm_fft_test_kokkos cpu atomic  ffte1d            $NG $NP
  run cpu_best   $N $CPU_PPN  ./pm_fft_test_kokkos cpu atomic  $CPU_FFT          $NG $NP
  run gpu_fft    $N 1         ./pm_fft_test_orig   cpu scatter heffte-slab       $NG $NP
  run gpu_part   $N 1         ./pm_fft_test_kokkos gpu atomic  heffte-slab       $NG $NP
  run gpu_slab   $N 1         ./pm_fft_test_kokkos gpu atomic  heffte-slab-gpu   $NG $NP
  run gpu_pen    $N 1         ./pm_fft_test_kokkos gpu atomic  heffte-pencil-gpu $NG $NP
done
echo "summary: $OUT/summary.txt"

#!/bin/sh
#PBS -q debug-g
#PBS -l walltime=00:20:00
#PBS -W group_list=gu99
#PBS -j oe

# 通信を減らす 2 案の効果を測る（PR #16・#17 をマージしてからビルドして使う）
#   単精度: PM_COMM_FLOAT=1（FFT の外の転置の全対全を float で送る。PR #17）
#   p2p   : PM_HEFFTE_ALG=p2p_plined（heFFTe の中の組み替えの通信方式）
#   GPU: heFFTe slab / pencil × {既定, 単精度, p2p, 両方}
#   CPU: FFTE-1D（Ng=1024 で最良だった proc/node: 8 ノード 16、16 ノード 2）× {既定, 単精度}
#   すべて PM_FINE_TIMER=1（FFTE の FFT の中の通信時間を sub 行で出す。PR #16）
#   PM_CHECKSUM=1 で倍精度版とのチェックサムの差も見る（計時の区間の外で計算する）
# ノード数ごとに 3 回投入する:
#   for N in 8 16; do for i in 1 2 3; do
#     qsub -l select=$N:mpiprocs=72:ompthreads=1 -v N=$N jobs/miyabi/pm_comm_opt.sh; done; done
#   集計: python3 jobs/miyabi/agg_summary.py runs/commopt_ng1024_n8_*/summary.txt
#         grep -h "checksum" runs/commopt_ng1024_n8_*/*.log | sort | uniq -c
. /work/gu99/u99001/Particle-Mesh/jobs/miyabi/pm_common.sh
N=${N:?N=ノード数 を指定}; NG=${NG:-1024}; NP=${NP:-512}; CHECKSUM=${CHECKSUM:-1}
if [ -z "$CPU_PPN" ]; then
  if [ "$N" -ge 16 ]; then CPU_PPN=2; else CPU_PPN=16; fi
fi
OUT=$PM/runs/commopt_ng${NG}_n${N}_$(date +%Y%m%d_%H%M)_${PBS_JOBID%%.*}
mkdir -p $OUT
echo "# Ng=$NG Np=$NP nodes=$N cpu_ppn=$CPU_PPN checksum=$CHECKSUM EXTRA=$PM_EXTRA" >> $OUT/summary.txt

# 条件ごとの環境変数を計算ノードへ渡すラッパー（pm_common.sh の pm_wrap.sh と同じ設定 + 追加の変数）
#   pm_wrap_opt.sh <OMP_NUM_THREADS> <PM_INTERP> <PM_DEPOSIT> <PM_CHECKSUM> <追加の変数 "A=1,B=2"> <bin> <args...>
cat > ./pm_wrap_opt.sh <<'EOS'
#!/bin/bash
export OMP_NUM_THREADS=$1 OMP_PROC_BIND=close OMP_PLACES=cores OMP_STACKSIZE=64M
export PM_INTERP=$2 PM_DEPOSIT=$3 PM_CHECKSUM=$4 PM_WISDOM_DIR=/work/gu99/u99001/Particle-Mesh
export PM_FINE_TIMER=1
for kv in $(echo "$5" | tr "," " "); do export "$kv"; done
shift 5
ulimit -s unlimited
exec "$@"
EOS
chmod +x ./pm_wrap_opt.sh

# run_opt <tag> <nodes> <procs/node> <interp> <fft> <追加の変数>
run_opt () {
  TAG=$1; N=$2; P=$3; INTERP=$4; FFT=$5; VARS=$6
  T=$((72 / P))
  LOG=$OUT/${TAG}_ng${NG}_n${N}_p${P}.log
  mpiexec -n $((N * P)) --map-by ppr:${P}:node:PE=${T} --bind-to core \
    ./pm_wrap_opt.sh $T $INTERP atomic $CHECKSUM "$VARS" ./pm_fft_test_kokkos \
    --ng=$NG --np=$NP --source=auto --fft=$FFT --method=2 $PM_EXTRA > $LOG 2>&1
  echo "## $TAG Ng=$NG nodes=$N ppn=$P fft=$FFT exit=$?" >> $OUT/summary.txt
  # Total の後ろの sub 行（PM_FINE_TIMER=1）も拾うため、閉じの ===== まで取る
  sed -n '/===== Timer Results =====/,/^=====/p' $LOG | grep -v "=====" >> $OUT/summary.txt
}

for B in slab pencil; do
  F=heffte-${B}-gpu
  run_opt gpu_${B}      $N 1 gpu $F "PM_COMM_FLOAT=0"
  run_opt gpu_${B}_f32  $N 1 gpu $F "PM_COMM_FLOAT=1"
  run_opt gpu_${B}_p2p  $N 1 gpu $F "PM_COMM_FLOAT=0,PM_HEFFTE_ALG=p2p_plined"
  run_opt gpu_${B}_both $N 1 gpu $F "PM_COMM_FLOAT=1,PM_HEFFTE_ALG=p2p_plined"
done
run_opt cpu_ffte1d     $N $CPU_PPN cpu ffte1d "PM_COMM_FLOAT=0"
run_opt cpu_ffte1d_f32 $N $CPU_PPN cpu ffte1d "PM_COMM_FLOAT=1"
echo "summary: $OUT/summary.txt"

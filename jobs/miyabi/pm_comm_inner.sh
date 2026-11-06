#!/bin/sh
#PBS -q debug-g
#PBS -l walltime=00:20:00
#PBS -W group_list=gu99
#PBS -j oe

# FFTE の FFT の中の全対全も単精度で送る案 PM_FFT_COMM_FLOAT=1 の効果を測る（この PR をマージしてからビルドして使う）
#   外の転置の単精度（PM_COMM_FLOAT=1、PR #17）と組み合わせて 4 条件:
#     base / outer（外のみ。既存の f32 と同じ） / inner（中のみ） / both（外＋中）
#   CPU: FFTE-1D と FFTE-2D（proc/node は FFTE-1D が Ng=1024 で最良だった値（8 ノード 16、16 ノード 2）を両方に使う
#   PM_FINE_TIMER=1 で FFT の中の通信時間（sub FFT Comm）も出る。変換の時間は通信に含めない（FFT の計算側に入る）
#   PM_CHECKSUM=1 で倍精度版とのチェックサムの差も見る
# ノード数ごとに 3 回投入する:
#   for N in 8 16; do for i in 1 2 3; do
#     qsub -l select=$N:mpiprocs=72:ompthreads=1 -v N=$N jobs/miyabi/pm_comm_inner.sh; done; done
#   集計: python3 jobs/miyabi/agg_summary.py runs/commin_ng1024_n8_*/summary.txt
. /work/gu99/u99001/Particle-Mesh/jobs/miyabi/pm_common.sh
N=${N:?N=ノード数 を指定}; NG=${NG:-1024}; NP=${NP:-512}; CHECKSUM=${CHECKSUM:-1}
if [ -z "$CPU_PPN" ]; then
  if [ "$N" -ge 16 ]; then CPU_PPN=2; else CPU_PPN=16; fi
fi
OUT=$PM/runs/commin_ng${NG}_n${N}_$(date +%Y%m%d_%H%M)_${PBS_JOBID%%.*}
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

for F in ffte1d ffte2d; do
  run_opt cpu_${F}_base  $N $CPU_PPN cpu $F "PM_COMM_FLOAT=0,PM_FFT_COMM_FLOAT=0"
  run_opt cpu_${F}_outer $N $CPU_PPN cpu $F "PM_COMM_FLOAT=1,PM_FFT_COMM_FLOAT=0"
  run_opt cpu_${F}_inner $N $CPU_PPN cpu $F "PM_COMM_FLOAT=0,PM_FFT_COMM_FLOAT=1"
  run_opt cpu_${F}_both  $N $CPU_PPN cpu $F "PM_COMM_FLOAT=1,PM_FFT_COMM_FLOAT=1"
done
echo "summary: $OUT/summary.txt"

#!/bin/sh
#PBS -q regular-g
#PBS -l walltime=00:40:00
#PBS -W group_list=gu99
#PBS -j oe

# Ng=2048 の測定（16・32・64 ノード）。1 回の投入で 1 条件を 1 回だけ測る
#   COND（4 条件。それぞれ最速と見られる設定）:
#     gpu_slab    heFFTe slab GPU、外の転置を単精度（PM_COMM_FLOAT=1）、1 ノード 1 プロセス
#     gpu_pencil  heFFTe pencil GPU、外の転置を単精度、1 ノード 1 プロセス
#     cpu_ffte1d  FFTE-1D（slab 相当）CPU、外＋FFT の中を単精度（PM_COMM_FLOAT=1,PM_FFT_COMM_FLOAT=1）
#     cpu_ffte2d  FFTE-2D（pencil 相当）CPU、外＋中を単精度
#   「最速」は Ng=512 の測定（miyabi-測定結果-Ng512-20261110）からの選び方。Ng=2048 では未確認
#   CPU の proc/node は Ng=1024 で最良だった値（16 ノード以上は 2）を既定にした。Ng=2048 では未測定なので
#   CPU_PPN=4 などで上書きできる
# 投入（ノード数 N は 16・32・64。Np は既定 1024³。3 回ずつなら i を回す）:
#   for C in gpu_slab gpu_pencil cpu_ffte1d cpu_ffte2d; do
#     qsub -l select=16:mpiprocs=72:ompthreads=1 -v N=16,COND=$C jobs/miyabi/pm_ng2048_scale.sh; done
#   まず 16 ノードを 1 回ずつ（4 本）流して動作を確かめ、ノード時間を見てから 32・64 と 3 回に広げる
# 集計: runs/ng2048s_*/summary.txt（`## <COND> ... exit=` の行のあとに区間ごとの中央値が続く）
. /work/gu99/u99001/Particle-Mesh/jobs/miyabi/pm_common.sh
N=${N:?N=ノード数 を指定}; COND=${COND:?COND=gpu_slab|gpu_pencil|cpu_ffte1d|cpu_ffte2d を指定}
NG=${NG:-2048}; NP=${NP:-1024}
case $COND in
  gpu_slab)   FFT=heffte-slab-gpu;   INTERP=gpu; P=1; VARS="PM_COMM_FLOAT=1,PM_FFT_COMM_FLOAT=0" ;;
  gpu_pencil) FFT=heffte-pencil-gpu; INTERP=gpu; P=1; VARS="PM_COMM_FLOAT=1,PM_FFT_COMM_FLOAT=0" ;;
  cpu_ffte1d) FFT=ffte1d;            INTERP=cpu; P=${CPU_PPN:-$([ "$N" -ge 16 ] && echo 2 || echo 16)}; VARS="PM_COMM_FLOAT=1,PM_FFT_COMM_FLOAT=1" ;;
  cpu_ffte2d) FFT=ffte2d;            INTERP=cpu; P=${CPU_PPN:-$([ "$N" -ge 16 ] && echo 2 || echo 16)}; VARS="PM_COMM_FLOAT=1,PM_FFT_COMM_FLOAT=1" ;;
  *) echo "unknown COND=$COND"; exit 2 ;;
esac
T=$((72 / P))
OUT=$PM/runs/ng2048s_${COND}_n${N}_$(date +%Y%m%d_%H%M)_${PBS_JOBID%%.*}
mkdir -p $OUT
echo "# Ng=$NG Np=$NP nodes=$N cond=$COND fft=$FFT ppn=$P vars=$VARS" >> $OUT/summary.txt

# 試運転（pm_ng2048_trial.sh）と同じ設定。チェックサムと細かい計時を有効にし、条件ごとの変数を渡す
cat > ./pm_wrap_scale.sh <<'EOS'
#!/bin/bash
export OMP_NUM_THREADS=$1 OMP_PROC_BIND=close OMP_PLACES=cores OMP_STACKSIZE=64M
export PM_INTERP=$2 PM_DEPOSIT=atomic PM_CHECKSUM=1 PM_WISDOM_DIR=/work/gu99/u99001/Particle-Mesh
export PM_FINE_TIMER=1
for kv in $(echo "$3" | tr "," " "); do export "$kv"; done
shift 3
ulimit -s unlimited
exec "$@"
EOS
chmod +x ./pm_wrap_scale.sh

LOG=$OUT/${COND}_ng${NG}_n${N}_p${P}.log
mpiexec -n $((N * P)) --map-by ppr:${P}:node:PE=${T} --bind-to core \
  ./pm_wrap_scale.sh $T $INTERP "$VARS" ./pm_fft_test_kokkos \
  --ng=$NG --np=$NP --source=auto --fft=$FFT --method=2 $PM_EXTRA > $LOG 2>&1
echo "## $COND Ng=$NG nodes=$N ppn=$P fft=$FFT exit=$?" >> $OUT/summary.txt
sed -n '/===== Timer Results =====/,/^=====/p' $LOG | grep -v "=====" >> $OUT/summary.txt
echo "summary: $OUT/summary.txt"

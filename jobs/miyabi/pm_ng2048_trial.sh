#!/bin/sh
#PBS -q regular-g
#PBS -l walltime=00:30:00
#PBS -W group_list=gu99
#PBS -j oe

# Ng=2048 の試運転（1 条件を 1 回だけ。メモリ・時間・チェックサムの確認が目的）
#   既定: 8 ノード、Np=1024（1024³ 粒子）、GPU の heFFTe slab、倍精度、1 ノード 1 プロセス
#   int の上限（2048³ ≒ 86 億）を超える箇所が残っていないか、メモリが足りるかを見る
# 投入（ノード数・Np・FFT は -v で変えられる）:
#   qsub -l select=8:mpiprocs=72:ompthreads=1 -v N=8 jobs/miyabi/pm_ng2048_trial.sh
#   pencil にする: ... -v N=8,FFT=heffte-pencil-gpu ...
# 見る点: cat runs/ng2048_*/summary.txt、grep -h checksum runs/ng2048_*/*.log、ログの exit と OOM
. /work/gu99/u99001/Particle-Mesh/jobs/miyabi/pm_common.sh
N=${N:?N=ノード数 を指定}; NG=${NG:-2048}; NP=${NP:-1024}; FFT=${FFT:-heffte-slab-gpu}
OUT=$PM/runs/ng2048_n${N}_$(date +%Y%m%d_%H%M)_${PBS_JOBID%%.*}
mkdir -p $OUT
echo "# Ng=$NG Np=$NP nodes=$N fft=$FFT" >> $OUT/summary.txt

# pm_comm_opt.sh の pm_wrap_opt.sh と同じ設定（チェックサムと細かい計時を有効にする）
cat > ./pm_wrap_trial.sh <<'EOS'
#!/bin/bash
export OMP_NUM_THREADS=$1 OMP_PROC_BIND=close OMP_PLACES=cores OMP_STACKSIZE=64M
export PM_INTERP=$2 PM_DEPOSIT=$3 PM_CHECKSUM=1 PM_WISDOM_DIR=/work/gu99/u99001/Particle-Mesh
export PM_FINE_TIMER=1 PM_COMM_FLOAT=0
shift 3
ulimit -s unlimited
exec "$@"
EOS
chmod +x ./pm_wrap_trial.sh

LOG=$OUT/trial_ng${NG}_n${N}.log
mpiexec -n $N --map-by ppr:1:node:PE=72 --bind-to core \
  ./pm_wrap_trial.sh 72 gpu atomic ./pm_fft_test_kokkos \
  --ng=$NG --np=$NP --source=auto --fft=$FFT --method=2 $PM_EXTRA > $LOG 2>&1
echo "## trial Ng=$NG nodes=$N fft=$FFT exit=$?" >> $OUT/summary.txt
sed -n '/===== Timer Results =====/,/^=====/p' $LOG | grep -v "=====" >> $OUT/summary.txt
echo "summary: $OUT/summary.txt"

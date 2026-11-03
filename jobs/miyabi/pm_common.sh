# pm_cpu_tune.sh / pm_report.sh から読み込む共通部分（単体では実行しない）
module load nvidia/25.9 nv-hpcx/25.9
PM=/work/gu99/u99001/Particle-Mesh
cd $PM

# 計算ノード側で環境変数を設定するラッパー（mpiexec は export した変数を渡さないため）
#   pm_wrap.sh <OMP_NUM_THREADS> <PM_INTERP> <PM_DEPOSIT> <PM_CHECKSUM> <bin> <args...>
cat > ./pm_wrap.sh <<'EOS'
#!/bin/bash
export OMP_NUM_THREADS=$1 OMP_PROC_BIND=close OMP_PLACES=cores OMP_STACKSIZE=64M
export PM_INTERP=$2 PM_DEPOSIT=$3 PM_CHECKSUM=$4 PM_WISDOM_DIR=/work/gu99/u99001/Particle-Mesh
EOS
# 細かい計測（既定は無効）。ジョブスクリプトで export しておくと計算ノード側へ渡す
#   PM_FINE_TIMER=1        : FFTE の FFT / IFFT の中の通信時間を "sub FFT Comm" / "sub IFFT Comm" として出す
#   PM_HEFFTE_TRACE=<頭>   : heFFTe の内部の区間をランクごとのファイルに書く（heFFTe を tracing 付きでビルドしたときだけ）
echo "export PM_FINE_TIMER=${PM_FINE_TIMER:-0} PM_HEFFTE_TRACE=${PM_HEFFTE_TRACE:-}" >> ./pm_wrap.sh
cat >> ./pm_wrap.sh <<'EOS'
shift 4
ulimit -s unlimited
exec "$@"
EOS
chmod +x ./pm_wrap.sh

# 計測のぶれ対策: 既定で各区間を「反復ごとの値の中央値」で出す（1 回だけ遅い反復が混ざっても影響しない）
#   PM_EXTRA を空にすると従来どおり平均（--stat=avg）
PM_EXTRA=${PM_EXTRA---stat=median}

# run <tag> <nodes> <procs/node> <bin> <interp> <deposit> <fft> <ng> <np>
#   1 ノードの 72 コアを procs/node 個に等分し、各プロセスをその範囲のコアに固定する
run () {
  TAG=$1; N=$2; P=$3; BIN=$4; INTERP=$5; DEP=$6; FFT=$7; NG=$8; NP=$9
  T=$((72 / P))
  LOG=$OUT/${TAG}_ng${NG}_n${N}_p${P}.log
  mpiexec -n $((N * P)) --map-by ppr:${P}:node:PE=${T} --bind-to core \
    ./pm_wrap.sh $T $INTERP $DEP 0 $BIN --ng=$NG --np=$NP --source=auto --fft=$FFT --method=2 $PM_EXTRA > $LOG 2>&1
  echo "## $TAG Ng=$NG nodes=$N ppn=$P fft=$FFT exit=$?" >> $OUT/summary.txt
  # Total の後ろの "sub ..."（PM_FINE_TIMER=1 の内訳）も拾うため、閉じの ===== まで取る
  sed -n '/===== Timer Results =====/,/^=====/p' $LOG | grep -v "=====" >> $OUT/summary.txt
}

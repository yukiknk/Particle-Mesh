#!/usr/bin/env python3
"""PM_HEFFTE_TRACE で書かれた heFFTe の区間記録（ランクごとの .log）を区間名ごとにまとめる。

使い方:
  python3 jobs/miyabi/heffte_trace_summary.py <頭>_slab_1 [--iters=12]

  <頭>_slab_1_<rank>.log をすべて読み、区間名ごとに「ランクごとの合計時間」の平均と最大、
  1 ランクあたりの回数を出す。--iters を渡すと 1 反復あたりの値も出す（既定 12 = 暖機 2 + 計測 10）。
  通し番号 0 は暖機用のオブジェクト、1 以降が計測用（--fft で選んだ順）。

注意:
  - heFFTe 本体を -DHeffte_ENABLE_TRACING=ON でビルドしていないとファイルはできない
  - 暖機の反復も含む。GPU のカーネルは非同期なので、区間の境目で同期していない処理は後ろの区間に入ることがある
"""
import glob
import re
import statistics
import sys
from collections import defaultdict


def main(args):
    iters = 12
    roots = []
    for a in args:
        if a.startswith("--iters="):
            iters = int(a.split("=", 1)[1])
        else:
            roots.append(a)
    if not roots:
        print(__doc__)
        sys.exit(1)
    for root in roots:
        files = sorted(glob.glob(root + "_*.log"), key=lambda f: int(re.findall(r"_(\d+)\.log$", f)[0]))
        if not files:
            print(f"# {root}: ファイルがない")
            continue
        per_rank = []   # ランクごとに {区間名: [合計, 回数]}
        for fn in files:
            d = defaultdict(lambda: [0.0, 0])
            for line in open(fn):
                tok = line.split()
                if len(tok) < 3:
                    continue
                try:
                    dur = float(tok[-1])
                    float(tok[-2])
                except ValueError:
                    continue
                name = " ".join(tok[:-2])
                d[name][0] += dur
                d[name][1] += 1
            per_rank.append(d)
        names = sorted({n for d in per_rank for n in d}, key=lambda n: -statistics.mean(d[n][0] for d in per_rank))
        print(f"# {root}: {len(files)} ranks, iters={iters}")
        print(f"{'区間':<32} {'平均[s]':>10} {'最大[s]':>10} {'平均/反復[ms]':>14} {'回数/ランク':>10}")
        for n in names:
            tot = [d[n][0] if n in d else 0.0 for d in per_rank]
            cnt = [d[n][1] if n in d else 0 for d in per_rank]
            mean = statistics.mean(tot)
            print(f"{n:<32} {mean:10.4f} {max(tot):10.4f} {mean / iters * 1e3:14.3f} {statistics.median(cnt):10.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])

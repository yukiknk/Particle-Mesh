#!/usr/bin/env python3
"""複数回の pm_report / pm_cpu_tune の summary.txt をまとめ、回ごとの中央値で 1 つの summary を作る。

使い方:
  python3 jobs/miyabi/agg_summary.py runs/report_ng512_*/summary.txt > runs/report_ng512_median.txt

出力は summary.txt と同じ形式（各区間の値 = 各回の値の中央値）。
最後に「回ごとの Total のばらつき」（最小 / 中央値 / 最大）を # 行で出す。最大/最小 が 1.3 を超える条件には ! を付ける。
"""
import re
import statistics
import sys

HEAD = re.compile(r"## (\S+) Ng=(\d+) nodes=(\d+) ppn=(\d+) fft=(\S+) exit=(\d+)")
LINE = re.compile(r"(.+?)\s*:\s*([\d.]+) sec")


def parse(fn):
    runs, order, cur = {}, [], None
    for line in open(fn):
        m = HEAD.match(line)
        if m:
            cur = m.groups()[:5]
            if cur not in runs:
                order.append(cur)
            runs[cur] = {"exit": int(m[6]), "t": {}}
            continue
        m = LINE.match(line)
        if m and cur:
            runs[cur]["t"][m[1].strip()] = float(m[2])
    return runs, order


def main(files):
    if not files:
        print(__doc__)
        sys.exit(1)
    allruns, order = [], []
    for f in files:
        r, o = parse(f)
        allruns.append(r)
        for k in o:
            if k not in order:
                order.append(k)
    print(f"# median of {len(files)} runs: " + " ".join(files))
    spread = []
    for k in order:
        ok = [r[k]["t"] for r in allruns if k in r and r[k]["exit"] == 0 and r[k]["t"]]
        tag, ng, nodes, ppn, fft = k
        print(f"## {tag} Ng={ng} nodes={nodes} ppn={ppn} fft={fft} exit={0 if ok else 1} n={len(ok)}")
        if not ok:
            continue
        names = [n for n in ok[0] if n != "Total"]
        for n in names:
            v = statistics.median([t.get(n, 0.0) for t in ok])
            print(f"{n:<20}: {v:12.6f} sec (median)")
        tot = sum(statistics.median([t.get(n, 0.0) for t in ok]) for n in names)
        print("-------------------------")
        print(f"{'Total':<20}: {tot:12.6f} sec (median)")
        ts = [t["Total"] for t in ok if "Total" in t]
        if ts:
            spread.append((k, min(ts), statistics.median(ts), max(ts)))
    print("# Total per run: min / median / max  (! = max/min > 1.3)")
    for (tag, ng, nodes, ppn, fft), lo, md, hi in spread:
        flag = "!" if lo > 0 and hi / lo > 1.3 else " "
        print(f"# {flag} {tag:<11} Ng={ng} nodes={nodes:>2} ppn={ppn}  {lo:.4f} / {md:.4f} / {hi:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])

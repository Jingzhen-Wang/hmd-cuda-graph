"""
plot_sweep.py -- parse benchmarks/sweep/sweep.out (produced by run_sweep.sl)
and plot naive-GPU (Phase 2) vs CUDA-Graph (Phase 3) compute_accel() time
against system size, to show where graph capture/replay pays off.

Usage (from the dataviz-matched venv set up for this project):
  python3 benchmarks/sweep/plot_sweep.py [sweep.out] [--out FILE.png]
"""
import argparse
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZE_RE    = re.compile(r"^##### size=(\w+) #####")
VARIANT_RE = re.compile(r"^--- (hmd_gpu|hmd_graph) ")
NGLOB_RE   = re.compile(r"^nglob\s*=\s*(\d+)")
TIME_RE    = re.compile(r"^CPU & COMT\s*=\s*([\d.eE+-]+)\s+([\d.eE+-]+)")

VARIANT_LABEL = {"hmd_gpu": "GPU, naive (Phase 2)", "hmd_graph": "GPU + CUDA Graph (Phase 3)"}
VARIANT_COLOR = {"hmd_gpu": "#eb6834", "hmd_graph": "#1baf7a"}


def parse(path):
    """Returns {size_name: {variant: (nglob, compute_time)}}"""
    data = {}
    size = variant = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = SIZE_RE.match(line)
            if m:
                size = m.group(1)
                data.setdefault(size, {})
                variant = None
                continue
            m = VARIANT_RE.match(line)
            if m:
                variant = m.group(1)
                continue
            m = NGLOB_RE.match(line)
            if m and size is not None:
                data[size].setdefault(variant, [None, None])
                data[size][variant][0] = int(m.group(1))
                continue
            m = TIME_RE.match(line)
            if m and size is not None and variant is not None:
                data[size].setdefault(variant, [None, None])
                data[size][variant][1] = float(m.group(1))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_out", nargs="?", default="benchmarks/sweep/sweep.out")
    ap.add_argument("--out", default="benchmarks/sweep_timing.png")
    args = ap.parse_args()

    data = parse(args.sweep_out)
    if not data:
        sys.exit(f"no sweep data parsed from {args.sweep_out}")

    # Preserve the size order the sweep actually ran in (tiny->large),
    # falling back to sorting by nglob if some are missing.
    order = [s for s in ("tiny", "small", "medium", "large") if s in data]
    nglobs = [data[s]["hmd_gpu"][0] for s in order]

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for variant in ("hmd_gpu", "hmd_graph"):
        ys = [data[s][variant][1] for s in order]
        ax.plot(nglobs, ys, marker="o", markersize=7, linewidth=2.2,
                 color=VARIANT_COLOR[variant], label=VARIANT_LABEL[variant], zorder=3)
        for x, y in zip(nglobs, ys):
            ax.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5, color="#0b0b0b")

    ax.set_xscale("log")
    ax.set_xlabel("System size, nglob (atoms, log scale)", fontsize=10.5, color="#0b0b0b")
    ax.set_ylabel("compute_accel() time, CPU & COMT (s)", fontsize=10.5, color="#0b0b0b")
    ax.set_title("Naive GPU vs. CUDA Graph across system size\n(300 steps, nproc=2, nthrd=4, num_device=2)",
                 fontsize=11, color="#0b0b0b", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9.5)
    ax.yaxis.grid(True, color="#e3e2db", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(f"saved {args.out}")
    for s in order:
        g, gt = data[s]["hmd_gpu"]
        _, pt = data[s]["hmd_graph"]
        speedup = gt / pt if pt else float("nan")
        print(f"  {s:7s} nglob={g:6d}  naive={gt:7.3f}s  graph={pt:7.3f}s  graph/naive speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()

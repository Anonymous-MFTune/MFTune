import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

# ================================
# Config
# ================================
CSV_FILE = "discussion/mysql/result-4.csv"
FIDELITY_COL = "fidelity"
THROUGHPUT_COL = "throughput"

HIGH_FID_VALUE = 180           # fidelity[0] == 180 represents high-fidelity
TOP_K = 20                     # top-K
PHASE1_CONSEC_HIGH = 17        # consecutive high-fidelity as a flag-> phase 1 end
PRINT_STEP = 20                # every how many evaluation index to print
# ================================


def parse_fidelity(x):
    """transform string like '[30, 20, 45000, 4, 0.5]' to list"""
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return ast.literal_eval(str(x))
    except Exception:
        raise ValueError(f"cannot parse fidelity value: {x!r}")


def is_high_fidelity(flist):
    """judge if fidelity list represents high-fidelity"""
    return len(flist) > 0 and flist[0] == HIGH_FID_VALUE


def main():
    df = pd.read_csv(CSV_FILE)
    if FIDELITY_COL not in df.columns or THROUGHPUT_COL not in df.columns:
        raise KeyError(f"数据必须包含列: {FIDELITY_COL!r} 和 {THROUGHPUT_COL!r}")

    df[FIDELITY_COL] = df[FIDELITY_COL].apply(parse_fidelity)
    df[THROUGHPUT_COL] = df[THROUGHPUT_COL].astype(float)

    fids = df[FIDELITY_COL].tolist()
    thps = df[THROUGHPUT_COL].tolist()
    total_len = len(df)

    low_archive = []            # all throughput in low-fidelity
    high_archive = []           # all throughput in high-fidelity

    # top-20 best-so-far sequences
    low_topk_avg_seq = []       # low fidelity top-20 avg (best-so-far)
    low_topk_std_seq = []       # corresponding std
    high_topk_avg_seq = []      # high fidelity top-20 avg (best-so-far)
    high_topk_std_seq = []      # corresponding std

    best_low_avg_so_far = 0.0
    best_low_std_so_far = 0.0
    best_high_avg_so_far = 0.0
    best_high_std_so_far = 0.0

    consec_high = 0
    phase1_end_idx = None       # Phase 1 endpoint（1-based）

    for idx in range(total_len):  # idx: 0-based
        fidelity = fids[idx]
        thp = thps[idx]

        if is_high_fidelity(fidelity):
            # high-fidelity
            consec_high += 1
            high_archive.append(thp)
            if phase1_end_idx is None and consec_high >= PHASE1_CONSEC_HIGH:
                phase1_end_idx = idx + 1
        else:
            # low-fidelity
            consec_high = 0
            low_archive.append(thp)

        # ----- update low-fidelity top-20 of the archive -----
        if len(low_archive) == 0:
            current_low_avg = 0.0
            current_low_std = 0.0
        else:
            sorted_low = sorted(low_archive, reverse=True)
            low_top_vals = sorted_low[:TOP_K]
            current_low_avg = float(np.mean(low_top_vals))
            current_low_std = float(np.std(low_top_vals, ddof=0))

        if current_low_avg > best_low_avg_so_far:
            best_low_avg_so_far = current_low_avg
            best_low_std_so_far = current_low_std

        low_topk_avg_seq.append(best_low_avg_so_far)
        low_topk_std_seq.append(best_low_std_so_far)

        # ----- update high-fidelity archive's top-20 -----
        if len(high_archive) == 0:
            current_high_avg = 0.0
            current_high_std = 0.0
        else:
            sorted_high = sorted(high_archive, reverse=True)
            high_top_vals = sorted_high[:TOP_K]
            current_high_avg = float(np.mean(high_top_vals))
            current_high_std = float(np.std(high_top_vals, ddof=0))

        if current_high_avg > best_high_avg_so_far:
            best_high_avg_so_far = current_high_avg
            best_high_std_so_far = current_high_std

        high_topk_avg_seq.append(best_high_avg_so_far)
        high_topk_std_seq.append(best_high_std_so_far)

    if phase1_end_idx is None:
        phase1_end_idx = total_len

    # get global start idx where both low_top20>0 and high_top20>0
    global_start_idx = None
    for i in range(1, total_len + 1):
        if low_topk_avg_seq[i-1] > 0 and high_topk_avg_seq[i-1] > 0:
            global_start_idx = i
            break

    if global_start_idx is None:
        print("警告：整个过程中没有 low_top20>0 且 high_top20>0 的点，无法打印。")
        return

    # ========================= Tikz format =========================
    xs = list(range(1, total_len + 1))
    low_avg = np.array(low_topk_avg_seq)
    low_std = np.array(low_topk_std_seq)
    high_avg = np.array(high_topk_avg_seq)
    high_std = np.array(high_topk_std_seq)

    sample_idx_set = set()
    i = global_start_idx
    while i <= total_len:
        sample_idx_set.add(i)
        i += PRINT_STEP
    sample_idx_set.add(phase1_end_idx)
    sample_idx_set.add(total_len)
    sample_indices = sorted(sample_idx_set)

    plot_indices = list(range(1, len(sample_indices) + 1))

    print("\n================ TikZ SAMPLES ================")
    print(f"% PHASE1_END = {phase1_end_idx}")
    print(f"% REAL sampled indices = {sample_indices}")
    print(f"% PLOT indices        = {plot_indices}\n")

    print("REAL_IDX_PLOT_IDX = {")
    for real_i, plot_i in zip(sample_indices, plot_indices):
        print(f"({real_i}, {plot_i})")
    print("};\n")

    def print_coords(name, ys, use_plot_idx=False):
        print(f"{name} = {{")
        for real_i, plot_i in zip(sample_indices, plot_indices):
            idx = real_i - 1
            x = plot_i if use_plot_idx else real_i
            print(f"({x}, {ys[idx]:.4f})")
        print("};\n")

    print_coords("LOW_MEAN_REAL", low_avg, use_plot_idx=False)
    print_coords("LOW_UPPER_REAL", low_avg + low_std, use_plot_idx=False)
    print_coords("LOW_LOWER_REAL", low_avg - low_std, use_plot_idx=False)

    print_coords("HIGH_MEAN_REAL", high_avg, use_plot_idx=False)
    print_coords("HIGH_UPPER_REAL", high_avg + high_std, use_plot_idx=False)
    print_coords("HIGH_LOWER_REAL", high_avg - high_std, use_plot_idx=False)

    print_coords("LOW_MEAN_PLOT", low_avg, use_plot_idx=True)
    print_coords("LOW_UPPER_PLOT", low_avg + low_std, use_plot_idx=True)
    print_coords("LOW_LOWER_PLOT", low_avg - low_std, use_plot_idx=True)

    print_coords("HIGH_MEAN_PLOT", high_avg, use_plot_idx=True)
    print_coords("HIGH_UPPER_PLOT", high_avg + high_std, use_plot_idx=True)
    print_coords("HIGH_LOWER_PLOT", high_avg - high_std, use_plot_idx=True)

    print("============== END TikZ ==============\n")


    iters = xs

    plt.figure(figsize=(10, 4))
    plt.plot(iters, low_avg, label="Low-fidelity top-20 avg (best-so-far)",
             marker='o', linewidth=1, color='tab:blue')
    plt.fill_between(iters, low_avg - low_std, low_avg + low_std,
                     color='tab:blue', alpha=0.15, linewidth=0)
    plt.plot(iters, high_avg, label="High-fidelity top-20 avg (best-so-far)",
             marker='*', linewidth=1, color='tab:orange')
    plt.fill_between(iters, high_avg - high_std,
                     high_avg + high_std,
                     color='tab:orange', alpha=0.15, linewidth=0)

    plt.axvline(x=phase1_end_idx, color='gray', linestyle='--', alpha=0.6,
                label=f"Phase 1 end (idx={phase1_end_idx})")

    plt.xlabel("Evaluation index")
    plt.ylabel("Throughput")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter


"""
Code for discussion: How does EvoFD Help? 
- iterate results/<system>/MFTune-a5/run_*/fidelity_pop_gen_*.csv
- spearman_corr as X axis，evaluated_time as Y axis, red stars indicate fidelity setting
- if front_level is 0, plot as o 
- save each generation as a PDF format -> evo_process/<system>/MFTune-a5/<run_name>/gen_*.pdf
"""


RESULTS_ROOT = "results"
OUT_ROOT = "discussion/fidelity_evo_process"

STAR_SIZE = 450
CIRCLE_SIZE = 650
CIRCLE_LINEWIDTH = 2.5
ALPHA_POINTS = 0.9

CSV_PATTERN = "fidelity_pop_gen_*.csv"

# ---------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # remove whitespace around column names and convert to lowercase
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def extract_run_label(run_dirname: str) -> str:
    # accommodate various run naming conventions
    # extract run label from directory name
    m = re.search(r"run[_-]?(\d+)", run_dirname)
    if m:
        return f"run{m.group(1)}"
    return run_dirname

def plot_generation(df: pd.DataFrame, system: str, run_label: str, gen_name: str, out_path: str):
    df = clean_columns(df)

    col_x = next((c for c in df.columns if "spearman" in c), None)
    col_y = next((c for c in df.columns if "evaluated" in c), None)
    col_front = next((c for c in df.columns if "front" in c), None)

    if col_x is None or col_y is None:
        print(f"[skip] {system}/{run_label}/{gen_name}: lack of spearman_corr or evaluated_time column")
        return

    df = coerce_numeric(df, [col_x, col_y])
    if col_front:
        df[col_front] = pd.to_numeric(df[col_front], errors="coerce")

    df = df.dropna(subset=[col_x, col_y])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    plt.scatter(df[col_x], df[col_y], marker='*', color='red', s=STAR_SIZE, alpha=ALPHA_POINTS, label='population')

    # only highlight the points with front_level == 0
    if col_front and (df[col_front] == 0).any():
        sub = df[df[col_front] == 0]
        plt.scatter(
            sub[col_x], sub[col_y],
            marker='o', s=CIRCLE_SIZE,
            facecolors='none', edgecolors='teal', linewidths=CIRCLE_LINEWIDTH,
            label='front_level=0'
        )

    plt.xlabel("Fidelity Level", fontsize=32)
    plt.ylabel("Evaluation Cost (s)", fontsize=32)
    # plt.ylim(35, 75)
    # plt.xlim(0.6, 1.0)
    ax.xaxis.offsetText.set_fontsize(24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)


    # plt.title(f"{system} | {run_label} | {gen_name}")
    # plt.grid(True, linestyle='--', alpha=0.3)
    # plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))

    '''----------optimize the plot--------'''
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # If you want to keep only the "arrow axes", hide the bottom and left spines as well
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Keep the ticks on the bottom/left side
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # use arrows to draw X/Y axes (in axis coordinates: bottom-left (0,0) to top-right (1,1))
    ax.annotate("", xy=(1, 0), xytext=(0, 0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0, 1), xytext=(0, 0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.5))

    plt.savefig(out_path, format="pdf")
    plt.close()


def main():
    if not os.path.isdir(RESULTS_ROOT):
        print(f"[Error] without found path: {RESULTS_ROOT} (make sure place the script in the same level of results)")
        return

    systems = sorted([d for d in os.listdir(RESULTS_ROOT)
                      if os.path.isdir(os.path.join(RESULTS_ROOT, d))])

    total_plots = 0

    for system in systems:
        algo_dir = os.path.join(RESULTS_ROOT, system, "MFTune-a5")
        if not os.path.isdir(algo_dir):
            continue

        run_dirs = sorted([d for d in os.listdir(algo_dir)
                           if os.path.isdir(os.path.join(algo_dir, d)) and d.startswith("run")])
        if not run_dirs:
            continue

        for run_dir in run_dirs:
            run_path = os.path.join(algo_dir, run_dir)
            run_label = extract_run_label(run_dir)

            # all csv files for this run
            csv_paths = sorted(glob.glob(os.path.join(run_path, CSV_PATTERN)))
            if not csv_paths:
                print(f"[Tips] No CSV found: {system}/{run_label}")
                continue

            out_dir = os.path.join(OUT_ROOT, system, "MFTune-a5", run_label)
            ensure_dir(out_dir)

            for csv_file in csv_paths:
                gen_name = os.path.splitext(os.path.basename(csv_file))[0]  # fidelity_pop_gen_0
                out_pdf = os.path.join(out_dir, f"{gen_name}.pdf")

                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"[Error] invalid CSV file {csv_file}: {e}")
                    continue

                plot_generation(df, system, run_label, gen_name, out_pdf)
                total_plots += 1
                print(f"[Finish] save: {out_pdf}")

    if total_plots == 0:
        print("[Finish] without any plot generated. Check if the results directory is structured correctly.")
    else:
        print(f"[Finish] All done! {total_plots} plots generated.")


if __name__ == "__main__":
    main()

import os
import ast
import numpy as np
import pandas as pd

# ---------------------- Config ----------------------
HIGH_FIDELITY_DIC = {
    'mysql':      [180, 50, 100000, 4, 0.5],
    'postgresql': [180, 50, 100000, 4, 0.5],
    'httpd':      [8, 50, 180, True],
    'tomcat':     [8, 50, 180, True],
    'gcc':        [10, 50, 6, 4, 50],
    'clang':      [10, 50, 6, 4, 50],
}

TARGET_DIC = {
    'mysql':      'throughput',
    'postgresql': 'throughput',
    'httpd':      'RPS',
    'tomcat':     'RPS',
    'gcc':        'run_time',
    'clang':      'run_time',
}

BUDGET_DIC = {
    'mysql':      86400,
    'postgresql': 86400,
    'httpd':      43200,
    'tomcat':     43200,
    'gcc':        14400,
    'clang':      14400,
}


ALGO_FILES = {
    'hyperband':  'HBTuner_results.csv',
    'bohb':       'BOHBTuner_results.csv',
    'dehb': 'DEHBTuner_results.csv',
    'priorband': 'PriorBand_results.csv',
    'smac': 'SMACTuner_results.csv',
    'bestconfig': 'BestConfigTuner_results.csv',
    'flash': 'FLASHTuner_results.csv',
    'ga':         'GATuner_results.csv',
    'hebo': 'HEBOTuner_results.csv',
    'promise': 'PromiseTuner_results.csv',
    'MFTune-I' :  'GATuner_results.csv',
    'MFTune-II':  'GATuner_results.csv',
    'MFTune-a5':  'GATuner_results.csv',

}

BASE_DIR = './results'
SYSTEMS = ['mysql', 'postgresql', 'httpd', 'tomcat', 'gcc', 'clang']

MF_NAME = 'MFTune-a5'
OUTPUT_CSV = 'RQ2/time_saving.csv'


# ---------------------- Utils ----------------------
def is_minimization(system: str) -> bool:
    return system in ['gcc', 'clang']


def best_so_far(seq, minimize: bool):
    out = []
    cur = None
    for v in seq:
        if cur is None:
            cur = v
        else:
            cur = min(cur, v) if minimize else max(cur, v)
        out.append(cur)
    return out


def pad_last(values, target_len):
    vals = list(values)
    if len(vals) == 0:
        return [np.nan] * target_len
    while len(vals) < target_len:
        vals.append(vals[-1])
    return vals


def pad_cost_like(costs, target_len, default_interval=30.0):
    cs = list(costs)
    if len(cs) == 0:
        return [0.0 + i * default_interval for i in range(target_len)]
    if len(cs) >= 2:
        interval = cs[-1] - cs[-2]
        if interval <= 0:
            interval = default_interval
    else:
        interval = default_interval
    while len(cs) < target_len:
        cs.append(cs[-1] + interval)
    return cs


def truncate_by_budget(costs, perfs, budget):
    costs = np.asarray(costs, dtype=float)
    perfs = np.asarray(perfs, dtype=float)
    mask = costs <= budget
    if not np.any(mask):
        # Fallback: keep the first point if everything exceeds the budget
        return costs[:1], perfs[:1]
    return costs[mask], perfs[mask]


def list_runs(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        d for d in os.listdir(path)
        if d.startswith('run_') and os.path.isdir(os.path.join(path, d))
    )


# ---------------------- Curve builder (generic for all algos) ----------------------
def build_avg_curve(system, algo, filename, target, high_fidelity):
    algo_dir = os.path.join(BASE_DIR, system, algo)
    if not os.path.isdir(algo_dir):
        return None

    minimize = is_minimization(system)
    all_costs, all_perfs = [], []
    max_len = 0

    for run in list_runs(algo_dir):
        f = os.path.join(algo_dir, run, filename)
        if not os.path.isfile(f):
            continue

        df = pd.read_csv(f)
        if 'cost' not in df.columns or target not in df.columns:
            continue

        cum_cost = df['cost'].cumsum()
        perf = df[target]

        if 'fidelity' in df.columns:
            fid = df['fidelity'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            hf_list = list(high_fidelity)
            hf_mask = fid.apply(lambda x: list(x) == hf_list)

            if not hf_mask.any():
                # 该 run 没有高保真数据，跳过
                continue

            cum_cost = cum_cost.loc[hf_mask]
            perf = perf.loc[hf_mask]

        if len(cum_cost) == 0:
            continue

        # best-so-far
        perf_bs = best_so_far(perf.to_list(), minimize=minimize)

        all_costs.append(cum_cost.to_list())
        all_perfs.append(perf_bs)
        max_len = max(max_len, len(cum_cost))

    if len(all_costs) == 0:
        return None

    # padding
    all_costs = [pad_cost_like(c, max_len) for c in all_costs]
    all_perfs = [pad_last(p, max_len) for p in all_perfs]

    mean_cost = np.mean(np.array(all_costs, dtype=float), axis=0)
    mean_perf = np.mean(np.array(all_perfs, dtype=float), axis=0)
    return mean_cost, mean_perf


# ---------------------- Speedup helper ----------------------
def first_time_to_reach(costs, perfs, target_value, minimize):
    costs = np.asarray(costs, dtype=float)
    perfs = np.asarray(perfs, dtype=float)

    if minimize:
        idx = np.where(perfs <= target_value)[0]
    else:
        idx = np.where(perfs >= target_value)[0]

    if idx.size == 0:
        return np.nan
    return float(costs[idx[0]])


# ---------------------- Main ----------------------
def main():
    algo_savings = {
        algo: {} for algo in ALGO_FILES.keys() if algo != MF_NAME
    }

    for system in SYSTEMS:
        target = TARGET_DIC[system]
        budget = BUDGET_DIC[system]
        hf = HIGH_FIDELITY_DIC[system]
        minimize = is_minimization(system)

        print("=" * 80)
        print(f"System: {system} | Objective: {target} | Mode: {'min' if minimize else 'max'}")
        print(f"Budget for truncation: {budget} s\n")

        algo_results = {}

        for algo, filename in ALGO_FILES.items():
            res = build_avg_curve(system, algo, filename, target, hf)
            if res is None:
                print(f"[{algo}] No valid data found for system '{system}'.")
                continue

            cost_raw, perf_raw = res
            cost_tr, perf_tr = truncate_by_budget(cost_raw, perf_raw, budget)

            algo_results[algo] = {
                'cost_raw': cost_raw,
                'perf_raw': perf_raw,
                'cost_tr': cost_tr,
                'perf_tr': perf_tr,
                'final_perf_tr': float(perf_tr[-1]),
            }

        if MF_NAME not in algo_results:
            print(f"No {MF_NAME} data for system '{system}', skip this system.\n")
            continue

        if len(algo_results) <= 1:
            print("Not enough algorithms with data for this system (need >= 2 including MFTune-a5).\n")
            continue

        mf_res = algo_results[MF_NAME]
        mf_cost_tr = mf_res['cost_tr']
        mf_perf_tr = mf_res['perf_tr']

        for baseline_algo, baseline_res in algo_results.items():
            if baseline_algo == MF_NAME:
                continue

            baseline_cost_tr = baseline_res['cost_tr']
            baseline_perf_tr = baseline_res['perf_tr']
            baseline_final_perf = baseline_res['final_perf_tr']

            # baseline reaches its own best performance
            t_baseline = first_time_to_reach(
                baseline_cost_tr,
                baseline_perf_tr,
                baseline_final_perf,
                minimize=minimize
            )

            # MFTune reaches the best performance of baseline
            t_mf = first_time_to_reach(
                mf_cost_tr,
                mf_perf_tr,
                baseline_final_perf,
                minimize=minimize
            )

            if np.isnan(t_baseline) or np.isnan(t_mf):
                saving = np.nan
                print(f"  Baseline {baseline_algo}: some curve did not reach target (saving = NaN).")
            else:
                saving = round((t_baseline - t_mf) / 3600, 1)
                print(f"  Baseline {baseline_algo}: t_baseline={t_baseline/3600:.3f}h, "
                      f"t_{MF_NAME}={t_mf/3600:.3f}h, saving={saving:.3f}h")

            if baseline_algo in algo_savings:
                algo_savings[baseline_algo][system] = saving

        print()

    rows = []
    for algo in ALGO_FILES.keys():
        if algo == MF_NAME:
            continue
        row = {'algo': algo}
        for system in SYSTEMS:
            row[system] = algo_savings.get(algo, {}).get(system, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows, columns=['algo'] + SYSTEMS)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

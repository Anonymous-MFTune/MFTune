import os
import numpy as np
# from rpy2.robjects import r, pandas2ri
# pandas2ri.activate()
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
from rpy2.robjects.packages import importr
from collections import Counter
import ast
from utils.utils import PerfDataUtils

# devtools = importr("devtools")
# devtools.install_github("klainfo/ScottKnottESD", ref="development")
sk = importr('ScottKnottESD')

HIGH_FIDELITY_DIC = {
    'mysql': [180, 50, 100000, 4, 0.5],
    'postgresql': [180, 50, 100000, 4, 0.5],
    'httpd': [8, 50, 180, True],
    'tomcat': [8, 50, 180, True],
    'gcc': [10, 50, 6, 4, 50],
    'clang': [10, 50, 6, 4, 50],
    'x264': ['Action', '1080p', 60, '20s'],
}
TARGET_DIC = {
    'mysql': 'throughput',
    'postgresql': 'throughput',
    'httpd': 'RPS',
    'tomcat': 'RPS',
    'gcc': 'run_time',
    'clang': 'run_time',
    'x264': 'run_time',
}


def calculate_and_save_scott_knott(system, all_results, save_path='RQ1/scott_knott.csv'):
    algorithms = list(all_results.keys())
    performance_df = pd.DataFrame(all_results)
    with localconverter(pandas2ri.converter):
        r_df = pandas2ri.py2rpy(performance_df)

    # Scott-Knott ranks
    r_sk = sk.sk_esd(r_df, version='p')
    # Scott-Knott rankings
    # r_sk = sk.sk_esd(performance_df, version='p')
    column_order = [i - 1 for i in r_sk[3]]

    if system in ['gcc', 'clang', 'x264']:
        # as for minimization problem, we need to reverse the ranking
        column_order = [i - 1 for i in r_sk[3]][::-1]  # Reverse the rankings and change to minimize

    ranking_results = pd.DataFrame(
        {
            "Algorithms": [performance_df.columns[i] for i in column_order],
            "rankings": list(map(int, r_sk[1]))
        }
    ).set_index("Algorithms")

    # save ranking and mean value, make sure direct to corresponding algorithms
    results_to_save = pd.DataFrame(index=algorithms)

    # results_to_save[(system, 'r')] = ranking_results.loc[algorithms, 'rankings']
    # results_to_save[(system, 'mean')] = performance_df.mean().round(3)

    # save r in "[x]" format and mean in "mean (std)" format
    r_values = ranking_results.loc[algorithms, 'rankings']
    results_to_save[(system, 'r')] = r_values.apply(lambda x: f"[{x}]")

    # calculate mean and std and save as "mean (std)" format
    if system in ['gcc', 'clang']:
        means = (1000 * performance_df.mean()).round(2)
        stds = (1000 * performance_df.std()).round(2)
    else:
        means = performance_df.mean().round(2)
        stds = performance_df.std().round(2)
    mean_std = means.astype(str) + " (" + stds.astype(str) + ")"
    results_to_save[(system, 'mean')] = mean_std

    # calculate the improvement of mftune compared to other algorithms
    mf_algo = 'MFTune-a5'
    if mf_algo in performance_df.columns:
        baseline_mean = performance_df[mf_algo].mean()
        if system in ['mysql', 'postgresql', 'tomcat', 'httpd']:
            improvements = ((baseline_mean - performance_df.mean()) / performance_df.mean() * 100).round(2)
        elif system in ['gcc', 'clang', 'x264']:
            improvements = ((performance_df.mean() - baseline_mean) / performance_df.mean() * 100).round(2)
        # results_to_save[(system, f'Improvement (%)')] = improvements.values
        results_to_save[(system, 'Improvement (%)')] = improvements.reindex(results_to_save.index)

    else:
        results_to_save[(system, f'Improvement (%)')] = np.nan

    # set MultiIndex
    results_to_save.columns = pd.MultiIndex.from_tuples(results_to_save.columns)

    # if file exist, combine with current results
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path, header=[0, 1], index_col=0)
        if system in existing_df.columns.levels[0]:
            existing_df = existing_df.drop(columns=system, level=0)
        combined_df = existing_df.join(results_to_save, how='outer')
    else:
        combined_df = results_to_save

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # let's ensure the column order is consistent with systems processed
    combined_df = combined_df.reindex(index=algorithms)
    combined_df.to_csv(save_path)



def append_all_system_summary(save_path='RQ2/scott_knott.csv', rank_decimals=2):
    """
    calculate the average ranking across all systems
    and append two columns ('all_system','r') and ('all_system','mean').
    'r' in [x.xx] format; 'mean' in '—'.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Not found: {save_path}")

    df = pd.read_csv(save_path, header=[0, 1], index_col=0)

    # 1. calculate average ranking
    systems_all = []
    for s, m in df.columns:
        if s not in systems_all:
            systems_all.append(s)
    # construct the order of systems excluding 'all_system' (keep the original order, not alphabetical)
    systems_no_all = [s for s in systems_all if s != 'all_system']

    rank_cols = [(s, 'r') for s in systems_no_all if (s, 'r') in df.columns]
    ranks_num = df[rank_cols].applymap(lambda v: pd.to_numeric(str(v).strip().strip('[]'), errors='coerce'))
    avg_rank = ranks_num.mean(axis=1).round(rank_decimals).apply(lambda x: f"[{x}]" if pd.notnull(x) else "")

    # 2) add two columns for average ranking of all_system
    df[('all_system', 'r')] = avg_rank
    df[('all_system', 'mean')] = '—'

    # 3) reorder columns: systems_no_all + all_system
    metrics_priority = ['r', 'mean', 'Improvement (%)']
    ordered_systems = systems_no_all + ['all_system']

    new_cols = []
    for s in ordered_systems:
        for m in metrics_priority:
            if (s, m) in df.columns:
                if s == 'all_system' and m == 'Improvement (%)':
                    continue
                new_cols.append((s, m))

    df = df.reindex(columns=pd.MultiIndex.from_tuples(new_cols))
    df.to_csv(save_path)



def make_wide_layout(input_csv='RQ2/scott_knott.csv',
                     output_csv='RQ2/scott_knott_wide.csv',
                     metric_order=('r', 'mean')):
    """
    convert the (columns=system×metric, index=algorithm) table
    to (index=system, columns=algorithm×metric) wide table.
    keep the system order consistent with the original CSV, not alphabetical.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Not found: {input_csv}")

    df = pd.read_csv(input_csv, header=[0, 1], index_col=0)

    # record original system order
    system_order = []
    for s, m in df.columns:
        if s not in system_order:
            system_order.append(s)

    stacked = df.stack(level=0)
    stacked.index.set_names(['Algorithms', 'System'], inplace=True)

    stacked = stacked.swaplevel(0, 1)
    wide = stacked.unstack(level=1)

    # make sure the column order is (algorithm, r/mean)
    if wide.columns.nlevels == 2:
        if wide.columns.names == [None, 'Algorithms'] or set(metric_order).issubset(set(wide.columns.levels[0])):
            wide.columns = wide.columns.swaplevel(0, 1)
        algos = list(wide.columns.levels[0])
        new_cols = []
        for a in algos:
            for m in metric_order:
                if (a, m) in wide.columns:
                    new_cols.append((a, m))
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(new_cols))

    # reorder rows to match original system order
    wide = wide.reindex(system_order)
    wide.index.name = 'System'
    wide.to_csv(output_csv)

    return wide



def main():

    systems = ['httpd', 'tomcat', 'gcc', 'clang']
    # systems = ['mysql', 'tomcat', 'gcc', 'clang']

    algorithms = ['hyperband', 'bohb', 'dehb', 'priorband', 'smac', 'bestconfig', 'flash', 'ga', 'hebo', 'promise', 'MFTune-a5']
    # algorithms = ['MFTune-1', 'MFTune-3', 'MFTune-5', 'MFTune-7', 'MFTune-9']
    algorithms = ['flash', 'ga', 'smac', 'hebo', 'bestconfig',]



    for system in systems:
        print(f"==================== Processing system: {system} ====================")
        high_fidelity = HIGH_FIDELITY_DIC.get(system)
        optimization_target = TARGET_DIC.get(system)
        base_path = f'./results/{system}'

        all_results = {}

        # extract results from all algorithm
        for algorithm in algorithms:
            if algorithm in ['bestconfig', 'flash', 'smac', 'hebo', 'ga', 'promise']:
                all_results[algorithm] = PerfDataUtils.single_fidelity_perf_collection(os.path.join(base_path, algorithm), algorithm, system, optimization_target)
                print(f"[{algorithm}]:", all_results[algorithm])
            elif algorithm in ['hyperband', 'bohb', 'dehb', 'ga_multi_fidelity', 'priorband', 'MFTune-a1', 'MFTune-a3', 'MFTune-a5', 'MFTune-a7', 'MFTune-a9']:
                all_results[algorithm] = PerfDataUtils.multi_fidelity_perf_collection(os.path.join(base_path, algorithm), algorithm, high_fidelity, system, optimization_target)
                print(f"[{algorithm}]:", all_results[algorithm])

            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        # calculate and save Scott-Knott ranking
        save_path = './RQ2'
        if not os.path.exists(save_path):
            os.makedirs('RQ2', exist_ok=True)
        save_path = os.path.join('RQ2', 'scott_knott.csv')

        calculate_and_save_scott_knott(system, all_results, save_path)

        # add summary for all systems
        append_all_system_summary(save_path='RQ2/scott_knott.csv', rank_decimals=2)

        # convert to wide layout and save
        make_wide_layout(input_csv='RQ2/scott_knott.csv',
                         output_csv='RQ2/scott_knott_wide.csv',
                         metric_order=('r', 'mean'))


if __name__ == '__main__':
    main()



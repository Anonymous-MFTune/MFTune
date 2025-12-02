import os
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HIGH_FIDELITY_DIC = {
    'mysql': [180, 50, 100000, 4, 0.5],
    'postgresql': [180, 50, 100000, 4, 0.5],
    'httpd': [8, 50, 180, True],
    'tomcat': [8, 50, 180, True],
    'gcc': [10, 50, 6, 4, 50],
    'clang': [10, 50, 6, 4, 50],
}
TARGET_DIC = {
    'mysql': 'throughput',
    'postgresql': 'throughput',
    'httpd': 'RPS',
    'tomcat': 'RPS',
    'gcc': 'run_time',
    'clang': 'run_time',
}
# BUDGET_DIC = {
#     'mysql': 86400,
#     'postgresql': 86400,
#     'httpd': 43200,
#     'tomcat': 43200,
#     'gcc': 14400,
#     'clang': 14400,
# }

BUDGET_DIC = {
    'mysql': 24,
    'postgresql': 24,
    'httpd': 12,
    'tomcat': 12,
    'gcc': 4,
    'clang': 4,
}

INTERVAL_DIC = {
    'mysql': 7200,
    'postgresql': 7200,
    'httpd': 3600,
    'tomcat': 3600,
    'gcc': 1200,
    'clang': 1200,
}

ALGORITHM_DIC = {
    'ga': 'GA',
    'flash': 'FLASH',
    'smac': 'SMAC',
    'bestconfig': 'BestConfig',
    'hyperband': 'Hyperband',
    'hebo': 'HEBO',
    'bohb': 'BOHB',
    'dehb': 'DEHB',
    'ga_multi_fidelity': 'MFTune',
    'priorband': 'PriorBand',
    'promise': 'PromiseTune',
    'MFTune-a5': 'MFTune',
}


def enforce_monotonic(sequence, system):
    """Ensure that throughput is a monotonically non-decreasing/non-increasing convergent process"""
    new_seq = sequence.copy()
    if system in ['gcc', 'clang']:
        # as for minimization problem, we need to ensure non-increasing
        for i in range(1, len(new_seq)):
            if new_seq[i] > new_seq[i - 1]:
                new_seq[i] = new_seq[i - 1]
    else:
        for i in range(1, len(new_seq)):
            if new_seq[i] < new_seq[i - 1]:
                new_seq[i] = new_seq[i - 1]

    return new_seq


def pad_with_last_value(data, target_length):
    """Pad the sequence with the last value until it reaches the target length."""
    data = list(data)
    while len(data) < target_length:
        data.append(data[-1])
    return data


def pad_cumulative_cost(cost_series, target_length, default_interval=30):
    cost_series = list(cost_series)
    if len(cost_series) >= 2:
        interval = cost_series[-1] - cost_series[-2]
    else:
        interval = default_interval

    while len(cost_series) < target_length:
        cost_series.append(cost_series[-1] + interval)
    return cost_series


def process_single_fidelity_runs_postfilter(base_path, algorithm, system, optimization_target):
    run_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('run_')])
    targets, costs = [], []
    max_length = 0

    for run in run_dirs:
        run_path = os.path.join(base_path, run)
        if algorithm == 'bestconfig':
            file_path = os.path.join(run_path, 'BestConfigTuner_results.csv')
        elif algorithm == 'flash':
            file_path = os.path.join(run_path, 'FLASHTuner_results.csv')
        elif algorithm == 'smac':
            file_path = os.path.join(run_path, 'SMACTuner_results.csv')
        elif algorithm == 'ga':
            file_path = os.path.join(run_path, 'GATuner_results.csv')
        elif algorithm == 'hebo':
            file_path = os.path.join(run_path, 'HEBOTuner_results.csv')
        elif algorithm == 'flash_old':
            file_path = os.path.join(run_path, 'FLASHTuner_results.csv')
        elif algorithm == 'smac_old':
            file_path = os.path.join(run_path, 'SMACTuner_results.csv')
        elif algorithm == 'promise':
            file_path = os.path.join(run_path, 'PromiseTuner_results.csv')
        else:
            continue
        df = pd.read_csv(file_path)
        t = enforce_monotonic(df[optimization_target].tolist(), system)
        c = df['cost'].tolist()
        targets.append(t)
        costs.append(c)
        max_length = max(max_length, len(t))

    targets = [pad_with_last_value(t, max_length) for t in targets]
    costs = [pad_with_last_value(c, max_length) for c in costs]

    mean_target = np.mean(targets, axis=0)
    std_target = np.std(targets, axis=0)
    mean_cost = np.mean(costs, axis=0)

    return np.cumsum(mean_cost), mean_target, std_target


def process_multi_fidelity_runs_postfilter(base_path, algorithm, high_fidelity, system, optimization_target):
    run_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('run_')])
    all_runs_target, all_runs_cost = [], []
    max_length = 0

    for run in run_dirs:
        run_path = os.path.join(base_path, run)
        if algorithm == 'hyperband':
            data_file = os.path.join(run_path, 'HBTuner_results.csv')
        elif algorithm == 'MFTune-a5':
            data_file = os.path.join(run_path, 'GATuner_results.csv')
        elif algorithm == 'flash_multi_fidelity':
            data_file = os.path.join(run_path, 'FLASHTuner_results.csv')
        elif algorithm == 'bo_multi_fidelity':
            data_file = os.path.join(run_path, 'BayesTuner_results.csv')
        elif algorithm == 'bohb':
            data_file = os.path.join(run_path, 'BOHBTuner_results.csv')
        elif algorithm == 'dehb':
            data_file = os.path.join(run_path, 'DEHBTuner_results.csv')
        elif algorithm == 'priorband':
            data_file = os.path.join(run_path, 'PriorBand_results.csv')


        df = pd.read_csv(data_file)
        df['fidelity'] = df['fidelity'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        cumulative_cost, selected_target = 0, []
        cost_series = []

        for _, row in df.iterrows():
            cumulative_cost += row['cost']
            if row['fidelity'] == high_fidelity:
                
                selected_target.append(row[optimization_target])
                cost_series.append(cumulative_cost)

        if selected_target:
            selected_target = enforce_monotonic(selected_target, system)
            all_runs_target.append(selected_target)
            all_runs_cost.append(cost_series)
            max_length = max(max_length, len(selected_target))

    padded_target = [pad_with_last_value(t, max_length) for t in all_runs_target]
    padded_cost = [pad_cumulative_cost(cost_series, max_length) for cost_series in all_runs_cost]

    mean_target = np.mean(padded_target, axis=0)
    std_target = np.std(padded_target, axis=0)
    mean_cost = np.mean(padded_cost, axis=0)

    return mean_cost, mean_target, std_target


def downsample_by_cost_interval(costs, means, stds, interval=7200, max_points=12):
    """
    Downsample the time series by cost interval (e.g., every 3600s).
    Extracts the first point where cumulative cost >= interval * i.
    """
    new_costs = [costs[0]]
    new_means = [means[0]]
    new_stds = [stds[0]]
    next_threshold = interval
    idx = 0

    for i in range(max_points):
        while idx < len(costs) and costs[idx] < next_threshold:
            idx += 1
        if idx >= len(costs):
            break
        new_costs.append(costs[idx])
        new_means.append(means[idx])
        new_stds.append(stds[idx])
        next_threshold += interval

    return new_costs, new_means, new_stds


def sample_best_upto_fixed_intervals(costs, means, stds, interval=7200, max_points=12, system=None):
    """
    fixed-interval sampling of the best performance up to that point.
    For each interval, find the best performance achieved up to that cost.
    x-axis:interval * i
    """
    new_costs = [costs[0]]
    new_means = [means[0]]
    new_stds = [stds[0]]
    # new_costs = [costs[0]]
    # new_means = [means[0]]
    # new_stds = [stds[0]]

    current_best_perf = means[0]
    current_best_std = 0
    idx = 0
    total_points = len(costs)

    for i in range(1, max_points + 1):
        current_time = i * interval

        while idx < total_points and costs[idx] <= current_time:

            if system in ['mysql', 'postgresql', 'tomcat', 'httpd'] and means[idx] > current_best_perf:
                current_best_perf = means[idx]
                current_best_std = stds[idx]
            elif system in ['gcc', 'clang'] and means[idx] < current_best_perf:
                current_best_perf = means[idx]
                current_best_std = stds[idx]
            idx += 1

        new_costs.append(current_time)
        new_means.append(current_best_perf)
        new_stds.append(current_best_std)

    return new_costs, new_means, new_stds


def plot_convergence_postfilter(system, algorithms, high_fidelity=None, optimization_target='throughput', budget=86400, interval=7200):

    base_path = f'./results/{system}'
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.get_cmap("tab10")
    COLORS = [cmap(i) for i in range(len(algorithms))]
    # COLORS = ['#D95319', '#77AC30'] 003049 D62828
    COLORS = ['#D62828', 'teal', '#003049'] #'#54C783',
    LINESTYLES = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",]
    # MARKERS = ["o", "^", "*"]
    # COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000", ]
    MARKERS = ["*", "o",  "s",  "^",  "D",  "v",  ">", "<",  "p",  "h",  "X"]

    for i, algo in enumerate(algorithms):
        if algo in ['hyperband', 'MFTune-a5', 'flash_multi_fidelity', "bo_multi_fidelity", 'bohb', 'dehb', 'priorband']:
            mean_cost, mean_perf, std_perf = process_multi_fidelity_runs_postfilter(
                os.path.join(base_path, algo), algo, high_fidelity, system, optimization_target)
        else:
            mean_cost, mean_perf, std_perf = process_single_fidelity_runs_postfilter(
                os.path.join(base_path, algo), algo, system, optimization_target)

        # downsampling to reduce number of points in the curve

        # mean_cost, mean_perf, std_perf = downsample_by_cost_interval(mean_cost, mean_perf, std_perf, interval)
        mean_cost, mean_perf, std_perf = sample_best_upto_fixed_intervals(mean_cost, mean_perf, std_perf, interval, max_points=14, system=system)

        # adding initial point (0,0) for convergence plot
        # mean_cost = np.insert(mean_cost, 0, 0)
        # mean_perf = np.insert(mean_perf, 0, 0)
        # std_perf = np.insert(std_perf, 0, 0)

        mean_cost = np.array(mean_cost)
        mean_perf = np.array(mean_perf)
        std_perf = np.array(std_perf)

        #  # convert to hours
        cost_hours = mean_cost / 3600.0

        if system in ['gcc', 'clang']:
            # change the unit second to millisecond
            mean_perf = np.array(mean_perf) * 1000
            std_perf = np.array(std_perf) * 1000

        plt.plot(
            cost_hours,
            mean_perf,
            # label= "MFTune" if algo=='ga_multi_fidelity' else algo,
            label=ALGORITHM_DIC.get(algo, algo),
            color=COLORS[i],
            linestyle=LINESTYLES[i],
            marker=MARKERS[i],
            markersize=10,
            # markerfacecolor='none',
            # markeredgecolor=COLORS[i],
            linewidth=1
            )
        plt.fill_between(cost_hours, np.array(mean_perf) - np.array(std_perf), np.array(mean_perf) + np.array(std_perf),
                         alpha=0.15, color=COLORS[i])

    # set scientific notation for x-axis
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((3, 3))  # 10^3
    # ax.xaxis.set_major_formatter(formatter)
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
    # ax.xaxis.offsetText.set_fontsize(32)

    # ax.xaxis.offsetText.set_fontsize(14)
    ax.xaxis.offsetText.set_va('bottom')
    ax.xaxis.offsetText.set_ha('left')


    plt.xlabel("Budget (h)", fontsize=36)
    if system in ['gcc', 'clang']:
        plt.ylabel("Runtime (ms)", fontsize=36)
    elif system in ['httpd', 'tomcat']:
        plt.ylabel("Throughput (rps)", fontsize=36)
    elif system in ['mysql', 'postgresql']:
        plt.ylabel("Throughput (tps)", fontsize=36)

    ax.tick_params(axis='both', which='major', labelsize=32)
    # plt.title(f"{system}", fontsize=28)
    plt.xlim([0, budget+0.1])
    # plt.grid(True)
    plt.legend(loc="best", frameon=False, fontsize=32)
    # plt.ylim([450, 600])
    out_path = f'./RQ1/{system}.pdf'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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

    plt.savefig(out_path, bbox_inches='tight', format='pdf')
    # plt.show()



def main():

    systems = ['mysql', 'postgresql', 'httpd', 'tomcat', 'gcc', 'clang']
    algorithms = ['ga', 'MFTune-a5']

    for system in systems:
        high_fidelity = HIGH_FIDELITY_DIC[system]
        optimization_target = TARGET_DIC[system]
        budget = BUDGET_DIC[system]
        interval = INTERVAL_DIC[system]
        plot_convergence_postfilter(system, algorithms, high_fidelity, optimization_target, budget, interval)


if __name__ == "__main__":
    main()

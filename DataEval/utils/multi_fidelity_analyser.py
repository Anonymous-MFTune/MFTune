import json
import os
import re
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
import seaborn as sns
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from scipy.stats import spearmanr
from matplotlib.ticker import FuncFormatter, MultipleLocator


class MultiFidelityAnalyser:
    def __init__(self, run, system):
        self.run = run
        self.data_path = f'results/{system}/MFTune-a5-dis/run_{run}_ga_multi_fidelity'
        self.json_path = f'config/{system}_knobs.json'
        self.config_min_max = None
        self.load_json()
        self.save_path = f"discussion/mf_analysis/{system}"
        os.makedirs(self.save_path, exist_ok=True)
        self.system = system


    def load_json(self, num_configs=20):
        """
        Load config info from JSON file
        :param num_configs: config no
        """
        with open(self.json_path, 'r') as f:
            json_data = json.load(f)

        selected_configs = list(json_data.keys())[:num_configs]


        self.config_min_max = {}
        for config_name in selected_configs:  #
            value = json_data.get(config_name, None)
            if value is None:
                print(f"Config {config_name} not found in JSON.")
                continue

            if 'min' in value and 'max' in value:
                self.config_min_max[config_name] = {'min': value['min'], 'max': value['max']}


    def normalize_configs(self, config_data):
        """
        normalize configs
        """
        for config_name in config_data.columns:
            if config_name in self.config_min_max:
                min_val = self.config_min_max[config_name]["min"]
                max_val = self.config_min_max[config_name]["max"]
                config_data[config_name] = (config_data[config_name] - min_val) / (max_val - min_val)
            else:
                min_val = config_data[config_name].min()
                max_val = config_data[config_name].max()
                config_data[config_name] = (config_data[config_name] - min_val) / (max_val - min_val)
                print(f"Using column-based normalization for {config_name}: min={min_val}, max={max_val}")

        return config_data

    def identify_evolution_stage_ids(self):
        """
        automatically identify low fidelity id of evolutionary stage
        """
        id_to_gens = {}

        for file in os.listdir(self.data_path):
            if file.startswith('lf_id') and 'config_pop_gen_' in file and '_on_hf' not in file and '2d' not in file:
                id_prefix = file.split('_config')[0]
                gen_num = int(file.split('_gen_')[1].split('.')[0])

                if id_prefix not in id_to_gens:
                    id_to_gens[id_prefix] = set()
                id_to_gens[id_prefix].add(gen_num)

        evolution_stage_ids = []
        for id_prefix, gens in id_to_gens.items():
            if any(g > 0 for g in gens):  # if exist gen > 0 -> evolutionary stage
                evolution_stage_ids.append(id_prefix)

        return evolution_stage_ids

    def best_ind_convergence_visualization(self):
        evolution_ids = self.identify_evolution_stage_ids()

        for evo_id in evolution_ids:

            lf_gen_files = sorted(
                [f for f in os.listdir(self.data_path) if
                 f.startswith(f'{evo_id}_config_pop_gen_') and '_on_hf' not in f],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            hf_gen_files = sorted(
                [f for f in os.listdir(self.data_path) if
                 f.startswith(f'{evo_id}_config_pop_gen_') and '_on_hf' in f],
                key=lambda x: int(x.split('_')[-3].split('.')[0])
            )

            objective = 'throughput'
            if self.system in ['mysql', 'postgresql']:
                objective = 'throughput'
            elif self.system in ['tomcat', 'httpd']:
                objective = 'RPS'
            elif self.system in ['gcc', 'clang']:
                objective = 'run_time'

            generations = []
            lf_objective_list = []
            hf_objective_list = []

            for lf_file, hf_file in zip(lf_gen_files, hf_gen_files):
                lf_gen_df = pd.read_csv(os.path.join(self.data_path, lf_file))
                hf_gen_df = pd.read_csv(os.path.join(self.data_path, hf_file))

                generation = int(lf_file.split('_')[-1].split('.')[0])
                generations.append(generation)
                if objective == 'run_time':
                    lf_objective_list.append(lf_gen_df[objective].min())
                    hf_objective_list.append(hf_gen_df[objective].min())
                else:
                    lf_objective_list.append(lf_gen_df[objective].max())
                    hf_objective_list.append(hf_gen_df[objective].max())


            fig, ax = plt.subplots(figsize=(8, 6))
            plt.plot(generations, lf_objective_list, marker='o', color='#b6e2b6', markerfacecolor='#b6e2b6',
                     markeredgecolor='#006400', linestyle='-', label='Low Fidelity')
            plt.plot(generations, hf_objective_list, marker='o', color='#f4cccc', markerfacecolor='#f4cccc',
                     markeredgecolor='#8B0000', linestyle='-', label='Full Fidelity')

            plt.xlabel('Generation', fontsize=18)
            if self.system in ['mysql', 'postgresql']:
                plt.ylabel(f'Best Throughput', fontsize=18)
                plt.ylim(300, 600)
            elif self.system in ['tomcat', 'httpd']:
                plt.ylabel(f'Best Requests Per Second', fontsize=18)
            elif self.system in ['gcc', 'clang']:
                plt.ylabel(f'Best Runtime (s)', fontsize=18)


            # plt.title(f'{evo_id} - Run {self.run}: LF vs Mapped HF (Best Perf Convergence)')
            plt.legend(fontsize=14)
            pdf_path = os.path.join(self.save_path, f"run_{self.run}_evo_stage_{evo_id}_best_convergence.pdf")

            # remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            #  hide bottom and left spines as well
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            # use arrows to draw X/Y axes (in axes fraction: bottom-left (0,0) to top-right (1,1))
            ax.annotate("", xy=(1, 0), xytext=(0, 0), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=1.5))
            ax.annotate("", xy=(0, 1), xytext=(0, 0), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=1.5))
            ax.tick_params(labelsize=14)

            plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
            # plt.show()
            plt.close()

    def average_pop_convergence_visualization(self):
        evolution_ids = self.identify_evolution_stage_ids()

        for evo_id in evolution_ids:

            lf_gen_files = sorted(
                [f for f in os.listdir(self.data_path) if
                 f.startswith(f'{evo_id}_config_pop_gen_') and '_on_hf' not in f],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            hf_gen_files = sorted(
                [f for f in os.listdir(self.data_path) if
                 f.startswith(f'{evo_id}_config_pop_gen_') and '_on_hf' in f],
                key=lambda x: int(x.split('_')[-3].split('.')[0])
            )

            objective = 'throughput'
            if self.system in ['mysql', 'postgresql']:
                objective = 'throughput'
            elif self.system in ['tomcat', 'httpd']:
                objective = 'RPS'
            elif self.system in ['gcc', 'clang']:
                objective = 'run_time'

            generations = []
            lf_mean_objectives = []
            hf_mean_objectives = []

            for lf_file, hf_file in zip(lf_gen_files, hf_gen_files):
                lf_gen_df = pd.read_csv(os.path.join(self.data_path, lf_file))
                hf_gen_df = pd.read_csv(os.path.join(self.data_path, hf_file))

                generation = int(lf_file.split('_')[-1].split('.')[0])
                generations.append(generation)
                lf_mean_objectives.append(lf_gen_df[objective].mean())
                hf_mean_objectives.append(hf_gen_df[objective].mean())

            fig, ax = plt.subplots(figsize=(8, 7))
            plt.plot(generations, lf_mean_objectives, marker='o', markersize=10, color='#b6e2b6', markerfacecolor='#b6e2b6',
                     markeredgecolor='#006400', linestyle='-', label='Low Fidelity')
            plt.plot(generations, hf_mean_objectives, marker='o', markersize=10,  color='#f4cccc', markerfacecolor='#f4cccc',
                     markeredgecolor='#8B0000', linestyle='-', label='Full Fidelity')

            plt.xlabel('Generation', fontsize=20)
            if self.system in ['mysql', 'postgresql']:
                plt.ylabel(f'Mean Population Throughput', fontsize=20)
                plt.ylim(250, 550)
            elif self.system in ['tomcat', 'httpd']:
                plt.ylabel(f'Mean Population Requests Per Second', fontsize=20)
            elif self.system in ['gcc', 'clang']:
                plt.ylabel(f'Mean Population Runtime (s)', fontsize=20)
            # plt.title(f'{evo_id} - Run {self.run}: LF vs HF (Avg Perf Convergence)')
            plt.legend(fontsize=18)
            # pdf_path = os.path.join(self.save_path, f"run_{self.run}_evo_stage_{evo_id}_avg_convergence.pdf")
            pdf_path = os.path.join(self.save_path, f"dis_evo_{self.system}.pdf")

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            ax.annotate("", xy=(1, 0), xytext=(0, 0), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=1.5))
            ax.annotate("", xy=(0, 1), xytext=(0, 0), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=1.5))
            ax.tick_params(labelsize=16)
            plt.savefig(pdf_path,  bbox_inches='tight', pad_inches=0.4, format='pdf')
            # plt.show()
            plt.close()


    def hf_best_ind_convergence_visualization(self):
        # full/high fidelity (_on_hf) file (the last stage)
        hf_gen_files = [f for f in os.listdir(self.data_path) if
                        f.startswith('hf_config_pop_gen_') and '_on_hf' not in f]

        # generation order
        hf_gen_files = sorted(hf_gen_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        generations = []
        hf_max_throughputs = []

        for hf_file in hf_gen_files:
            hf_gen_df = pd.read_csv(os.path.join(self.data_path, hf_file))

            hf_max_throughput = hf_gen_df['throughput'].max()

            generation = int(hf_file.split('_')[-1].split('.')[0])
            generations.append(generation)
            hf_max_throughputs.append(hf_max_throughput)

        plt.plot(generations, hf_max_throughputs, marker='o', color='#404080', markerfacecolor='#404080',
                 markeredgecolor='black', linestyle='-', label='High Fidelity')

        plt.xlabel('Generation')
        plt.ylabel('Best Throughput')
        plt.title(f'Run {self.run}: High Fidelity Convergence in Final Stage')
        plt.legend()
        pdf_path = os.path.join(self.save_path, f"run_{self.run}_final_stage_hf_best_convergence.pdf")
        plt.savefig(pdf_path, format='pdf')
        # plt.show()
        plt.close()

    def hf_average_pop_convergence_visualization(self):

        hf_gen_files = [f for f in os.listdir(self.data_path) if
                        f.startswith('hf_config_pop_gen_') and '_on_hf' not in f]

        hf_gen_files = sorted(hf_gen_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        generations = []
        hf_mean_throughputs = []

        for hf_file in hf_gen_files:

            hf_gen_df = pd.read_csv(os.path.join(self.data_path, hf_file))
            hf_mean_throughput = hf_gen_df['throughput'].mean()
            generation = int(hf_file.split('_')[-1].split('.')[0])
            generations.append(generation)
            hf_mean_throughputs.append(hf_mean_throughput)

        plt.plot(generations, hf_mean_throughputs, marker='o', color='#404080', markerfacecolor='#404080',
                 markeredgecolor='black', linestyle='-', label='High Fidelity')

        plt.xlabel('Generation')
        plt.ylabel('Average Throughput')
        plt.title(f'Run {self.run}: High Fidelity Convergence in Final Stage')
        plt.legend()
        pdf_path = os.path.join(self.save_path, f"run_{self.run}_final_stage_hf_avg_convergence.pdf")
        plt.savefig(pdf_path, format='pdf')
        # plt.show()
        plt.close()


    def identify_filtering_stage_ids(self):
        """
        automatically identify filtering stage id (those lf only with gen 0 and without gen 1 .... )
        return a filtering id list  ['lf_id1', 'lf_id2']
        """
        id_to_gens = {}

        for file in os.listdir(self.data_path):
            if file.startswith('lf_id') and 'config_pop_gen_' in file and '_on_hf' not in file and '2d' not in file:
                id_prefix = file.split('_config')[0]
                gen_num = int(file.split('_gen_')[1].split('.')[0])

                if id_prefix not in id_to_gens:
                    id_to_gens[id_prefix] = set()
                id_to_gens[id_prefix].add(gen_num)

        selection_stage_ids = []
        for id_prefix, gens in id_to_gens.items():
            if gens == {0}:
                selection_stage_ids.append(id_prefix)

        return selection_stage_ids

    def analyze_filtering_stages(self):
        """
        Analyzing the correlations between high and low fidelity
        this is for the solutions obtained by lhs
        """
        filtering_ids = self.identify_filtering_stage_ids()
        filtering_pairs = self.load_filtering_stage_id_pairs()

        for stage_id in filtering_ids:
            if stage_id not in filtering_pairs:
                continue
            low_config, low_perf = filtering_pairs[stage_id]['low']
            high_config, high_perf = filtering_pairs[stage_id]['high']
            self.analyze_and_visualize_stage_pair(stage_id, low_config, low_perf, high_config, high_perf)

    def _is_bool_like_tokens(self, unique_vals_lower):
        """judge whether a column is boolean-like strings/tokens"""
        truthy = {"true", "yes", "y", "on", "t", "1"}
        falsy = {"false", "no", "n", "off", "f", "0"}
        allowed = truthy | falsy
        # as long as all non-missing values are in the allowed set, consider it boolean
        return all(v in allowed for v in unique_vals_lower if v is not None)

    def _to_lower_stripped(self, x):
        if pd.isna(x):
            return None
        # accommodate non-string types (e.g., numbers/booleans), convert to string first
        return str(x).strip().lower()

    def _encode_bool_series(self, s):
        """encode a boolean/boolean-like series to 1/0; missing as -1 (also accommodate mixed True/False/strings in object columns)"""
        truthy = {"true", "yes", "y", "on", "t", "1"}
        falsy = {"false", "no", "n", "off", "f", "0"}

        def _to_lower_stripped(x):
            if pd.isna(x):
                return None
            return str(x).strip().lower()

        def map_fn(x):
            if pd.isna(x):
                return -1
            # directly handle native booleans
            if isinstance(x, (bool, pd.BooleanDtype().type)):
                return 1 if x is True else 0
            v = _to_lower_stripped(x)
            if v in truthy: return 1
            if v in falsy:  return 0

            try:
                f = float(v)
                if f == 1.0: return 1
                if f == 0.0: return 0
            except Exception:
                pass
            return -1

        return s.map(map_fn).astype(int), {"type": "bool", "mapping": {"true": 1, "false": 0}, "missing": -1}

    def _encode_categorical_columns(self, dfs):

        from pandas.api.types import is_numeric_dtype, is_bool_dtype

        dfs = [df.copy() for df in dfs]
        encoders = {}
        cols = dfs[0].columns

        for col in cols:
            if is_bool_dtype(dfs[0][col]):
                for i, df in enumerate(dfs):
                    dfs[i][col], enc = self._encode_bool_series(df[col])
                encoders[col] = enc

        def is_encoded(col):
            return col in encoders

        def _is_bool_like_tokens(unique_vals_lower):
            truthy = {"true", "yes", "y", "on", "t", "1"}
            falsy = {"false", "no", "n", "off", "f", "0"}
            allowed = truthy | falsy
            return all(v in allowed for v in unique_vals_lower if v is not None)

        def _to_lower_stripped(x):
            if pd.isna(x):
                return None
            return str(x).strip().lower()

        for col in cols:
            if is_encoded(col):
                continue

            if is_numeric_dtype(dfs[0][col]):
                continue

            all_vals = pd.Index([])
            for df in dfs:
                all_vals = all_vals.append(pd.Index(df[col].unique()))
            all_vals = pd.Index(all_vals.unique())
            lowered = [_to_lower_stripped(v) for v in all_vals if not pd.isna(v)]

            if _is_bool_like_tokens(lowered):
                for i, df in enumerate(dfs):
                    dfs[i][col], enc = self._encode_bool_series(df[col])
                encoders[col] = enc
                continue

            ordered_vals = [v for v in all_vals if not pd.isna(v)]
            mapping = {v: idx for idx, v in enumerate(ordered_vals)}

            def map_cat(x):
                if pd.isna(x): return -1
                return mapping.get(x, -1)

            for i, df in enumerate(dfs):
                dfs[i][col] = df[col].map(map_cat).astype(int)

            encoders[col] = {"type": "category", "mapping": mapping, "missing": -1}

        return dfs, encoders

    def load_filtering_stage_id_pairs(self, n_config_cols: int = 20):
        """
        load the data for gen_0 of all low fidelity (also mapped in high fidelity)
        :return
            filtering_pairs = {
                'lf_id1': {
                    'low':  (config_df, perf_series),
                    'high': (config_df, perf_series)
                },
                ...}
        self.last_encoding_maps[id_prefix]
        """

        filtering_pairs = {}
        self.last_encoding_maps = {}

        for file_name in os.listdir(self.data_path):
            if 'config_pop_gen_0.csv' in file_name and '_on_hf' not in file_name and '2d' not in file_name:
                id_prefix = file_name.split('_config')[0]
                low_path = os.path.join(self.data_path, file_name)
                high_path = os.path.join(self.data_path, f"{id_prefix}_config_pop_gen_0_on_hf.csv")

                if not os.path.exists(high_path):
                    continue  # without high-fidelity file

                low_df = pd.read_csv(low_path)
                high_df = pd.read_csv(high_path)

                low_config = low_df.iloc[:, :n_config_cols].copy()
                high_config = high_df.iloc[:, :n_config_cols].copy()

                perf_col = 'throughput'
                if self.system in ['mysql', 'postgresql']:
                    perf_col = 'throughput'
                elif self.system in ['tomcat', 'httpd']:
                    perf_col = 'RPS'
                elif self.system in ['gcc', 'clang']:
                    perf_col = 'run_time'

                if perf_col not in low_df.columns or perf_col not in high_df.columns:
                    raise ValueError(f"{perf_col} 列在 {file_name} 或其高保真对应文件中缺失")

                low_perf = low_df[perf_col].copy()
                high_perf = high_df[perf_col].copy()

                (low_config_enc, high_config_enc), encoders = self._encode_categorical_columns([low_config, high_config])

                filtering_pairs[id_prefix] = {
                    'low': (low_config_enc, low_perf),
                    'high': (high_config_enc, high_perf)
                }
                self.last_encoding_maps[id_prefix] = encoders

        return filtering_pairs
    # def load_filtering_stage_id_pairs(self):
    #     """
    #     load the data for gen_0 of all low fidelity (also mapped in high fidelity)
    #     :return
    #         stage_pairs = {
    #             'lf_id1': {
    #                 'low': (config_df, perf_series),
    #                 'high': (config_df, perf_series)
    #             },
    #             ...}
    #     """
    #     filtering_pairs = {}
    #     for file_name in os.listdir(self.data_path):
    #         if 'config_pop_gen_0.csv' in file_name and '_on_hf' not in file_name and '2d' not in file_name:
    #             id_prefix = file_name.split('_config')[0]
    #             low_path = os.path.join(self.data_path, file_name)
    #             high_path = os.path.join(self.data_path, f"{id_prefix}_config_pop_gen_0_on_hf.csv")
    # 
    #             if not os.path.exists(high_path):
    #                 continue  # skip if high-fidelity file doesn't exist
    # 
    #             # read low-fidelity
    #             low_df = pd.read_csv(low_path)
    #             low_config = low_df.iloc[:, :20]
    #             low_perf = low_df['throughput']
    # 
    #             # read high-fidelity
    #             high_df = pd.read_csv(high_path)
    #             high_config = high_df.iloc[:, :20]
    #             high_perf = high_df['throughput']
    # 
    #             # mapping for enum type data
    #             binlog_mapping = {'noblob': 0, 'full': 1, 'minimal': 2}
    #             for df in [low_config, high_config]:
    #                 if 'binlog_row_image' in df.columns:
    #                     df['binlog_row_image'] = df['binlog_row_image'].map(binlog_mapping)
    # 
    #             filtering_pairs[id_prefix] = {
    #                 'low': (low_config, low_perf),
    #                 'high': (high_config, high_perf)
    #             }
    #     return filtering_pairs

    def analyze_and_visualize_stage_pair(self, stage_id, low_config, low_perf, high_config, high_perf):
        """reducing/visualizing low & high fidelity data in the stage of filtering """

        all_configs = pd.concat([low_config, high_config])
        norm_configs = self.normalize_configs(all_configs)
        mds = MDS(n_components=2, random_state=0)
        config_2d = mds.fit_transform(norm_configs)

        n_low = len(low_config)
        low_coords = config_2d[:n_low]
        high_coords = config_2d[n_low:]


        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')


        def plot_circles(ax, coords, perf, edge_color, face_color, marker='o'):
            for x, y, z in zip(coords[:, 0], coords[:, 1], perf):
                ax.scatter([x], [y], [z], s=20,
                           facecolors=face_color,
                           edgecolors=edge_color,
                           linewidths=0.8,
                           marker=marker)

        # full-fidelity: red | low-fidelity: green
        plot_circles(ax, high_coords, high_perf, edge_color='#8B0000', face_color='#f4cccc')  # red
        plot_circles(ax, low_coords, low_perf, edge_color='#006400', face_color='#b6e2b6')  # green

        # ax.scatter(low_coords[:, 0], low_coords[:, 1], low_perf, c=low_perf, cmap='Blues', marker='o', label='Low-Fidelity')
        # ax.scatter(high_coords[:, 0], high_coords[:, 1], high_perf, c=high_perf, cmap='Reds', marker='^', label='High-Fidelity')

        ax.set_xlabel("#D1", fontsize=18, labelpad=15)
        ax.set_ylabel("#D2", fontsize=18, labelpad=15)
        ax.set_zlabel("Throughput" if self.system in ['mysql', 'postgresql', 'httpd', 'tomcat']
                      else "Runtime", fontsize=18)

        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        spearman_corr, _ = spearmanr(low_perf, high_perf)

        if self.system in ['mysql', 'postgresql', 'httpd', 'tomcat']:
            # Top 5% overlap
            top_5_low = np.argsort(low_perf)[-int(0.05 * len(low_perf)):]
            top_5_high = np.argsort(high_perf)[-int(0.05 * len(high_perf)):]
            top5_overlap_rate = len(set(top_5_low).intersection(set(top_5_high))) / len(top_5_low) * 100
            top5_overlap_idx = set(top_5_low).intersection(set(top_5_high))

            # Top 20 overlap
            top_20_low = np.argsort(low_perf)[-20:]
            top_20_high = np.argsort(high_perf)[-20:]
            top20_overlap_rate = len(set(top_20_low).intersection(set(top_20_high))) / 20 * 100
            top20_oeverlap_idx = set(top_20_low).intersection(set(top_20_high))
        elif self.system in ['gcc', 'clang']:
            # Top 5% overlap (lower is better)
            top_5_low = np.argsort(low_perf)[:int(0.05 * len(low_perf))]
            top_5_high = np.argsort(high_perf)[:int(0.05 * len(high_perf))]
            top5_overlap_rate = len(set(top_5_low).intersection(set(top_5_high))) / len(top_5_low) * 100
            top5_overlap_idx = set(top_5_low).intersection(set(top_5_high))

            # Top 20 overlap (lower is better)
            top_20_low = np.argsort(low_perf)[:20]
            top_20_high = np.argsort(high_perf)[:20]
            top20_overlap_rate = len(set(top_20_low).intersection(set(top_20_high))) / 20 * 100
            top20_oeverlap_idx = set(top_20_low).intersection(set(top_20_high))


        # connected lines
        for idx in top20_oeverlap_idx:
            x, y = low_coords[idx]
            z1 = high_perf[idx]
            z2 = low_perf[idx]
            ax.plot([x, x], [y, y], [z1, z2], color='red', linestyle='--', linewidth=1.5)

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Low-fidelity',
                       markerfacecolor='#b6e2b6', markeredgecolor='#006400', markersize=8, linewidth=0),
                Line2D([0], [0], marker='o', color='w', label='Full-fidelity',
                       markerfacecolor='#f4cccc', markeredgecolor='#8B0000', markersize=8, linewidth=0)
            ]
            ax.legend(handles=legend_elements, fontsize=16, loc='best')


        # default view: elev=30, azim=-60
        ax.view_init(elev=25, azim=-45)
        # plt.show()
        # ax.set_title(f"{stage_id} | Spearman: {spearman_corr:.2f} | Top 5% Overlap: {top5_overlap_rate:.2f}% | Top 20 Overlap: {top20_overlap_rate:.2f}%")
        print(f"{stage_id} | Spearman: {spearman_corr:.2f}")
        print(f"Top 5% Overlap Rate: {top5_overlap_rate:.2f}%")
        print(f"Top 20 Overlap Rate: {top20_overlap_rate:.2f}%")
        # pdf_path = os.path.join(self.save_path, f"run_{self.run}_filter_stage_{stage_id}_landscape.pdf")
        pdf_path = os.path.join(self.save_path, f"dis_landscape_{self.system}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.4, format='pdf')

        plt.close()




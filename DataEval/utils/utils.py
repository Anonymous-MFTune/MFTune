import os
import ast
import pandas as pd


class PerfDataUtils:

    @staticmethod
    def single_fidelity_perf_collection(base_path, algorithm, system, optimization_target):
        run_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('run_')])
        performance_values = []

        for run in sorted(run_dirs):
            run_path = os.path.join(base_path, run)
            if algorithm == 'bestconfig':
                data_file = os.path.join(run_path, 'BestConfigTuner_results.csv')
            elif algorithm == 'flash':
                data_file = os.path.join(run_path, 'FLASHTuner_results.csv')
            elif algorithm == 'smac':
                data_file = os.path.join(run_path, 'SMACTuner_results.csv')
            elif algorithm == 'ga':
                data_file = os.path.join(run_path, 'GATuner_results.csv')
            elif algorithm == 'hebo':
                data_file = os.path.join(run_path, 'HEBOTuner_results.csv')
            elif algorithm == 'promise':
                data_file = os.path.join(run_path, 'PromiseTuner_results.csv')

            if os.path.exists(data_file):
                df = pd.read_csv(data_file)

                if system in ['mysql', 'postgresql', 'tomcat', 'httpd'] and optimization_target in df.columns:
                    valid_df = df[df[optimization_target] > 0]
                    if not valid_df.empty:
                        max_target = df[optimization_target].max()
                        performance_values.append(max_target)
                    else:
                        print(f"Warning: All {optimization_target} values are zero or invalid in {data_file}")
                elif system in ['gcc', 'clang', 'x264'] and optimization_target in df.columns:
                    valid_df = df[df[optimization_target] > 0]
                    if not valid_df.empty:
                        min_target = df[optimization_target].min()
                        performance_values.append(min_target)
                    else:
                        print(f"Warning: All {optimization_target} values are zero or invalid in {data_file}")
                else:
                    print(f"Warning: {optimization_target} column not found in {data_file}.")
            else:
                print(f"Warning: {data_file} not found.")

        return performance_values

    @staticmethod
    def multi_fidelity_perf_collection(base_path, algorithm, high_fidelity, system, optimization_target):
        
        run_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('run_')])
        performance_values = []
        data_file = ''
        for run in sorted(run_dirs):
            run_path = os.path.join(base_path, run)
            if algorithm == 'hyperband':
                data_file = os.path.join(run_path, 'HBTuner_results.csv')
            elif algorithm == 'bohb':
                data_file = os.path.join(run_path, 'BOHBTuner_results.csv')
            elif algorithm == 'dehb':
                data_file = os.path.join(run_path, 'DEHBTuner_results.csv')
            elif algorithm == 'ga_multi_fidelity':
                data_file = os.path.join(run_path, 'GATuner_results.csv')
            elif algorithm == 'priorband':
                data_file = os.path.join(run_path, 'PriorBand_results.csv')
            elif algorithm in ['MFTune-a1', 'MFTune-a3', 'MFTune-a5', 'MFTune-a7', 'MFTune-a9', 'MFTune-I', 'MFTune-II']:
                data_file = os.path.join(run_path, 'GATuner_results.csv')

            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                df['fidelity'] = df['fidelity'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

                matched_df = df[df['fidelity'].apply(lambda x: x == high_fidelity)]
                if system in ['mysql', 'postgresql', 'tomcat', 'httpd']:
                    matched_df = matched_df[matched_df[optimization_target] > 0]
                    if not matched_df.empty:
                        max_target = matched_df[optimization_target].max()
                        performance_values.append(max_target)
                    else:
                        print(f"Warning: No match for high_fidelity {high_fidelity} in {data_file}")

                elif system in ['gcc', 'clang', 'x264']:
                    matched_df = matched_df[matched_df[optimization_target] > 0]

                    if not matched_df.empty:
                        min_target = matched_df[optimization_target].min()
                        performance_values.append(min_target)
                    else:
                        print(f"Warning: No match for high_fidelity {high_fidelity} in {data_file}")


            else:
                print(f"Warning: {data_file} not found.")

        return performance_values

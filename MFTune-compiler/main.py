import argparse
import subprocess
import time
import socket
from tuner.ga_tuner import GATuner
from utils.params_parsing import parse_args
from tuner.mf_analyser import MFAnalyser
from tuner.mf_sampler import MFSampler
from tuner.flash_tuner import FLASHTuner
from tuner.bestconfig_tuner import BestConfigTuner
from tuner.smac_tuner import SMACTuner
from tuner.hyperband_tuner import HBTuner
from tuner.hebo_tuner import HEBOTuner
from tuner.bohb_tuner import BOHBTuner
from tuner.dehb_tuner import DEHBTuner
from tuner.promise_tune import PromiseTuner
from tuner.priorband import PriorBandTuner
from tuner.mf1_tuner import MF1Tuner
from tuner.mf2_tuner import MF2Tuner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='params_setup.ini', help='config file')
    parser.add_argument('--fidelity_type', type=str, required=False, help='Fidelity type to use')
    parser.add_argument('--tuning_method', type=str, required=False, help='Tuning method to use (GA, GA3, etc.)')
    parser.add_argument('--run', type=str, required=False, help='No. of runs')
    parser.add_argument('--service_name', type=str, required=False, help='service name')
    parser.add_argument('--container_name', type=str, required=False, help='container name')

    # parser.add_argument('--web_host', type=str, required=False, help='service of docker')
    opt = parser.parse_args()


    # parse the mysql_params_setup.ini file
    args_compiler, args_workload, args_tune = parse_args(opt.config)


    # mysql_params_setup.ini fixed the tuning_method, fidelity, db. In order to run in parallel, each process needs to pass
    # the corresponding db service, tuning method, and fidelity type.
    args_compiler['container_name'] = opt.container_name
    args_tune['tuning_method'] = opt.tuning_method
    args_tune['fidelity_type'] = opt.fidelity_type
    run = int(opt.run)


    if args_tune['tuning_method'] == 'ga':
        optimizer = GATuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_ga()
    elif args_tune['tuning_method'] == 'mf_analyser':
        analyser = MFAnalyser(args_compiler, args_workload, args_tune, run)
        analyser.analyse_and_verify_configs()
    elif args_tune['tuning_method'] == 'mf_sampler':
        sampler = MFSampler(args_compiler, args_workload, args_tune, run)
        sampler.sampling_and_evaluate()
    elif args_tune['tuning_method'] == 'flash':
        optimizer = FLASHTuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_flash()
    elif args_tune['tuning_method'] == 'bestconfig':
        optimizer = BestConfigTuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_bestconfig()
    elif args_tune['tuning_method'] == 'smac':
        optimizer = SMACTuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_smac()
    elif args_tune['tuning_method'] == 'hyperband':
        optimizer = HBTuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_hyperband()
    elif args_tune['tuning_method'] == 'bohb':
        optimizer = BOHBTuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_bohb()
    elif args_tune['tuning_method'] == 'dehb':
        optimizer = DEHBTuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_dehb()
    elif args_tune['tuning_method'] == 'hebo':
        optimizer = HEBOTuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_hebo()
    elif args_tune['tuning_method'] == 'promise':
        optimizer = PromiseTuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_promise()
    elif args_tune['tuning_method'] == 'priorband':
        optimizer = PriorBandTuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_priorband()
    elif args_tune['tuning_method'] == 'mf1':
        optimizer = MF1Tuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_mf1()
    elif args_tune['tuning_method'] == 'mf2':
        optimizer = MF2Tuner(args_compiler, args_workload, args_tune, run)
        optimizer.tune_mf2()


if __name__ == "__main__":
    main()

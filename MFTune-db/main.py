import argparse
from tuner.ga_tuner import GATuner
from utils.params_parsing import parse_args
from tuner.mf_analyser import MFAnalyser
from tuner.flash_tuner import FLASHTuner
from tuner.bestconfig_tuner import BestConfigTuner
from tuner.smac_tuner import SMACTuner
from tuner.hyperband_tuner import HBTuner
from tuner.mf_sampler import MFSampler
from tuner.bohb_tuner import BOHBTuner
from tuner.dehb_tuner import DEHBTuner
from tuner.hebo_tuner import HEBOTuner
from tuner.promise_tuner import PromiseTuner
from tuner.priorband_tuner import PriorBandTuner
from tuner.mf1_tuner import MF1Tuner
from tuner.mf2_tuner import MF2Tuner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./params_setup/mysql_params_setup.ini', help='config file')
    parser.add_argument('--db_host', type=str, required=False, help='Database host to use for this run')
    parser.add_argument('--fidelity_type', type=str, required=False, help='Fidelity type to use')
    parser.add_argument('--tuning_method', type=str, required=False, help='Tuning method to use (GA, GA3, etc.)')
    parser.add_argument('--run', type=str, required=False, help='No. of runs')
    opt = parser.parse_args()

    # parse the mysql_params_setup.ini file
    args_db, args_workload, args_tune = parse_args(opt.config)

    # dynamically set the corresponding db service, tuning method, fidelity type, and no. of run.
    args_db['host'] = opt.db_host
    args_tune['tuning_method'] = opt.tuning_method
    args_tune['fidelity_type'] = opt.fidelity_type
    run = int(opt.run)


    if args_tune['tuning_method'] == 'ga':
        optimizer = GATuner(args_db, args_workload, args_tune, run)
        optimizer.tune_ga()
    elif args_tune['tuning_method'] == 'flash':
        optimizer = FLASHTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_flash()
    elif args_tune['tuning_method'] == 'bestconfig':
        optimizer = BestConfigTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_bestconfig()
    elif args_tune['tuning_method'] == 'smac':
        optimizer = SMACTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_smac()
    elif args_tune['tuning_method'] == 'hyperband':
        optimizer = HBTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_hyperband()
    elif args_tune['tuning_method'] == 'mf_analyser':
        analyser = MFAnalyser(args_db, args_workload, args_tune, run)
        analyser.analyse_and_verify_configs()
    elif args_tune['tuning_method'] == 'mf_sampler':
        collector = MFSampler(args_db, args_workload, args_tune, run)
        collector.sampling_and_evaluate()
    elif args_tune['tuning_method'] == 'bohb':
        optimizer = BOHBTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_bohb()
    elif args_tune['tuning_method'] == 'dehb':
        optimizer = DEHBTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_dehb()
    elif args_tune['tuning_method'] == 'hebo':
        optimizer = HEBOTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_hebo()
    elif args_tune['tuning_method'] == 'promise':
        optimizer = PromiseTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_promise()
    elif args_tune['tuning_method'] == 'priorband':
        optimizer = PriorBandTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_priorband()
    elif args_tune['tuning_method'] == 'mf1':
        optimizer = MF1Tuner(args_db, args_workload, args_tune, run)
        optimizer.tune_mf1()
    elif args_tune['tuning_method'] == 'mf2':
        optimizer = MF2Tuner(args_db, args_workload, args_tune, run)
        optimizer.tune_mf2()


if __name__ == "__main__":
    main()

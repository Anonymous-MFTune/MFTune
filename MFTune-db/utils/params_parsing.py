# -*- coding: utf-8 -*-

import configparser
import json
from collections import defaultdict


class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


knob_config = {}

default_value = {'lua_path': 'oltp_read_write'}

auto_setting = ['knob_num', 'initial_tunable_knob_num']


def get_default_dict(dic):
    config_dic = defaultdict(str)
    for k in dic:
        config_dic[k] = dic[k]

    for key in default_value.keys():
        if key not in config_dic.keys() or config_dic[key] == '':
            config_dic[key] = default_value[key]
    return config_dic


def parse_args(file):
    cf = DictParser()
    cf.read(file, encoding="utf-8")
    config_dict = cf.read_dict()
    global knob_config
    f = open(config_dict['database']['knob_config_file'])
    knob_config = json.load(f)

    return get_default_dict(config_dict["database"]), get_default_dict(config_dict['workload']), get_default_dict(config_dict['tune'])



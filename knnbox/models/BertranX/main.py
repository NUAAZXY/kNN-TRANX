import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from asdl.grammar import GrammarRule, Grammar, ReduceAction
from dataset.dataset import Dataset
from config.config import init_arg_parser
from test import test
from train import train
from utils import GridSearch


if __name__ == '__main__':

    # load config file
    args = init_arg_parser()
    parameters = yaml.load(open(args.config_file).read(), Loader=yaml.FullLoader)
    params = parameters['experiment_env']

    # Fix seed for deterministic results
    SEED = params['seed']

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    is_cuda = torch.cuda.is_available()

    print("Cuda Status on system is {}".format(is_cuda))

    if params['dataset'] is 'conala' or 'codesearchnet':
        asdl_text = open('./asdl/grammar.txt').read()
    if params['dataset'] is 'django':
        asdl_text = open('./asdl/grammar2.txt').read()
    all_productions, grammar, primitives_type = Grammar.from_text(asdl_text)
    act_list = [GrammarRule('<pad>', None, []), GrammarRule('<s>', None, []),
                GrammarRule('</s>', None, []), GrammarRule('<unk>', None, [])]

    act_list += [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in all_productions]

    Reduce = ReduceAction('Reduce')
    act_dict = dict(
        [(act.label, act) if isinstance(act, GrammarRule) or isinstance(act, ReduceAction) else (act, act) for act in
         act_list])
    act_dict[Reduce.label] = Reduce

    # Select device
    device = torch.device('cuda:%s' % (params['GPU']) if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda(params['GPU'])
    else:
        map_location = 'cpu'

    gridsearch = GridSearch(params)

    if params['train']:
        # Load train set
        if params['dataset'] == 'conala':
            train_set = Dataset(pd.read_csv(args.train_path_conala + 'conala-train.csv'))
            dev_set = Dataset(pd.read_csv(args.train_path_conala + 'conala-val.csv'))
            test_set = Dataset(pd.read_csv(args.test_path_conala + 'conala-test.csv'))
            # i = 0
            # for data1 in train_set.examples['snippet_tokens']:
            #     for data2 in test_set.examples['snippet_tokens']:
            #         if data1 == data2:
            #             i += 1
            #             print(data1, data2)
            # print(i)
        elif params['dataset'] == 'django':
            train_set = Dataset(pd.read_csv(args.train_path_django + 'train.csv'))
            dev_set = Dataset(pd.read_csv(args.dev_path_django + 'dev.csv'))
        else:
            sys.exit('Wrong train path for' + params['dataset'])

        train(train_set, dev_set, args, gridsearch, act_dict, grammar, primitives_type, device, map_location, is_cuda,test_set)

    if params['test']:
        # Load test set
        if params['dataset'] == 'conala':
            test_set = Dataset(pd.read_csv(args.test_path_conala + 'conala-test.csv'))
        elif params['dataset'] == 'apps':
            test_set = Dataset(pd.read_csv(args.test_path_apps + 'test.csv'))
        else:
            sys.exit('Wrong test path for' + params['dataset'])

        test(test_set, gridsearch, args, map_location, act_dict, grammar, primitives_type, device, is_cuda)

import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml

# from asdl.grammar import GrammarRule, Grammar, ReduceAction
from knnbox.models.BertranX.asdl.grammar import GrammarRule, Grammar, ReduceAction
from knnbox.models.BertranX.dataset.dataset import Dataset
from knnbox.models.BertranX.config.config import init_arg_parser
from knnbox.models.BertranX.test import test
from knnbox.models.BertranX.train import train
from knnbox.models.BertranX.utils import GridSearch
import csv
import pickle
from knnbox.models.BertranX.model.nl2code import nl2code
from knnbox.models.BertranX.utils import evaluate_action, decode_action
def build_data(dataset):
    if dataset == 'knntranx_adaptive':
        parameters = yaml.load(open('knnbox/models/BertranX/config/config.yml').read(), Loader=yaml.FullLoader)
    else:
        parameters = yaml.load(open('knnbox/models/BertranX/config/config_django.yml').read(), Loader=yaml.FullLoader)
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
        asdl_text = open('knnbox/models/BertranX/asdl/grammar.txt').read()
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
    # train_set = Dataset(pd.read_csv(args.train_path_conala + 'conala-test.csv'))
    print(params)
    path_folder_config = 'knnbox/models/BertranX/outputs/{0}.dataset_{1}._word_freq_{2}.nl_embed_{3}.action_embed_{4}.att_size_{5}.hidden_size_{6}.epochs_{7}.dropout_enc_{8}.dropout_dec_{9}.batch_size{10}.parent_feeding_type_{11}.parent_feeding_field_{12}.change_term_name_{13}.seed_{14}/'.format(
        params['model'],
        params['dataset'],
        params['word_freq'],
        params['nl_embed_size'],
        params['action_embed_size'],
        params['att_size'],
        params['hidden_size'],
        params['epochs'],
        params['dropout_encoder'],
        params['dropout_decoder'],
        params['batch_size'],
        params['parent_feeding_type'],
        params['parent_feeding_field'],
        params['change_term_name'],
        params['seed']
    )
    print(path_folder_config)
    import sys
    sys.path.append('knnbox/models/BertranX')
    vocab = pickle.load(open(path_folder_config + 'vocab', 'rb'))
    print(len(vocab.source))
    return params, gridsearch, map_location, act_dict, grammar, primitives_type, device, is_cuda, path_folder_config, vocab

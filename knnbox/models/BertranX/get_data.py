import yaml

from knnbox.models.BertranX.components.vocabulary.create_vocab_dataset import create_vocab
from knnbox.models.BertranX.config.config import init_arg_parser
from knnbox.models.BertranX.dataset.data_conala.json_to_csv import data_creation
from knnbox.models.BertranX.dataset.data_conala.preprocess_conala import preprocess_data_conala
from knnbox.models.BertranX.dataset.data_django.preprocess_django import *


if __name__ == '__main__':
    args = init_arg_parser()
    params = yaml.load(open(args.config_file).read(), Loader=yaml.FullLoader)
    params = params['experiment_env']

    if params['dataset'] is 'conala' or 'codesearchnet' or 'apps':
        asdl_text = open('./asdl/grammar.txt').read()
    if params['dataset'] is 'django':
        asdl_text = open('./asdl/grammar2.txt').read()

    grammar, _, _ = Grammar.from_text(asdl_text)
    act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
    assert (len(grammar) == len(act_list))
    Reduce = ReduceAction('Reduce')
    act_dict = dict([(act.label, act) for act in act_list])
    act_dict[Reduce.label] = Reduce

    if params['dataset'] == 'conala':
        preprocess_data_conala(args.raw_path_conala, act_dict, params)
        data_creation(args.train_path_conala, args.raw_path_conala, params['number_merge_ex'], mode=params['mode'])
        data_creation(args.test_path_conala, args.raw_path_conala, params['number_merge_ex'], mode='test')

        if params['create_vocab'] == True:
            pydf_train = pd.read_csv(args.train_path_conala + 'conala-train.csv')
            pydf_valid = pd.read_csv(args.train_path_conala + 'conala-val.csv')

            pydf_vocabulary = pd.concat([pydf_train[['intent', 'snippet_actions']],
                                         pydf_valid[['intent', 'snippet_actions']]])

            create_vocab(pydf_train, act_dict, params)
            # pydf_train = pd.read_csv(args.train_path_conala + 'conala-train.csv')
            # create_vocab(pydf_train, act_dict, params)

    if params['dataset'] == 'django':
        Django.process_django_dataset(params, act_dict)

        if params['create_vocab'] == True:
            pydf_train = pd.read_csv(args.train_path_django + 'train.csv')
            pydf_valid = pd.read_csv(args.train_path_django + 'dev.csv')

            pydf_vocabulary = pd.concat([pydf_train[['intent', 'snippet_actions']],
                                         pydf_valid[['intent', 'snippet_actions']]])

            create_vocab(pydf_train, act_dict, params)
            # pydf_train = pd.read_csv('dataset/data_django/train.csv')
            # create_vocab(pydf_train, act_dict, params)

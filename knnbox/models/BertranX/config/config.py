import argparse


def init_arg_parser():

    arg_parser = argparse.ArgumentParser()

    # PATHS
    arg_parser.add_argument('--train_path_codesearchnet', default="./dataset/data_github/python/final/jsonl/train/", type=str,
                            help='path to the training target file')
    arg_parser.add_argument('--dev_path_codesearchnet', default="./dataset/data_github/python/final/jsonl/valid/", type=str,
                            help='path to the dev target file')
    arg_parser.add_argument('--test_path_codesearchnet', default="./dataset/data_github/python/final/jsonl/test/", type=str,
                            help='path to the test model file')
    arg_parser.add_argument('--raw_path_codesearchnet', default="./dataset/data_github/python/",
                            type=str, help='path to the codesearchnet data folder')
    arg_parser.add_argument('--vocab_path_codesearchnet', default="./knnbox/models/BertranX/vocabulary/codesearchnet/", type=str,
                            help='path to the eval model file')

    arg_parser.add_argument('--raw_path_conala', default='./dataset/data_conala/conala-corpus/', type=str,
                            help='path to the raw target file')
    arg_parser.add_argument('--train_path_conala', default="./dataset/data_conala/train/", type=str,
                            help='path to the training target file')
    arg_parser.add_argument('--dev_path_conala', default="./dataset/data_conala/train/conala-val.csv", type=str,
                            help='path to the dev file')
    arg_parser.add_argument('--test_path_conala', default="./dataset/data_conala/test/", type=str,
                            help='path to the eval model file')
    arg_parser.add_argument('--vocab_path_conala', default="./knnbox/models/BertranX/vocabulary/conala/", type=str,
                            help='path to the eval model file')

    arg_parser.add_argument('--train_path_django', default="./dataset/data_django/", type=str,
                            help='path to the training target file')
    arg_parser.add_argument('--dev_path_django', default="./dataset/data_django/", type=str,
                            help='path to the dev file')
    arg_parser.add_argument('--test_path_django', default="./dataset/data_django/", type=str,
                            help='path to the eval model file')
    arg_parser.add_argument('--vocab_path_django', default="./knnbox/models/BertranX/vocabulary/django/", type=str,
                            help='path to the eval model file')

    arg_parser.add_argument('--train_path_apps', default="./dataset/data_apps/", type=str,
                            help='path to the training target file')
    arg_parser.add_argument('--test_path_apps', default="./dataset/data_apps/", type=str,
                            help='path to the eval model file')


    # YAML
    arg_parser.add_argument('config_file', metavar='CONFIG_FILE', default='config/config_token.yml',
                            type=str, help='the configuration file')

    args = arg_parser.parse_args()

    return args

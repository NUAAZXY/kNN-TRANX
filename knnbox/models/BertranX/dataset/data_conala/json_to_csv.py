from __future__ import print_function

import pickle

import pandas as pd


def data_creation(target_path, raw_path, number_of_examples, mode):
    """
    creation of json file for training/testing data
    """
    raw_data = {}
    path_json = raw_path + 'conala-%s.json.seq2seq' % mode
    if mode == 'train':  # merge mined and train examples
        dataset = json_merge(path_json, raw_path, number_of_examples)
    elif mode == 'debug':
        dataset = pickle.load(open(raw_path + 'conala-train.json.seq2seq', 'rb'))
    else:
        dataset = pickle.load(open(path_json, 'rb'))

    raw_data['intent'] = [example['intent_tokens'] for example in dataset]
    raw_data['snippet_actions'] = [example['snippet_actions'] for example in dataset]
    raw_data['snippet_tokens'] = [example['snippet_tokens'] for example in dataset]
    raw_data['parent_types'] = [example['type'] for example in dataset]
    raw_data['parent_fields'] = [example['field'] for example in dataset]
    raw_data['parent_cardinalities'] = [example['cardinality'] for example in dataset]
    raw_data['slot_map'] = [example['slot_map'] for example in dataset]

    if mode == 'test':
        # del raw_data['snippet_actions']
        # raw_data['snippet'] = raw_data.pop('snippet_tokens')
        pd.DataFrame(raw_data).to_csv(target_path + 'conala-%s.csv' % mode, index=False)
    elif mode == 'debug':
        dataset = pd.DataFrame(raw_data)
        dataset = dataset.sample(frac=1)
        df_val = dataset.iloc[:40]
        df_train = df_val[['intent', 'snippet_actions', 'snippet_tokens', 'slot_map', 'parent_types', 'parent_fields', 'parent_cardinalities']].copy()
        # df_train = df_train.rename(columns={'snippet_actions': 'snippet'})
        df_val = df_val[['intent', 'snippet_actions', 'snippet_tokens', 'slot_map', 'parent_types', 'parent_fields', 'parent_cardinalities']].copy()
        # df_val = df_val.rename(columns={'snippet_tokens': 'snippet'})
        df_train.to_csv(target_path + 'conala-train.csv', index=False)
        df_val.to_csv(target_path + 'conala-val.csv', index=False)
    else:
        dataset = pd.DataFrame(raw_data)
        # dataset = dataset.sample(frac=1, random_state=1)
        df_train = dataset.iloc[200:]
        df_val = dataset.iloc[:200]
        i = 0

        for df in df_train['snippet_tokens']:
            for db in df_val['snippet_tokens']:
                if df == db:
                    print(df)
                    print(db)
                    i+=1
        print(i)
        print(len(df_train), len(df_val))
        # print(len(df_train), len(df_val))
        df_train = df_train[['intent', 'snippet_actions', 'snippet_tokens', 'slot_map', 'parent_types', 'parent_fields', 'parent_cardinalities']].copy()
        # df_train = df_train.rename(columns={'snippet_actions': 'snippet'})
        df_val = df_val[['intent', 'snippet_actions', 'snippet_tokens', 'slot_map', 'parent_types', 'parent_fields', 'parent_cardinalities']].copy()
        # df_val = df_val.rename(columns={'snippet_tokens': 'snippet'})
        df_train.to_csv(target_path + 'conala-%s.csv' % mode, index=False)
        df_val.to_csv(target_path + 'conala-val.csv', index=False)


def json_merge(path_train, path_pickle, number_of_examples):
    """
    merge train and mined data for training
    :param json_train: pathtraining json file
    :param path_pickle: path to raw files
    :param number_of_examples: number of mined examples to merge
    :return: merged training data
    """
    path_mined = path_pickle + 'conala-mined.jsonl.seq2seq'

    # data_train = pickle.load(open(path_train, 'rb'))
    #
    # data_mined = pickle.load(open(path_mined, 'rb'))
    with open(path_mined, 'rb') as mined:
        data_mined = pickle.load(mined)

    with open(path_train, 'rb') as train:
        data_train = pickle.load(train)

    for i in range(number_of_examples):
        data_train.append(data_mined[i])

    return data_train


if __name__ == '__main__':
    data_creation("train/", './data_conala/conala-corpus/', 10, mode='train')

import json
import pickle
import sys
from collections import Counter
from collections import deque
import ast

import astor
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

from knnbox.models.BertranX.asdl.ast_operation import make_iterlists, seq2ast, Grammar, GrammarRule
from knnbox.models.BertranX.asdl.grammar import ReduceAction
from knnbox.models.BertranX.config.config import init_arg_parser
from knnbox.models.BertranX.dataset.data_conala.preprocess_conala import decanonicalize_code_conala
from knnbox.models.BertranX.dataset.data_django.preprocess_django import Django
from knnbox.models.BertranX.dataset.utils import tokenize_for_bleu_eval
from knnbox.CodeBLEU.calc_code_bleu_method import calc_code_bleu


def evaluate_action(examples, model, act_dict, metric='BLEU', is_cuda=False, return_decode_result=False):
    """
    Compute the action sequence and evaluate it with BLEU
    """
    intents, snippet, decode_results_bleu, decode_results = decode_action(examples, model, act_dict, is_cuda)


    accuracy = accuracy_score(decode_results_bleu, snippet)

    ref = []
    dec_false = []
    intent = []
    hyp = []
    gold = []
    for i in range(len(snippet)):
        hyp.append(decode_results_bleu[i])
        gold.append(snippet[i][0])
        if decode_results_bleu[i] != snippet[i][0]:
            ref.append(snippet[i][0])
            dec_false.append(decode_results_bleu[i])
            intent.append(intents[i])

    intent = [' '.join(snip[0]) for snip in intent]
    ref = [' '.join(snip) for snip in ref]
    dec_false = [' '.join(decode) for decode in dec_false]
    decode_results = [*zip(intent, ref, dec_false)]
    hyp = [' '.join(h) for h in hyp]
    gold = [' '.join(g) for g in gold]
    BLEU = calc_code_bleu(gold, hyp)
    if return_decode_result:
        return BLEU*100, accuracy*100, decode_results
    else:
        if metric == 'BLEU':
            return BLEU*100, accuracy*100
        else:
            return accuracy*100, BLEU*100


def decanonicalize_code(code, slot_map, dataset):
    if dataset == 'conala':
        return decanonicalize_code_conala(code, slot_map)
    elif dataset == 'django':
        return Django.decanonicalize_code_django(code, slot_map)
    elif dataset == 'apps':
        return code



def decode_action(examples, model, act_dict, is_cuda):
    """
    Predict sequence of actions and reconstruct code at inference
    """

    model.eval()

    decode_results = []
    decode_results_bleu = []
    intents = []
    snippet = []
    err = 0
    for index, example in tqdm(examples.iterrows(), desc='Decoding', file=sys.stdout, total=len(examples),
                               disable=is_cuda):
        hyps = model.parse_gen(eval(example.intent))
        if not hyps:
            decode_results.append([])
            decode_results_bleu.append([])
        for i, hyp in enumerate(hyps):
            try:
                hyp_ = hyp.actions
                hyp_ = [('Reduce', action[1]) if action[0] == 'Reduce_primitif' else action for action in hyp_]
                hyp_ = [(act_dict[a[0]], a[1]) if a[0] in act_dict else a for a in hyp_]
                code = seq2ast(make_iterlists(deque(hyp_)))
                code = astor.to_source(code).rstrip()
                code = decanonicalize_code(code, eval(example.slot_map), model.args['dataset'])
                code_bleu = tokenize_for_bleu_eval(code)
                decode_results.append(code)
                decode_results_bleu.append(code_bleu)
                break
            except Exception as e:
                print(i, e)
                if i == len(hyps) - 1:
                    err += 1
                    decode_results.append([])
                    decode_results_bleu.append([])
                pass
        intents.append([eval(example.intent)])
        snippet.append([eval(example.snippet_tokens)])

    return intents, snippet, decode_results_bleu, decode_results


# Time counter

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def common_subseq(seq_actions, window):
    sub_seqs = dict()
    for seq_action in seq_actions:
        seq_action = eval(seq_action)
        sub_sequences = [str(seq_action[i: i + window]) for i in range(len(seq_action) - window)]
        for sub_seq in sub_sequences:
            if sub_seq in sub_seqs:
                sub_seqs[sub_seq] += 1
            else:
                sub_seqs[sub_seq] = 1
    return sub_seqs


def get_ngrams_subseq(subseq_dict):
    # finding duplicate values
    # from dictionary using flip
    flipped = {}

    for key, value in subseq_dict.items():
        if value not in flipped:
            flipped[value] = [key]
        else:
            flipped[value].append(key)

    sub_seq  = {}
    for key, value in flipped.items():
        sub_seq[len(value)] = key

    return sub_seq


def count_unknown_token(examples):
    tokenizer = pickle.load(open('./components/vocabulary/conala/vocab_actionbert_tokenizer', 'rb'))

    unknown_token = 0

    for x in examples:
        indexed_tokens = tokenizer.convert_tokens_to_ids(eval(x))
        # print(indexed_tokens)
        token_ids = Counter(indexed_tokens)

        unknown_token += token_ids[100]

    return unknown_token


def get_text(dataset, name, mode):
    if mode == 'conala':
        with open('./{}.txt'.format(name), 'w') as f:
            for sentence in dataset:
                if sentence['rewritten_intent'] != None:
                    f.write('<s> ' + '%s' % sentence['rewritten_intent'] + '</s> \n')
                else:
                    f.write('<s> ' + '%s' % sentence['intent'] + '</s> \n')
    elif mode == 'codesearchnet':
        with open('./{}.txt'.format(name), 'w') as f:
            for sentence in dataset.values:
                sentence = '<s> ' + ' '.join(eval(sentence[0])) + ' </s>'
                f.write(sentence + '\n')
    else:
        asdl_text = open('./asdl/grammar.txt').read()
        grammar, _, _ = Grammar.from_text(asdl_text)
        act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
        assert (len(grammar) == len(act_list))
        Reduce = ReduceAction('Reduce')
        act_dict = dict([(act.label, act) for act in act_list])
        act_dict[Reduce.label] = Reduce
        total_len_examples = 0
        with open('./{}.txt'.format(name), 'w') as f:
            for sentence in dataset['snippet_actions'].values:
                sentence = [action if action in act_dict else 'id' for action in eval(sentence)]
                total_len_examples += len(sentence)

                sentence = '<s> ' + ' '.join(sentence) + ' </s>'
                f.write(sentence + '\n')
        print('total_len_examples', total_len_examples)


def get_vocab(dataset, name, mode):
    if mode == 'conala':
        with open('./{}.txt'.format(name), 'w') as f:
            for sentence in dataset:
                if sentence['rewritten_intent'] != None:
                    for word in sentence['rewritten_intent'].split():
                        f.write('%s' % word + '\n')
                else:
                    for word in sentence['intent'].split():
                        f.write('%s' % word + '\n')
    elif mode == 'codesearchnet':
        with open('./{}.txt'.format(name), 'w') as f:
            for sentence in dataset.values:
                for word in eval(sentence[0]):
                    f.write('%s' % word + '\n')
    else:
        asdl_text = open('./asdl/grammar.txt').read()
        grammar, _, _ = Grammar.from_text(asdl_text)
        act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
        assert (len(grammar) == len(act_list))
        Reduce = ReduceAction('Reduce')
        act_dict = dict([(act.label, act) for act in act_list])
        act_dict[Reduce.label] = Reduce
        with open('./{}.txt'.format(name), 'w') as f:
            for sentence in dataset['snippet_actions']:
                for action in eval(sentence):
                    if action in act_dict:
                        f.write(action + '\n')
                    else:
                        f.write('id' + '\n')


def accuracy_score(predicted, ground_truth):
    score = 0
    for i in range(len(predicted)):
        # if [str(predicted[i][iteration]) for iteration in range(len(predicted[i]))] == [str(ground_truth[i][0][iteration]) for iteration in range(len(ground_truth[i][0]))]:
        if predicted[i] == ground_truth[i][0]:
            score += 1
            print('*'*100)
        # else:
        #     print('notok', predicted[i])
        #     print(ground_truth[i][0])
        print(predicted[i])
        print(ground_truth[i][0])
    return score/len(predicted)


def plot_zipf_map(dataset, subseq_length):

    seq_actions = dataset['snippet']

    # Determine same occurrences of subseq of length subseq_length
    sub_seq_duplicates = common_subseq(seq_actions, subseq_length)

    # Count number of subseq
    sub_seq = get_ngrams_subseq(sub_seq_duplicates)

    # reorder sub_seq
    sub_seq = sorted(sub_seq.items())

    y, x = zip(*sub_seq)

    plt.yscale("log")

    plt.title("Zipf law for subsequences length of {}".format(subseq_length))
    plt.xlabel("Rang de la sous séquence de taille {}".format(subseq_length))
    plt.ylabel("Fréquence de la sous séquence de taille {}".format(subseq_length))

    plt.scatter(x, y, marker='s', s=10, color='red')
    plt.show()

    plt.close('all')


class GridSearch:
    """ This generates all the possible experiments specified by a yaml config file """

    def __init__(self, yamlparams):

        self.HP = yamlparams

    def generate_setup(self):

        setuplist = []  # init
        K = list(self.HP.keys())
        for key in K:
            value = self.HP[key]
            if type(value) is list:
                if setuplist:
                    setuplist = [elt + [V] for elt in setuplist for V in value]
                else:
                    setuplist = [[V] for V in value]
            else:
                for elt in setuplist:
                    elt.append(value)
        print('#%d' % (len(setuplist)), 'runs to be performed')

        for setup in setuplist:
            yield dict(zip(K, setup))

def terminal_type(terminal):
    try:
        a = float(terminal)
    except (TypeError, ValueError, OverflowError):
        try:
            a = ast.literal_eval(terminal)
            return a
        except:
            a = ast.literal_eval(terminal)
            return eval(a)
            #return str(terminal)
    else:
        try:
            b = int(a)
        except (TypeError, ValueError, OverflowError):
            return a
        else:
            return b

if __name__ == '__main__':
    args = init_arg_parser()
    params = yaml.load(open(args.config_file).read(), Loader=yaml.FullLoader)

    params = params['experiment_env']

    train_set = pd.read_csv(args.train_path_conala + 'conala-train.csv')
    dev_set = pd.read_csv(args.train_path_conala + 'conala-val.csv')
    test_set = pd.read_csv(args.test_path_conala + 'conala-test.csv')

    pydf = pd.concat([train_set, dev_set, test_set])

    #  train_set = pd.read_csv(args.train_path_codesearchnet + 'train.csv')
    #
    # subseq_length = 7
    #
    # sub_seq_duplicates = common_subseq(train_set['snippet_actions'], subseq_length)
    #
    # seq = train_set['snippet']
    #
    # print('number of independent subseq in dataset', len(sub_seq_duplicates))  # 23000 conala;
    # print('number of independent subseq > 2 in dataset', len({key: val for key, val in sub_seq_duplicates.items() if val >= 3}))  # 2600 conala
    #
    # plot_zipf_map(train_set, subseq_length)
    #
    # nl = train_set['intent']
    #
    # count_unknown = count_unknown_token(nl)
    #
    # print('nombre de mots inconnus:', count_unknown)
    #
    # conala_path_train = args.train_path_conala
    # conala_path_test = args.test_path_conala
    #
    # train = pd.read_csv(conala_path_train + 'conala-train.csv')
    # dev = pd.read_csv(conala_path_train + 'conala-val.csv')
    # test = pd.read_csv(conala_path_test + 'conala-test.csv')
    #
    # get_text(train, 'conala_intent_train', mode='seq')
    # get_text(dev, 'conala_intent_heldout', mode='seq')
    # get_text(test, 'conala_intent_test', mode='seq')
    # get_vocab(train, 'conala_intent_vocab', mode='seq')
    #
    # conala_path_train = args.raw_path_conala + 'conala-train.json'
    # conala_path_test = args.raw_path_conala + 'conala-test.json'
    #
    # conala = json.load(open(conala_path_train))
    #
    # conala_dev = conala[:200]
    # conala_train = conala[200:]
    # conala_test = json.load(open(conala_path_test))

    # get_text(conala_train, 'conala_intent_train', mode='conala')
    # get_text(conala_dev, 'conala_intent_heldout', mode='conala')
    # get_text(conala_test, 'conala_intent_test', mode='conala')
    # get_vocab(conala, 'conala_intent_vocab', mode='conala')
    #
    # codesearchnet_path = args.train_path_codesearchnet
    # codesearchnet = pd.read_csv(codesearchnet_path + 'actions.csv')
    #
    # codesearchnet_dev = codesearchnet.iloc[:200]
    # codesearchnet_train = codesearchnet.iloc[200:3000]
    # codesearchnet_test = codesearchnet.iloc[3000:]
    #
    # get_text(codesearchnet_train, 'codesearchnet_intent_train', mode='seq')
    # get_text(codesearchnet_dev, 'codesearchnet_intent_heldout', mode='seq')
    # get_text(codesearchnet_test, 'codesearchnet_intent_test', mode='seq')
    # get_vocab(codesearchnet, 'codesearchnet_intent_vocab', mode='seq')

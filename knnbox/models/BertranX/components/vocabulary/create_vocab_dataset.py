import os
import pickle
import sys
from pathlib import Path

from transformers import BertTokenizer

from utils import GridSearch
from .vocab import VocabEntry, Vocab


def create_vocab(dataset, act_dict, params):

    gridsearch = GridSearch(params)

    for params in gridsearch.generate_setup():

        # Create folder for configuration
        path_folder_config = './outputs/{0}.dataset_{1}._word_freq_{2}.nl_embed_{3}.action_embed_{4}.att_size_{5}.hidden_size_{6}.epochs_{7}.dropout_enc_{8}.dropout_dec_{9}.batch_size{10}.parent_feeding_type_{11}.parent_feeding_field_{12}.change_term_name_{13}.seed_{14}/'.format(
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

        Path(path_folder_config).mkdir(parents=True, exist_ok=True)

        NL_vocab = VocabEntry()
        primitive_vocab = VocabEntry()
        action_vocab = VocabEntry()
        list_words = []
        if params['model'] == 'seq2seq':
            for word_list in dataset['intent']:
                list_words.append(eval(word_list))
            NL_vocab.add_tokens(list_words, params['word_freq'])
        elif params['model'] == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            #for word_list in dataset['intent']:
            #    list_words.append(eval(word_list))
            # tokenizer.add_tokens(list_words)
            tokenizer.add_tokens(
                ['var_0', 'str_0', 'var_1', 'str_1', 'var_2', 'str_2', 'var_3', 'str_3', 'var_4', 'str_4', 'var_5',
                 'str_5',
                 'var_6', 'str_6', 'None', 'True', 'False'])
            NL_vocab.add_bert(tokenizer.get_vocab())
            pickle.dump(tokenizer, open(os.path.join(path_folder_config + 'bert_tokenizer'), 'wb'))
        list_primitives = []
        for action_list in dataset['snippet_actions']:
            list_primitives.append([action for action in eval(action_list) if action not in act_dict and action != 'Reduce'])
        primitive_vocab.add_tokens(list_primitives, params['word_freq'])
        for action in act_dict.keys():
            action_vocab.add(action)
        vocab = Vocab(source=NL_vocab, primitive=primitive_vocab, code=action_vocab)
        print('generated vocabulary %s' % repr(vocab), file=sys.stderr)
        pickle.dump(vocab, open(os.path.join(path_folder_config + 'vocab'), 'wb'))


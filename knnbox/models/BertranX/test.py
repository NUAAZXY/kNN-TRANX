import csv
import pickle

import torch

from knnbox.models.BertranX.model.nl2code import nl2code
from knnbox.models.BertranX.utils import evaluate_action


def test(test_set, gridsearch, args, map_location, act_dict, grammar, primitives_type, device, is_cuda):
    model_score = []

    for params in gridsearch.generate_setup():

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

        vocab = pickle.load(open(path_folder_config + 'vocab', 'rb'))

        model = nl2code(params, act_dict, vocab, grammar, primitives_type, device, path_folder_config)

        print(path_folder_config)

        model.to(device)

        model.load_state_dict(
            torch.load(
                path_folder_config + 'model.pt',
                map_location=map_location

            ),
        )

        model.eval()

        BLEU, accuracy, decode_results = evaluate_action(test_set.examples, model, act_dict, params['metric'],
                                                         is_cuda=is_cuda,
                                                         return_decode_result=True)

        # print(f'BLEU score test_set = {BLEU}')
        if params['metric'] == 'BLEU':
            print('{0} test_set = {1}'.format(params['metric'], BLEU))
            print('accuracy metric value:', accuracy)

            with open(path_folder_config + '{0}.with_model_{1}&beam={2}.{3}={4}.csv'.format(params['dataset'],
                                                                                            params['beam_size'],
                                                                                            params['model'],
                                                                                            params['metric'], BLEU),
                      'w', newline='') as myfile:
                wr = csv.writer(myfile, delimiter='\n')
                wr.writerow(decode_results)

            with open(path_folder_config + '{0}.with_model_{1}&beam={2}.accuracy={3}.csv'.format(params['dataset'],
                                                                                                 params['beam_size'],
                                                                                                 params['model'],
                                                                                                 accuracy), 'w',
                      newline='') as myfile:
                wr = csv.writer(myfile, delimiter='\n')
                wr.writerow(decode_results)

        else:
            print('{0} test_set = {1}'.format(params['metric'], accuracy))
            print('BLEU metric value:', BLEU)

            with open(path_folder_config + '{0}.with_model_{1}&beam={2}.accuracy={3}.csv'.format(params['dataset'],
                                                                                                 params['model'],
                                                                                                 params['beam_size'],
                                                                                                 accuracy), 'w',
                      newline='') as myfile:
                wr = csv.writer(myfile, delimiter='\n')
                wr.writerow(decode_results)

            with open(path_folder_config + '{0}.with_model_{1}&beam={2}.BLEU={3}.csv'.format(params['dataset'],
                                                                                             params['model'],
                                                                                             params['beam_size'], BLEU),
                      'w', newline='') as myfile:
                wr = csv.writer(myfile, delimiter='\n')
                wr.writerow(decode_results)

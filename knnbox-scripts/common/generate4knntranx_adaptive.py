r"""
This file is copied from fairseq_cli/generate.py
knnbox make slight change about parsing args so that is
can parse the arch specified on the cli instead of directly
using the arch inside checkpoint.
"""

import ast
import logging
import math
import os
import sys
from itertools import chain

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
# sys.path.append('/home/user/PycharmProjects/knn-box-master')
from knnbox.models.BertranX.dataset.dataset import Dataset, Batch
from knnbox.models.BertranX.build_data import build_data
from knnbox.models.knntranx_adaptive import KNNTRANX_adaptive
import pandas as pd
from knnbox.models.BertranX.utils import evaluate_action
import csv
torch.set_printoptions(profile="full")
print(sys.path)
def main(args, override_args):
    assert args.path is not None, "--path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(
            args.results_path, "generate-{}.txt".format(args.gen_subset)
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(args, h)
    else:
        return _main(args, override_args, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(args, override_args, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(args)

    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)
    # random.seed(args.seed)


    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    #test_epoch_itr = test_set.batch_iter(1)
    knn_type = args.arch.split("@")[0]
    # knn_type = 'knntranx_adaptive_django'
    knn_type = 'knntranx_adaptive'

    # knn_type = 'knntranx_adaptive'
    params, gridsearch, map_location, act_dict, grammar, primitives_type, device, is_cuda, path_folder_config, vocab = build_data(knn_type)
    # params = gridsearch.generate_setup()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(args)
    model = KNNTRANX_adaptive(args, params, act_dict, vocab, grammar, primitives_type, device, path_folder_config)
    model.load_state_dict(
        torch.load(
            path_folder_config + '/model.pt',
            map_location=map_location
        ),
        strict=False
    )
    if knn_type == 'knntranx_adaptive':
        test_set = Dataset(
            pd.read_csv('knnbox/models/BertranX/dataset/data_conala/test/' + 'conala-test.csv'))
    else:
        test_set = Dataset(pd.read_csv('knnbox/models/BertranX/dataset/data_django/' + 'test.csv'))
    # print(vocab.code.id2word[50], vocab.code.id2word[60])
    model.to(device)
    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))

    # Initialize generator
    gen_timer = StopwatchMeter()
    num_sentences = 0
    wps_meter = TimeMeter()
    model.eval()
    BLEU, accuracy, decode_results = evaluate_action(test_set.examples, model, act_dict, params['metric'],
                                                     is_cuda=is_cuda,
                                                     return_decode_result=True)

    print(BLEU, accuracy)
    print('{0} test_set = {1}'.format(params['metric'], BLEU))
    print('accuracy metric value:', accuracy)

    with open(path_folder_config + '{0}.with_model_{1}&beam={2}.{3}={4}.csv'.format(params['dataset'],
                                                                                    params['beam_size'],
                                                                                    params['model'], params['metric'],
                                                                                    BLEU), 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter='\n')
        wr.writerow(decode_results)

    with open(path_folder_config + '{0}.with_model_{1}&beam={2}.accuracy={3}.csv'.format(params['dataset'],
                                                                                         params['beam_size'],
                                                                                         params['model'], accuracy),
              'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter='\n')
        wr.writerow(decode_results)
    logger.info(
        "Translated {} sentences ({} tokens) in {:.1f}s ".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
        )
    )

def get_knn_generation_parser(interactive=False, default_task="translation"):
    parser = options.get_parser("Generation", default_task)
    options.add_dataset_args(parser, gen=True)
    options.add_distributed_training_args(parser, default_world_size=1)
    ## knnbox related code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # compared to options.get_generation_parser(..), knnbox only add one line code below 
    options.add_model_args(parser)
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
    options.add_generation_args(parser)
    if interactive:
        options.add_interactive_args(parser)
    return parser


def cli_main():
    parser = get_knn_generation_parser()
    #parser.add_argument("--knn-mode", choices=["build_datastore", "train_metak", "inference"],
    #                   help="choose the action mode")
    #parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
    #                    help="the directory of save or load datastore")
    #parser.add_argument("--knn-max-k", type=int, metavar="N", default=8,
    #                    help="The hyper-parameter max k of adaptive knn-mt")
    #parser.add_argument("--knn-k-type", choices=["fixed", "trainable"], default="trainable",
    #                    help="trainable k or fixed k, if choose `fixed`, we use all the"
    #                         "entries returned by retriever to calculate knn probs, "
    #                         "i.e. directly use --knn-max-k as k")
    #parser.add_argument("--knn-lambda-type", choices=["fixed", "trainable"], default="trainable",
    #                    help="trainable lambda or fixed lambda")
    #parser.add_argument("--knn-lambda", type=float, default=0.7,
    #                    help="if use a fixed lambda, provide it with --knn-lambda")
    #parser.add_argument("--knn-temperature-type", choices=["fixed", "trainable"], default="trainable",
    #                    help="trainable temperature or fixed temperature")
    #parser.add_argument("--knn-temperature", type=float, default=10,
    #                    help="if use a fixed temperature, provide it with --knn-temperature")
    #parser.add_argument("--knn-combiner-path", type=str, metavar="STR", default="/home/",
    #                    help="The directory to save/load adaptiveCombiner")
    #parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
    #                    help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
    args = options.parse_args_and_arch(parser)

    # override_parser = get_knn_generation_parser()
    # override_parser.add_argument("--knn-mode", choices=["build_datastore", "train_metak", "inference"],
    #                              help="choose the action mode")
    # override_parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
    #                              help="the directory of save or load datastore")
    # override_parser.add_argument("--knn-max-k", type=int, metavar="N", default=8,
    #                              help="The hyper-parameter max k of adaptive knn-mt")
    # override_parser.add_argument("--knn-k-type", choices=["fixed", "trainable"], default="trainable",
    #                              help="trainable k or fixed k, if choose `fixed`, we use all the"
    #                                   "entries returned by retriever to calculate knn probs, "
    #                                   "i.e. directly use --knn-max-k as k")
    # override_parser.add_argument("--knn-lambda-type", choices=["fixed", "trainable"], default="trainable",
    #                              help="trainable lambda or fixed lambda")
    # override_parser.add_argument("--knn-lambda", type=float, default=0.7,
    #                              help="if use a fixed lambda, provide it with --knn-lambda")
    # override_parser.add_argument("--knn-temperature-type", choices=["fixed", "trainable"], default="trainable",
    #                              help="trainable temperature or fixed temperature")
    # override_parser.add_argument("--knn-temperature", type=float, default=10,
    #                              help="if use a fixed temperature, provide it with --knn-temperature")
    # override_parser.add_argument("--knn-combiner-path", type=str, metavar="STR", default="/home/",
    #                              help="The directory to save/load adaptiveCombiner")
    # override_parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
    #                              help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
    # override_args = options.parse_args_and_arch(override_parser)
    override_args = None
    main(args, override_args)

if __name__ == "__main__":
    cli_main()

r"""
This file is copied from fairseq_cli/validate.py.
knnbox made 2 major changes:

change 1. We modified the part of parsing args so that is
can parse the arch specified on the cli instead of directly
using the arch inside checkpoint.

change 2. we add codes about `saving datastore vals`, `dump datastore`, etc.
"""
import traceback
import logging
import os
import sys
from itertools import chain
import pandas as pd
import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
# from transformers import BertTokenizer

## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from knnbox.datastore import Datastore, GreedyMergeDatastore, PckDatastore
from knnbox.common_utils import filter_pad_tokens, global_vars
import numpy as np
from knnbox.models.BertranX.dataset.dataset import Dataset, Batch
from knnbox.models.BertranX.build_data import build_data
from knnbox.models.knntranx import KNNTRANX
from knnbox.models.knntranx_adaptive import KNNTRANX_adaptive
# from knnbox.models.BertranX.build_data import build_data, get_kv
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(args, override_args=None):
    global pri_idx
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None
    knn_type = args.arch.split("@")[0]
    knn_type = 'knntranx_adaptive'
    # knn_type = 'knntranx_adaptive_django'


    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    if knn_type == "knntranx":
        params, gridsearch, map_location, act_dict, grammar, primitives_type, device, is_cuda, path_folder_config, vocab = build_data(knn_type)
        # params = gridsearch.generate_setup()
        model = KNNTRANX(args, params, act_dict, vocab, grammar, primitives_type, device, path_folder_config)
        model.load_state_dict(
            torch.load(
                path_folder_config + 'model.pt',
                map_location=map_location
            ),
        )
        model.to('cuda:0')
    elif knn_type == "knntranx_adaptive" or knn_type == "knntranx_adaptive_django":
        params, gridsearch, map_location, act_dict, grammar, primitives_type, device, is_cuda, path_folder_config, vocab = build_data(knn_type)
        # params = gridsearch.generate_setup()
        model = KNNTRANX_adaptive(args, params, act_dict, vocab, grammar, primitives_type, device, path_folder_config)
        model.load_state_dict(
            torch.load(
                path_folder_config + 'model.pt',
                map_location=map_location
            ),
        )
        model.to('cuda:0')
    else:
        models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
            [args.path],
            arg_overrides=overrides,
            suffix=getattr(args, "checkpoint_suffix", ""),
        )
        model = models[0]

    # Move models to GPU
    if knn_type == 'knntranx' or knn_type == "knntranx_adaptive" or knn_type == "knntranx_adaptive_django":
        # model.half()
        pass
        # model.cuda()
    else:
        for model in models:
            if use_fp16:
                model.half()
            if use_cuda:
                model.cuda()

        # Print args
        logger.info(model_args)

        # Build criterion
        criterion = task.build_criterion(model_args)
        criterion.eval()


    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if "datastore" not in global_vars():
        # create suitable datastore class if not exists
        if knn_type in ["vanilla_knn_mt", "adaptive_knn_mt", "kernel_smoothed_knn_mt", "vanilla_knn_mt_visual", "robust_knn_mt"]:
            global_vars()["datastore"] = Datastore(path=args.knn_datastore_path)
        if knn_type == "greedy_merge_knn_mt":
            global_vars()["datastore"] = GreedyMergeDatastore(path=args.knn_datastore_path)
        if knn_type == "pck_knn_mt":
            global_vars()["datastore"] = PckDatastore(
                path=args.knn_datastore_path,
                reduction_network_input_dim=args.decoder_embed_dim,
                reduction_network_output_dim=args.knn_reduct_dim,
                dictionary_len=len(task.tgt_dict),
                )
        if knn_type == "knntranx":
            print(vocab)
            global_vars()["datastore"] = PckDatastore(
                path=args.knn_datastore_path,
                reduction_network_input_dim=args.decoder_embed_dim,
                reduction_network_output_dim=args.knn_reduct_dim,
                dictionary_len=len(vocab.primitive)+len(vocab.code),
                )
        if "datastore_act" not in global_vars():
            if knn_type == "knntranx_adaptive":
                print(vocab)
                global_vars()["datastore_act"] = Datastore(
                    path=args.knn_datastore_path+'/act',
                    )
                global_vars()["datastore_pri"] = Datastore(
                    path=args.knn_datastore_path+'/pri',
                )
    if knn_type == 'knntranx_adaptive':
        datastore_act = global_vars()["datastore_act"]
        datastore_pri = global_vars()["datastore_pri"]
    else:
        datastore = global_vars()["datastore"]
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end

    for subset in args.valid_subset.split(","):
        if knn_type == 'knntranx' or knn_type == "knntranx_adaptive" or knn_type == "knntranx_adaptive_django":
            if knn_type == 'knntranx_adaptive':
                dataset = Dataset(pd.read_csv('knnbox/models/BertranX/dataset/data_conala/train/' + 'conala-train.csv'))
            else:
                dataset = Dataset(pd.read_csv('knnbox/models/BertranX/dataset/data_django/' + 'train.csv'))

        else:
            try:
                task.load_dataset(subset, combine=False, epoch=1)
                dataset = task.dataset(subset)
            except KeyError:
                raise Exception("Cannot find dataset: " + subset)
        if knn_type == 'knntranx' or knn_type == "knntranx_adaptive" or knn_type == "knntranx_adaptive_django":
            progress = progress_bar.progress_bar(
            dataset.batch_iter(1),
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            # Initialize data iterator
            itr = task.get_batch_iterator(
                dataset=dataset,
                max_tokens=args.max_tokens,
                max_sentences=args.batch_size,
                max_positions=utils.resolve_max_positions(
                    task.max_positions(),
                    *[m.max_positions() for m in models],
                ),
                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=args.required_batch_size_multiple,
                seed=args.seed,
                num_shards=args.distributed_world_size,
                shard_id=args.distributed_rank,
                num_workers=args.num_workers,
                data_buffer_size=args.data_buffer_size,
            ).next_epoch_itr(shuffle=False)
            progress = progress_bar.progress_bar(
                itr,
                log_format=args.log_format,
                log_interval=args.log_interval,
                prefix=f"valid on '{subset}' subset",
                default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
            )

        log_outputs = []
        mismatch_count = 0
        mismatch_index = []
        # final_target = Memmap(filename=os.path.join(args.knn_datastore_path, knn_type+".npy"), mode="w+")

        for i, sample in enumerate(progress):
            try:
                sample = utils.move_to_cuda(sample) if use_cuda else sample

                ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if knn_type in ["vanilla_knn_mt", "adaptive_knn_mt", "greedy_merge_knn_mt", "kernel_smoothed_knn_mt", "plac_knn_mt", "robust_knn_mt"]:
                    non_pad_tokens, mask = filter_pad_tokens(sample["target"])
                    datastore["vals"].add(non_pad_tokens)
                    datastore.set_pad_mask(mask)

                elif knn_type == "pck_knn_mt":
                    # print(sample)
                    non_pad_tokens, mask = filter_pad_tokens(sample["target"])
                    datastore["vals"].add(non_pad_tokens)
                    datastore.set_pad_mask(mask)
                    datastore.set_target(sample["target"])
                    # print(non_pad_tokens, mask, sample['target'])

                elif knn_type == "knntranx":
                    # batch = Batch(sample, args, act_dict, vocab, device, grammar)
                    sample_code = eval(sample[0].snippet_actions)
                    # print(sample[0].snippet_actions)
                    data = []
                    for a_tm1 in sample_code:
                        if a_tm1 in act_dict:
                            a_tm1_embed = vocab.code[a_tm1]
                        else:
                            a_tm1_embed = vocab.primitive[a_tm1]
                        data.append(a_tm1_embed)
                    data = torch.tensor(data, device=device)
                    non_pad_tokens, mask = filter_pad_tokens(data)
                    # datastore["vals"].add(non_pad_tokens)
                    datastore.set_pad_mask(mask)
                    datastore.set_target(data)
                elif knn_type == "knntranx_adaptive" or knn_type == "knntranx_adaptive_django":
                    sample_code = eval(sample[0].snippet_actions)
                    # print(sample[0].snippet_actions)
                    act = []
                    pri = []
                    act_idx = []
                    pri_idx = []
                    for idx, a_tm1 in enumerate(sample_code):
                        if a_tm1 in act_dict:
                            a_tm1_embed = vocab.code[a_tm1]
                            act.append(a_tm1_embed)
                            act_idx.append(idx)
                        else:
                            a_tm1_embed = vocab.primitive[a_tm1]
                            # print(a_tm1, a_tm1_embed)
                            pri.append(a_tm1_embed)
                            pri_idx.append(idx)
                        # data.append(a_tm1_embed)
                    act = torch.tensor(act)
                    pri = torch.tensor(pri)
                    act_idx = torch.tensor(act_idx)
                    pri_idx = torch.tensor(pri_idx)
                    non_pad_tokens_act, mask_act = filter_pad_tokens(act)
                    datastore_act.set_pad_mask(mask_act)
                    non_pad_tokens_pri, mask_pri = filter_pad_tokens(pri)
                    datastore_pri.set_pad_mask(mask_pri)
                elif knn_type == "vanilla_knn_mt_visual":
                    non_pad_tokens, mask = filter_pad_tokens(sample["target"])
                    datastore["vals"].add(non_pad_tokens)
                    datastore.set_pad_mask(mask)
                    # get the key-value pair related sentence_ids
                    target_len = mask.sum(dim=-1)
                    sentence_ids = []
                    for idx, sentence_id in enumerate(sample["id"].cpu().numpy()):
                        sentence_ids += [sentence_id]*target_len[idx]
                    sentence_ids = np.array(sentence_ids, dtype=int)
                    # get the key-value pair related token_postions
                    token_positions = []
                    for len_ in target_len:
                        token_positions += [i for i in range(len_)]
                    token_positions = np.array(token_positions, dtype=int)
                    # add them to datastore
                    datastore["sentence_ids"].add(sentence_ids)
                    datastore["token_positions"].add(token_positions)
                ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
                # print(datastore.datas)
                if knn_type == 'knntranx':
                    # print(sample[0])
                    # from knnbox.models.BertranX.utils import decode_action
                    # print('*' * 100)
                    # print(decode_action(sample[0].intent, act_dict, model, True))
                    x = model(sample)
                    datastore["vals"].add(non_pad_tokens)
                    # print(datastore["vals"].shape)
                    # print(sample[0].snippet_tokens)
                    # print(x)
                elif knn_type == 'knntranx_adaptive'or knn_type == "knntranx_adaptive_django":
                    x = model(sample, act_idx = act_idx, pri_idx = pri_idx, step = i)
                    if i < 7174000:
                        datastore_act["vals"].add(non_pad_tokens_act)
                    datastore_pri["vals"].add(non_pad_tokens_pri)

                else:
                    _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
                # print(datastore.datas)
                    progress.log(log_output, step=i)
                    log_outputs.append(log_output)
            except Exception as e:
                print(i, e)
                # traceback.print_exc()
                mismatch_count += 1
                mismatch_index.append(i)
                # print(datastore["vals"].shape)
            # if i == 10:
            #     break

        print(mismatch_count)
        if knn_type != 'knntranx' and knn_type != "knntranx_adaptive" and knn_type != "knntranx_adaptive_django":
            if args.distributed_world_size > 1:
                log_outputs = distributed_utils.all_gather_list(
                    log_outputs,
                    max_size=getattr(args, "all_gather_list_size", 16384),
                )
                log_outputs = list(chain.from_iterable(log_outputs))

            with metrics.aggregate() as agg:
                task.reduce_metrics(log_outputs, criterion)
                log_output = agg.get_smoothed_values()

            progress.print(log_output, tag=subset, step=i)

    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # release memory to make sure we have enough gpu memory to build faiss index
            del model, task, progress, criterion, dataset
    if use_cuda:
        torch.cuda.empty_cache()    # release gpu memory

    if knn_type in ["vanilla_knn_mt", "adaptive_knn_mt", "kernel_smoothed_knn_mt", "vanilla_knn_mt_visual", "plac_knn_mt", "robust_knn_mt"]:
        datastore.dump()    # dump to disk
        datastore.build_faiss_index("keys", use_gpu=(not args.build_faiss_index_with_cpu))   # build faiss index
    elif knn_type == "greedy_merge_knn_mt":
        datastore.dump() # dump the un-pruned datastore to disk
        datastore.build_faiss_index("keys", do_pca=args.do_pca, pca_dim=args.pca_dim, use_gpu=(not args.build_faiss_index_with_cpu)) # build faiss index with pre-PCA operation for pruned datastore
        if args.do_merge:
            datastore.prune(merge_neighbors=args.merge_neighbors_n) # prune the datastore. search n neighbors when do greedy merge
            datastore.dump() # dump the pruned datastore to disk
            datastore.build_faiss_index("keys", do_pca=args.do_pca, pca_dim=args.pca_dim, use_gpu=(not args.build_faiss_index_with_cpu)) # build faiss index for un-pruned datastore

    elif knn_type == "pck_knn_mt":
        datastore.dump() # dump the un-pruned datastore to disk
    elif knn_type == "knntranx":
        datastore.dump() # dump the un-pruned datastore to disk
    elif knn_type == "knntranx_adaptive" or knn_type == "knntranx_adaptive_django":
        datastore_act.dump()  # dump to disk
        datastore_act.build_faiss_index("keys", use_gpu=(not args.build_faiss_index_with_cpu))
        datastore_pri.dump()  # dump to disk
        datastore_pri.build_faiss_index("keys", use_gpu=(not args.build_faiss_index_with_cpu))

    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end


## knnbox code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_build_datastore_parser(default_task=None):
    r"""
    very similar to options.get_validation_parser() but parse arch as well.

    Difference:
    - when validate, we don't need to specify --arch and model args, because they are
    recorded in .pt file.

    - when building datastore, we need to load the saved model parameter to a knn-mt arch,
    which is different from the checkpoint original arch.
    For example, I have a nmt checkpoint with arch `transformer_iwslt_de_en`, and now I want to
    load it's parameter to arch `vanilla@transformer_iwslt_de_en`, I must specify
    arch = "vanilla@transfromer_iwslt_de_en".
    """
    parser = options.get_parser("Validation", default_task)
    options.add_dataset_args(parser, train=True)
    options.add_distributed_training_args(parser, default_world_size=1)
    # knnbox add one line below to parse arch
    options.add_model_args(parser)
    group = parser.add_argument_group("Evaluation")
    from fairseq.dataclass.data_class import CommonEvalParams
    options.gen_parser_from_dataclass(group, CommonEvalParams())
    return parser
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end

def cli_main():
    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # parser = options.get_validation_parser()
    parser = get_build_datastore_parser()
    args = options.parse_args_and_arch(parser)
    # args.add_argument("--dataset", default='conala', help="the name of dataset")

    ## only override args that are explicitly given on the command line
    # override_parser = options.get_validation_parser()
    override_parser = get_build_datastore_parser()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)
    # print(args)
    distributed_utils.call_main(args, main, override_args=override_args)



if __name__ == "__main__":
    cli_main()



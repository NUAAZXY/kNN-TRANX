#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import random
import sys

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from knnbox.models.BertranX.dataset.dataset import Dataset, Batch
from knnbox.models.BertranX.build_data import build_data
from knnbox.models.knntranx_adaptive import KNNTRANX_adaptive
import pandas as pd
from transformers import AdamW
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")
torch.set_printoptions(profile="full")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.autograd.set_detect_anomaly(True)
def main(args):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    metrics.reset()


    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)
    knn_type = args.arch.split("@")[0]
    # knn_type = 'knntranx_adaptive_django'
    knn_type = 'knntranx_adaptive'

    params, gridsearch, map_location, act_dict, grammar, primitives_type, device, is_cuda, path_folder_config, vocab = build_data(knn_type)
    # params = gridsearch.generate_setup()
    # device = 'cpu'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    model = KNNTRANX_adaptive(args, params, act_dict, vocab, grammar, primitives_type, device, path_folder_config)
    # path_folder_config = '/home/ubuntu/knnmt/knnbox/models/BertranX/outputs/womined_wo'
    model.load_state_dict(
        torch.load(
            path_folder_config + '/model.pt',
            map_location=map_location
        ),
        strict=False
    )
    for name, param in model.named_parameters():
        if 'combiner' not in name:
            param.requires_grad = False
    model.to(device)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    if knn_type == 'knntranx_adaptive':
        train_set = Dataset(
            pd.read_csv('knnbox/models/BertranX/dataset/data_conala/train/' + 'conala-val.csv'))

    else:
        train_set = Dataset(pd.read_csv('knnbox/models/BertranX/dataset/data_django/' + 'dev.csv'))

    logger.info("model: {} ({})".format(args.arch, model.__class__.__name__))
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )


    # Build trainer
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.batch_size
        )
    )

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = args.lr[0]
    optimizer = AdamW(model.parameters(), lr=lr, eps=args.adam_eps, betas=(0.9, 0.98))
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    epoch = 0
    step_ = 0
    print('save_dir', args.save_dir)
    # s = 8000
    # s = 10000
    s = 8000
    # b = 10000
    while lr > args.min_lr and epoch <= max_epoch:
        # train for one epoch
        model.train()
        epoch += 1
        if step_ == s:
            break
        for step, batch_examples in enumerate(train_set.batch_iter(1)):

            batch_examples_temp = [e for e in batch_examples if len(eval(e.snippet_actions)) <= params['len_max']]
            optimizer.zero_grad()
            x, src = model(batch_examples_temp)
            final_prob = model.get_normalized_probs(idx=None, log_probs=False)
            act_prob = final_prob['act']
            pri_prob = final_prob['pri']
            final_prob = torch.cat([act_prob, pri_prob], dim=-1).unsqueeze(0)
            loss = model.get_loss(final_prob, batch_examples_temp, x, src)
            loss = -loss[0]
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            step_ += 1
            if step_ == s:
                break

    model.combiner_act.dump(args.save_dir+'act')
    model.combiner_pri.dump(args.save_dir+'pri')


    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_losses = [None]
    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
        or num_updates >= max_update
        or (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates >= args.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        or num_updates >= max_update
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or num_updates >= max_update
        or (
            args.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    # parser.add_argument("--knn-mode", choices=["build_datastore", "train_metak", "inference"],
    #                     help="choose the action mode")
    # parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
    #                     help="the directory of save or load datastore")
    # parser.add_argument("--knn-max-k", type=int, metavar="N", default=8,
    #                     help="The hyper-parameter max k of adaptive knn-mt")
    # parser.add_argument("--knn-k-type", choices=["fixed", "trainable"], default="trainable",
    #                     help="trainable k or fixed k, if choose `fixed`, we use all the"
    #                          "entries returned by retriever to calculate knn probs, "
    #                          "i.e. directly use --knn-max-k as k")
    # parser.add_argument("--knn-lambda-type", choices=["fixed", "trainable"], default="trainable",
    #                     help="trainable lambda or fixed lambda")
    # parser.add_argument("--knn-lambda", type=float, default=0.7,
    #                     help="if use a fixed lambda, provide it with --knn-lambda")
    # parser.add_argument("--knn-temperature-type", choices=["fixed", "trainable"], default="trainable",
    #                     help="trainable temperature or fixed temperature")
    # parser.add_argument("--knn-temperature", type=float, default=10,
    #                     help="if use a fixed temperature, provide it with --knn-temperature")
    # parser.add_argument("--knn-combiner-path", type=str, metavar="STR", default="/home/",
    #                     help="The directory to save/load adaptiveCombiner")
    # parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
    #                     help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
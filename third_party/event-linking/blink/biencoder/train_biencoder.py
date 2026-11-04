# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

import torch.nn.functional as F
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder, to_bert_input
from blink.common.ranker_base import get_model_obj
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser



logger = None

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
    reranker, eval_dataloader, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_input, _ = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input)

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        label_ids = torch.LongTensor(
                torch.arange(context_input.size(0))  # the last batch can be smaller
        ).numpy()
        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results


# Sampler training (added for resa2el). The negatives of each mention come
# from its fixed candidate pool (scripts/build_pools.py), chosen by a sampler
# from src/resa2el/samplers.py. The page encoder is frozen; the page vectors
# are the released ones.
def _samplers_module():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from resa2el import samplers
    return samplers


def _load_pools(params, tensor_data, logger):
    d = np.load(params["pools"])
    keep = np.flatnonzero(d["gold"] >= 0)
    keep = keep[keep < len(tensor_data)]  # --debug keeps the first 200 mentions
    if logger:
        logger.info("sampler {}: {} of {} mentions have the correct page in the pool file".format(
            params["sampler"], len(keep), len(tensor_data)))
    pools = {k: d[k] for k in ("ids", "scores", "gold", "gold_score", "sims")}
    encoding = np.load(params["encoding"], mmap_mode="r")
    tensors = [t[torch.from_numpy(keep)] for t in tensor_data.tensors] + [torch.from_numpy(keep)]
    return pools, encoding, TensorDataset(*tensors)


def _sampled_loss(reranker, context_input, sample_idx, pools, encoding, sampler, k, rng, device, sampler_kw):
    """Cross-entropy of the correct page against k sampled negatives."""
    ids, scores, gold, gold_score, sims = (pools[x] for x in ("ids", "scores", "gold", "gold_score", "sims"))
    cand = np.empty((len(sample_idx), k + 1), dtype=np.int64)
    for r, i in enumerate(sample_idx):
        not_gold = ids[i] != gold[i]  # the correct page is removed from the pool
        pool_ids = ids[i][not_gold]
        sel = sampler(scores[i][not_gold].astype(np.float64), float(gold_score[i]),
                      sims[i][not_gold].astype(np.float64), k, rng, **sampler_kw)
        cand[r, 0] = gold[i]
        cand[r, 1:] = pool_ids[np.asarray(sel)]
    cand_enc = torch.from_numpy(encoding[cand.reshape(-1)].astype(np.float32))
    cand_enc = cand_enc.view(len(sample_idx), k + 1, -1).to(device)
    token_idx, segment_idx, mask = to_bert_input(context_input, reranker.NULL_IDX)
    ctx, _ = reranker.model(token_idx, segment_idx, mask, None, None, None)
    logits = torch.bmm(cand_enc, ctx.unsqueeze(2)).squeeze(2)  # batch x (k + 1)
    target = torch.zeros(len(sample_idx), dtype=torch.long, device=device)
    return F.cross_entropy(logits, target, reduction="mean")


def _tensor_data(split, params, tokenizer, logger):
    """Tokenized tensors of one split. Cached on disk (added for resa2el):
    tokenizing the 66k training mentions takes about 30 minutes of CPU."""
    samples = utils.read_dataset(split, params["data_path"])
    logger.info("Read %d %s samples." % (len(samples), split))
    cache = os.path.join(params["data_path"], "cache_{}_{}_{}_{}.pt".format(
        split, params["max_context_length"], params["max_cand_length"], len(samples)))
    if not params["debug"] and os.path.exists(cache):
        tensors = torch.load(cache, weights_only=True)
        logger.info("Loaded tokenized %s data from %s" % (split, cache))
        return samples, TensorDataset(*tensors)
    _, tensor_data = data.process_mention_data(
        samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    if not params["debug"]:
        torch.save(tuple(tensor_data.tensors), cache + ".tmp")
        os.replace(cache + ".tmp", cache)
        logger.info("Saved tokenized %s data to %s" % (split, cache))
    return samples, tensor_data


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params.get("freeze_cand_encoder"):
        for p in get_model_obj(model).cand_encoder.parameters():
            p.requires_grad = False
        logger.info("cand_encoder frozen")
    sampler_name = params.get("sampler") or "in_batch"
    pools = None
    if sampler_name != "in_batch":
        S = _samplers_module()
        sampler_fn = S.SAMPLERS[sampler_name]
        sampler_kw = {}
        if sampler_name == "resa2":
            sampler_kw = dict(n_stage1=params["resa2_stage1"], n_stage2=params["resa2_stage2"], tau=params["resa2_tau"])
        sampler_rng = np.random.default_rng(params["seed"])

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    train_samples, train_tensor_data = _tensor_data("train", params, tokenizer, logger)
    if sampler_name != "in_batch":
        pools, encoding, train_tensor_data = _load_pools(params, train_tensor_data, logger)

    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    # Load eval data
    # TODO: reduce duplicated code here
    valid_samples, valid_tensor_data = _tensor_data("valid", params, tokenizer, logger)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # evaluate before training
    results = evaluate(
        reranker, valid_dataloader, params, device=device, logger=logger,
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )
    config_file = open(os.path.join(model_output_path, "training_params.json"), 'w')
    json.dump(params, config_file)

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    n_updates = 0
    max_updates = params.get("max_updates")
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            if pools is not None:
                loss = _sampled_loss(reranker, batch[0], batch[-1].cpu().numpy(), pools, encoding,
                                     sampler_fn, params["negatives"], sampler_rng, device, sampler_kw)
            else:
                context_input, candidate_input, _= batch
                loss, _ = reranker(context_input, candidate_input)

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                n_updates += 1
                if max_updates and n_updates >= max_updates:
                    logger.info("max_updates {} reached".format(max_updates))
                    break

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                evaluate(
                    reranker, valid_dataloader, params, device=device, logger=logger,
                )
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")
        if max_updates and n_updates >= max_updates:
            break

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    reranker = load_biencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()
    # Sampler training (added for resa2el)
    parser.add_argument("--sampler", default="in_batch", choices=["in_batch", "uniform", "topscore", "resa2"])
    parser.add_argument("--pools", default=None, help="<split>.npz from scripts/build_pools.py")
    parser.add_argument("--encoding", default=None, help="all_entities_fp16.npy from scripts/build_pools.py")
    parser.add_argument("--negatives", type=int, default=7, help="negatives per mention")
    parser.add_argument("--freeze_cand_encoder", action="store_true", help="train the mention encoder only")
    parser.add_argument("--max_updates", type=int, default=None, help="stop after this many optimizer steps")
    parser.add_argument("--resa2_stage1", type=int, default=500)
    parser.add_argument("--resa2_stage2", type=int, default=75)
    parser.add_argument("--resa2_tau", type=float, default=0.25)

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__

    # params['output_path'] = "models/blink_wiki"
    # params['data_path'] = "out/blink_wiki"
    main(params)
